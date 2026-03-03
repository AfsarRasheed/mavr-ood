"""
Text-Guided Detection Pipeline (Main Orchestrator)
Runs all 6 steps: Scene Agent → Attribute Agent → GroundingDINO → CLIP → Spatial → SAM
"""

import gc
import numpy as np
import torch

from src.text_guided.scene_agent import scene_understanding
from src.text_guided.attribute_agent import attribute_matching_agent
from src.text_guided.query_parser import parse_query, spatial_filter
from src.text_guided.visualizer import generate_step_visualizations


def run_text_guided_pipeline(image_np, user_prompt, image_path,
                              gdino_model, sam_predictor, clip_verifier,
                              box_threshold=0.3, clip_threshold=0.20,
                              precomputed_scene=None, precomputed_attr=None):
    """
    Run the complete text-guided detection pipeline.

    Args:
        image_np: numpy RGB image (H, W, 3)
        user_prompt: user's text query e.g. "the grey car on the left"
        image_path: path to image file (for LLaVA scene analysis)
        gdino_model: loaded GroundingDINO model
        sam_predictor: loaded SAM predictor
        clip_verifier: loaded CLIP verifier
        box_threshold: GroundingDINO confidence threshold
        clip_threshold: CLIP similarity threshold
        precomputed_scene: pre-computed scene result (skip LLaVA call 1)
        precomputed_attr: pre-computed attribute result (skip LLaVA call 2)

    Returns:
        dict with keys:
            step_images: dict of step name -> RGB numpy image
            scene_result: scene analysis JSON
            parsed_query: parsed query dict
            final_masks: SAM segmentation masks
            selected_idx: selected box indices
            summary: text summary of results
    """
    from PIL import Image as PILImage
    import groundingdino.datasets.transforms as T

    H, W = image_np.shape[:2]

    print(f"\n{'='*60}")
    print(f"TEXT-GUIDED DETECTION")
    print(f"Prompt: \"{user_prompt}\"")
    print(f"{'='*60}")

    # ---- Step 1: Scene Understanding Agent ----
    if precomputed_scene is not None:
        scene_result = precomputed_scene
        print("[OK] Using pre-computed scene analysis (LLaVA already ran)")
    else:
        scene_result = scene_understanding(image_path)

    # ---- Step 2: Attribute Matching Agent ----
    if precomputed_attr is not None:
        attr_result = precomputed_attr
        print("[OK] Using pre-computed attribute matching (LLaVA already ran)")
    else:
        attr_result = attribute_matching_agent(image_path, scene_result, user_prompt)

    # ---- FREE LLaVA from GPU to make room for detection models ----
    try:
        import src.agents.vlm_backend as vlm_mod
        if hasattr(vlm_mod, '_model') and vlm_mod._model is not None:
            del vlm_mod._model
            vlm_mod._model = None
        if hasattr(vlm_mod, '_processor') and vlm_mod._processor is not None:
            del vlm_mod._processor
            vlm_mod._processor = None
        gc.collect()
        torch.cuda.empty_cache()
        print("[OK] LLaVA freed from GPU memory")
    except Exception as e:
        print(f"[WARN] Could not free LLaVA: {e}")
        gc.collect()
        torch.cuda.empty_cache()

    # Parse query (internal helper)
    parsed = parse_query(user_prompt)
    parsed['attr_agent_result'] = attr_result

    # If agent recommended a better prompt, use it (but validate it)
    if isinstance(attr_result, dict) and attr_result.get('recommended_prompt'):
        agent_prompt = attr_result['recommended_prompt'].strip()
        # Reject if it looks like template text (LLaVA sometimes copies the template)
        bad_keywords = ['groundingdino', 'optimized', 'detection prompt', 'example', 'template']
        is_template = any(kw in agent_prompt.lower() for kw in bad_keywords)
        if agent_prompt and len(agent_prompt) > 2 and not is_template:
            print(f"[i] Using agent's recommended prompt: '{agent_prompt}'")
            parsed['object_prompt'] = agent_prompt
        else:
            print(f"[i] Agent prompt rejected (template text), using parsed: '{parsed['object_prompt']}'")

    # ---- Step 3: Candidate Detection (GroundingDINO) ----
    print(f"[i] Running GroundingDINO with prompt: '{parsed['object_prompt']}'")

    image_pil = PILImage.fromarray(image_np)
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)

    device = next(gdino_model.parameters()).device

    caption = parsed['object_prompt'].lower().strip()
    if not caption.endswith("."):
        caption += "."

    image_tensor_dev = image_tensor.to(device)
    with torch.no_grad():
        outputs = gdino_model(image_tensor_dev[None], captions=[caption])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes_cxcywh = outputs["pred_boxes"].cpu()[0]

    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes_cxcywh[filt_mask]

    tokenizer = gdino_model.tokenizer
    tokenized = tokenizer(caption)

    from groundingdino.util.utils import get_phrases_from_posmap

    all_labels = []
    all_det_scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > 0.25, tokenized, tokenizer)
        score = logit.max().item()
        all_labels.append(f"{pred_phrase}({score:.2f})")
        all_det_scores.append(score)

    # Convert to xyxy
    all_boxes_xyxy = torch.zeros(len(boxes_filt), 4)
    if len(boxes_filt) > 0:
        scaled = boxes_filt.clone()
        scaled[:, 0] *= W
        scaled[:, 1] *= H
        scaled[:, 2] *= W
        scaled[:, 3] *= H
        all_boxes_xyxy[:, 0] = scaled[:, 0] - scaled[:, 2] / 2
        all_boxes_xyxy[:, 1] = scaled[:, 1] - scaled[:, 3] / 2
        all_boxes_xyxy[:, 2] = scaled[:, 0] + scaled[:, 2] / 2
        all_boxes_xyxy[:, 3] = scaled[:, 1] + scaled[:, 3] / 2

    print(f"[OK] GroundingDINO found {len(boxes_filt)} candidates")

    # ---- Step 4: CLIP Verification ----
    clip_pass_mask = []
    clip_scores_all = []

    if len(boxes_filt) > 0 and clip_verifier is not None:
        print(f"[i] Running CLIP verification (threshold={clip_threshold})...")

        for i in range(len(all_boxes_xyxy)):
            box = all_boxes_xyxy[i]
            x1, y1, x2, y2 = box.int().numpy()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            if x2 <= x1 or y2 <= y1:
                clip_pass_mask.append(False)
                clip_scores_all.append(0.0)
                continue

            crop = image_np[y1:y2, x1:x2]
            if crop.size == 0:
                clip_pass_mask.append(False)
                clip_scores_all.append(0.0)
                continue

            # Convert numpy crop to PIL for CLIP
            crop_pil = PILImage.fromarray(crop)
            similarity = clip_verifier.compute_similarity(crop_pil, parsed['object_prompt'])
            clip_scores_all.append(similarity)
            clip_pass_mask.append(similarity >= clip_threshold)

        n_passed = sum(clip_pass_mask)
        n_rejected = len(clip_pass_mask) - n_passed
        print(f"[OK] CLIP: {n_passed} passed, {n_rejected} rejected")

        # Fallback: if ALL rejected, keep the best one
        if n_passed == 0 and len(clip_scores_all) > 0:
            best_idx = int(np.argmax(clip_scores_all))
            clip_pass_mask[best_idx] = True
            print(f"[WARN] All rejected, keeping best (#{best_idx+1}, score={clip_scores_all[best_idx]:.3f})")
    else:
        clip_pass_mask = [True] * len(boxes_filt)
        clip_scores_all = [0.0] * len(boxes_filt)

    passed_indices = [i for i, passed in enumerate(clip_pass_mask) if passed]

    # ---- Step 5: Spatial Filtering ----
    selected_idx = None
    if len(passed_indices) > 0:
        if parsed['spatial'] and not parsed['detect_all']:
            passed_boxes = all_boxes_xyxy[passed_indices]
            local_idx = spatial_filter(passed_boxes, parsed['spatial'], image_shape=(H, W))
            if local_idx is not None:
                selected_idx = passed_indices[local_idx]
                print(f"[OK] Spatial filter '{parsed['spatial']}' selected candidate #{selected_idx+1}")
            else:
                selected_idx = passed_indices[0]
        else:
            selected_idx = passed_indices
            print(f"[OK] Keeping all {len(passed_indices)} verified candidates")

    # ---- Step 6: SAM Segmentation ----
    final_masks = None

    if selected_idx is not None:
        if isinstance(selected_idx, list):
            seg_indices = selected_idx
        else:
            seg_indices = [selected_idx]

        seg_boxes_cxcywh = boxes_filt[seg_indices]

        if len(seg_boxes_cxcywh) > 0:
            print(f"[i] Running SAM segmentation on {len(seg_boxes_cxcywh)} objects...")

            sam_predictor.set_image(image_np)

            seg_boxes_xyxy = torch.zeros(len(seg_boxes_cxcywh), 4)
            scaled = seg_boxes_cxcywh.clone()
            scaled[:, 0] *= W
            scaled[:, 1] *= H
            scaled[:, 2] *= W
            scaled[:, 3] *= H
            seg_boxes_xyxy[:, 0] = scaled[:, 0] - scaled[:, 2] / 2
            seg_boxes_xyxy[:, 1] = scaled[:, 1] - scaled[:, 3] / 2
            seg_boxes_xyxy[:, 2] = scaled[:, 0] + scaled[:, 2] / 2
            seg_boxes_xyxy[:, 3] = scaled[:, 1] + scaled[:, 3] / 2

            transformed_boxes = sam_predictor.transform.apply_boxes_torch(
                seg_boxes_xyxy.to(device), (H, W)
            )

            final_masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            final_masks = final_masks.cpu()
            print(f"[OK] SAM segmentation complete")

    # ---- Generate Visualizations ----
    step_images = generate_step_visualizations(
        image_np, scene_result, parsed,
        all_boxes_xyxy, all_labels, all_det_scores,
        clip_pass_mask, clip_scores_all,
        selected_idx, final_masks
    )

    # ---- Build Summary ----
    n_detected = len(boxes_filt)
    n_verified = sum(clip_pass_mask) if clip_pass_mask else 0
    if isinstance(selected_idx, list):
        n_selected = len(selected_idx)
    elif selected_idx is not None:
        n_selected = 1
    else:
        n_selected = 0

    summary_lines = [
        f"TEXT-GUIDED DETECTION RESULTS (Multi-Agent)",
        f"{'='*40}",
        f"Query: \"{user_prompt}\"",
        f"Detection Prompt: '{parsed['object_prompt']}'",
        f"Spatial: {parsed.get('spatial', 'None')}",
        f"",
        f"STEP 1 - Scene Understanding Agent (LLaVA):",
    ]

    if isinstance(scene_result, dict):
        summary_lines.append(f"  Scene: {scene_result.get('scene_type', 'N/A')}")
        summary_lines.append(f"  Lighting: {scene_result.get('lighting', 'N/A')}")
        objects = scene_result.get('objects', [])
        summary_lines.append(f"  Objects found: {len(objects)}")
        for obj in objects[:8]:
            summary_lines.append(f"    - {obj.get('name', '?')} ({obj.get('position', '?')}, {obj.get('color', '?')})")

    summary_lines.append(f"")
    summary_lines.append(f"STEP 2 - Attribute Matching Agent (LLaVA):")
    if isinstance(attr_result, dict):
        summary_lines.append(f"  Reasoning: {attr_result.get('reasoning', 'N/A')}")
        summary_lines.append(f"  Ambiguity: {attr_result.get('ambiguity', 'N/A')}")
        for m in attr_result.get('matched_objects', [])[:3]:
            summary_lines.append(f"  Match: {m.get('name','')} at {m.get('position','')} [{m.get('confidence','')}]")

    summary_lines.extend([
        f"",
        f"STEP 3 - GroundingDINO: {n_detected} candidates detected",
        f"STEP 4 - CLIP Verification: {n_verified}/{n_detected} passed",
    ])

    if clip_scores_all:
        for i, (label, score, passed) in enumerate(zip(all_labels, clip_scores_all, clip_pass_mask)):
            status = "PASS" if passed else "REJECT"
            summary_lines.append(f"  #{i+1} {label}: CLIP={score:.3f} [{status}]")

    summary_lines.extend([
        f"",
        f"STEP 5 - Spatial Filter: '{parsed.get('spatial', 'none')}' -> {n_selected} selected",
        f"STEP 6 - SAM Segmentation: {'Complete' if final_masks is not None else 'Skipped'}",
    ])

    summary = "\n".join(summary_lines)
    print(f"\n{summary}")

    return {
        "step_images": step_images,
        "scene_result": scene_result,
        "parsed_query": parsed,
        "final_masks": final_masks,
        "selected_idx": selected_idx,
        "summary": summary,
    }
