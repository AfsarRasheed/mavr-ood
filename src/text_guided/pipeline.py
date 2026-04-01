"""
Text-Guided Detection Pipeline (Main Orchestrator)
Runs all 7 steps: Scene Agent → Attribute Agent → GroundingDINO → CLIP → Spatial → SAM → Reasoning
"""

import gc
import os
import json
import tempfile
import numpy as np
import torch

from src.text_guided.scene_agent import scene_understanding
from src.text_guided.attribute_agent import attribute_matching_agent
from src.text_guided.query_parser import parse_query, llava_parse_query, spatial_filter
from src.text_guided.visualizer import generate_step_visualizations
from src.text_guided.reasoning_agent import reasoning_agent
from src.text_guided.candidate_reasoner import summarize_candidate_match
from src.text_guided.reliability import determine_match_decision
from src.agents.vlm_backend import run_vlm


def _box_center(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def _safe_crop(image_np, box_xyxy):
    from PIL import Image as PILImage

    H, W = image_np.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image_np[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return PILImage.fromarray(crop)


def _run_anchor_detection(gdino_model, image_tensor_dev, anchor_text, H, W, threshold=0.25):
    anchor_caption = anchor_text.lower().strip()
    if not anchor_caption.endswith("."):
        anchor_caption += "."

    with torch.no_grad():
        anchor_outputs = gdino_model(image_tensor_dev[None], captions=[anchor_caption])

    anchor_logits = anchor_outputs["pred_logits"].cpu().sigmoid()[0]
    anchor_boxes_raw = anchor_outputs["pred_boxes"].cpu()[0]
    anchor_filt = anchor_logits.max(dim=1)[0] > threshold
    anchor_boxes_cxcywh = anchor_boxes_raw[anchor_filt]

    if len(anchor_boxes_cxcywh) == 0:
        return None

    anchor_xyxy = torch.zeros_like(anchor_boxes_cxcywh)
    anchor_xyxy[:, 0] = (anchor_boxes_cxcywh[:, 0] - anchor_boxes_cxcywh[:, 2] / 2) * W
    anchor_xyxy[:, 1] = (anchor_boxes_cxcywh[:, 1] - anchor_boxes_cxcywh[:, 3] / 2) * H
    anchor_xyxy[:, 2] = (anchor_boxes_cxcywh[:, 0] + anchor_boxes_cxcywh[:, 2] / 2) * W
    anchor_xyxy[:, 3] = (anchor_boxes_cxcywh[:, 1] + anchor_boxes_cxcywh[:, 3] / 2) * H
    return anchor_xyxy


def _compute_spatial_score(box_xyxy, parsed, image_shape, anchor_boxes=None, anchor2_boxes=None):
    H, W = image_shape
    box = np.array(box_xyxy, dtype=np.float32)
    center = _box_center(box)
    x_norm = float(center[0] / max(W, 1))
    y_norm = float(center[1] / max(H, 1))
    area = float(max(box[2] - box[0], 1) * max(box[3] - box[1], 1))
    area_norm = area / float(max(H * W, 1))

    spatial = parsed.get("spatial")
    if not spatial:
        return 0.6

    if spatial == "left":
        return float(np.clip(1.0 - x_norm, 0.0, 1.0))
    if spatial == "right":
        return float(np.clip(x_norm, 0.0, 1.0))
    if spatial == "center":
        return float(np.clip(1.0 - abs(x_norm - 0.5) * 2.0, 0.0, 1.0))
    if spatial == "top":
        return float(np.clip(1.0 - y_norm, 0.0, 1.0))
    if spatial == "bottom":
        return float(np.clip(y_norm, 0.0, 1.0))
    if spatial == "largest":
        return float(np.clip(area_norm * 6.0, 0.0, 1.0))
    if spatial == "smallest":
        return float(np.clip(1.0 - area_norm * 6.0, 0.0, 1.0))
    if spatial in ("nearest", "ahead"):
        return float(np.clip(y_norm, 0.0, 1.0))
    if spatial == "farthest":
        return float(np.clip(1.0 - y_norm, 0.0, 1.0))

    if anchor_boxes is not None and len(anchor_boxes) > 0:
        anchor = anchor_boxes.numpy() if torch.is_tensor(anchor_boxes) else np.array(anchor_boxes)
        anchor_center = _box_center(anchor[0])
        dx = abs(center[0] - anchor_center[0]) / max(W, 1)
        dy = abs(center[1] - anchor_center[1]) / max(H, 1)
        dist = float(np.sqrt(dx ** 2 + dy ** 2))

        if spatial == "next_to":
            return float(np.clip(1.0 - dist * 1.5, 0.0, 1.0))
        if spatial == "behind":
            return float(np.clip((anchor_center[1] - center[1]) / max(H * 0.5, 1.0) + 0.5, 0.0, 1.0))
        if spatial == "in_front":
            return float(np.clip((center[1] - anchor_center[1]) / max(H * 0.5, 1.0) + 0.5, 0.0, 1.0))
        if spatial == "above":
            return float(np.clip((anchor_center[1] - center[1]) / max(H * 0.5, 1.0) + 0.5, 0.0, 1.0))
        if spatial == "below":
            return float(np.clip((center[1] - anchor_center[1]) / max(H * 0.5, 1.0) + 0.5, 0.0, 1.0))

    if spatial == "between" and anchor_boxes is not None and anchor2_boxes is not None and len(anchor_boxes) > 0 and len(anchor2_boxes) > 0:
        anchor1 = anchor_boxes.numpy() if torch.is_tensor(anchor_boxes) else np.array(anchor_boxes)
        anchor2 = anchor2_boxes.numpy() if torch.is_tensor(anchor2_boxes) else np.array(anchor2_boxes)
        midpoint = (_box_center(anchor1[0]) + _box_center(anchor2[0])) / 2.0
        dist = np.linalg.norm((center - midpoint) / np.array([max(W, 1), max(H, 1)], dtype=np.float32))
        return float(np.clip(1.0 - dist * 2.0, 0.0, 1.0))

    return 0.45


def _scene_consistency_score(parsed, scene_result):
    if not isinstance(scene_result, dict):
        return 0.5

    target_object = (parsed.get("target_object") or "").lower()
    color = ((parsed.get("attributes") or {}).get("color") or "").lower()
    objects = scene_result.get("objects", []) or []

    if not objects:
        return 0.5

    for obj in objects:
        name = str(obj.get("name", "")).lower()
        obj_color = str(obj.get("color", "")).lower()
        if target_object and target_object in name:
            if color and color not in name and color not in obj_color:
                return 0.65
            return 0.9

    return 0.45


def _attribute_agent_score(parsed, attr_result):
    if not isinstance(attr_result, dict):
        return 0.5

    ambiguity = str(attr_result.get("ambiguity", "")).lower()
    matched = attr_result.get("matched_objects", []) or []
    if not matched:
        return 0.45 if ambiguity in ("high", "unknown") else 0.55

    confidence_map = {"high": 0.92, "medium": 0.72, "low": 0.52}
    best = max(confidence_map.get(str(m.get("confidence", "")).lower(), 0.6) for m in matched[:3])

    if ambiguity == "high":
        best -= 0.1
    elif ambiguity == "none":
        best += 0.05

    return float(np.clip(best, 0.0, 1.0))


def _score_candidate(box_xyxy, det_score, clip_scores, parsed, image_shape, scene_result,
                     attr_result, anchor_boxes=None, anchor2_boxes=None):
    spatial_score = _compute_spatial_score(
        box_xyxy,
        parsed,
        image_shape=image_shape,
        anchor_boxes=anchor_boxes,
        anchor2_boxes=anchor2_boxes,
    )
    scene_score = _scene_consistency_score(parsed, scene_result)
    attr_agent_score = _attribute_agent_score(parsed, attr_result)
    object_score = float(np.clip((det_score + clip_scores.get("object_score", det_score)) / 2.0, 0.0, 1.0))
    attribute_score = float(np.clip(
        max(
            clip_scores.get("attribute_score", 0.0),
            clip_scores.get("color_score", 0.0),
            clip_scores.get("condition_score", 0.0),
        ),
        0.0,
        1.0,
    ))
    clip_score = float(np.clip(clip_scores.get("full_query_score", 0.0), 0.0, 1.0))

    final_score = (
        object_score * 0.28 +
        attribute_score * 0.22 +
        clip_score * 0.20 +
        spatial_score * 0.18 +
        scene_score * 0.06 +
        attr_agent_score * 0.06
    )

    return {
        "object_score": round(object_score, 4),
        "attribute_score": round(attribute_score, 4),
        "clip_score": round(clip_score, 4),
        "spatial_score": round(spatial_score, 4),
        "scene_consistency_score": round(scene_score, 4),
        "attribute_agent_score": round(attr_agent_score, 4),
        "color_contrast": round(float(clip_scores.get("color_contrast", 0.0)), 4),
        "condition_contrast": round(float(clip_scores.get("condition_contrast", 0.0)), 4),
        "final_score": round(float(np.clip(final_score, 0.0, 1.0)), 4),
    }


def _rerank_candidates_with_vlm(image_np, ranked_candidates, user_prompt, max_candidates=3):
    top = ranked_candidates[:max_candidates]
    if len(top) < 2:
        return ranked_candidates

    temp_paths = []
    try:
        for candidate in top:
            crop = _safe_crop(image_np, candidate["box_xyxy"])
            if crop is None:
                continue
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            crop.save(tmp.name)
            temp_paths.append(tmp.name)
            candidate["crop_path"] = tmp.name

        for idx, candidate in enumerate(top):
            crop_path = candidate.get("crop_path")
            if not crop_path:
                continue

            prompt = f"""User query: "{user_prompt}"
Candidate rank before reranking: {idx + 1}
Current structured scores: {json.dumps(candidate.get('scores', {}))}

Evaluate ONLY this crop as a candidate match for the user query.
Return ONLY valid JSON:
{{
  "query_match_score": 0.0,
  "satisfied_constraints": ["short phrases"],
  "violated_constraints": ["short phrases"],
  "reason": "one sentence"
}}"""
            response = run_vlm([{"role": "user", "content": prompt}], image_path=crop_path)
            payload = json.loads(response.strip().replace("```json", "").replace("```", "").strip())
            vlm_score = float(payload.get("query_match_score", 0.0))
            candidate["vlm_rerank"] = {
                "score": round(vlm_score, 4),
                "reason": payload.get("reason", ""),
                "satisfied_constraints": payload.get("satisfied_constraints", []),
                "violated_constraints": payload.get("violated_constraints", []),
            }
            candidate["scores"]["vlm_score"] = round(vlm_score, 4)
            candidate["scores"]["final_score"] = round(
                float(np.clip(candidate["scores"]["final_score"] * 0.8 + vlm_score * 0.2, 0.0, 1.0)),
                4,
            )
    except Exception as e:
        print(f"[WARN] VLM candidate reranking skipped: {e}")
    finally:
        for candidate in top:
            candidate.pop("crop_path", None)
        for path in temp_paths:
            try:
                os.unlink(path)
            except OSError:
                pass

    return sorted(ranked_candidates, key=lambda item: item["scores"]["final_score"], reverse=True)


def run_text_guided_pipeline(image_np, user_prompt, image_path,
                              gdino_model, sam_predictor, clip_verifier,
                              box_threshold=0.35, clip_threshold=0.25,
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

    # ---- Step 2.5: Advanced Query Parsing while LLaVA is still loaded ----
    parsed = llava_parse_query(user_prompt)
    parsed['attr_agent_result'] = attr_result

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

    print(f"[OK] GroundingDINO found {len(boxes_filt)} candidates")

    # ---- Retry logic if 0 candidates found ----
    if len(boxes_filt) == 0:
        # Retry 1: lower threshold
        retry_threshold = 0.20
        print(f"[WARN] 0 candidates — retrying with lower threshold ({retry_threshold})...")
        filt_mask = logits.max(dim=1)[0] > retry_threshold
        logits_filt = logits[filt_mask]
        boxes_filt = boxes_cxcywh[filt_mask]
        print(f"[i] Retry 1: {len(boxes_filt)} candidates at threshold {retry_threshold}")

    if len(boxes_filt) == 0 and parsed['object_prompt'] != user_prompt.lower().strip():
        # Retry 2: use raw user prompt
        raw_caption = user_prompt.lower().strip()
        if not raw_caption.endswith("."):
            raw_caption += "."
        print(f"[WARN] Still 0 — retrying with raw prompt: '{raw_caption}'")
        with torch.no_grad():
            outputs = gdino_model(image_tensor_dev[None], captions=[raw_caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes_cxcywh = outputs["pred_boxes"].cpu()[0]
        filt_mask = logits.max(dim=1)[0] > 0.20
        logits_filt = logits[filt_mask]
        boxes_filt = boxes_cxcywh[filt_mask]
        tokenized = tokenizer(raw_caption)
        print(f"[i] Retry 2: {len(boxes_filt)} candidates with raw prompt")

    # Build labels and convert to xyxy (AFTER retries so data is final)
    all_labels = []
    all_det_scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > 0.25, tokenized, tokenizer)
        score = logit.max().item()
        all_labels.append(f"{pred_phrase}({score:.2f})")
        all_det_scores.append(score)

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

    # ---- Step 4/5: CLIP + Full-Query Candidate Ranking ----
    clip_pass_mask = []
    clip_scores_all = []
    candidate_details = []
    selected_idx = None
    anchor_boxes = None

    # Detect anchor (reference) object if relational query
    if parsed.get('anchor') and parsed.get('spatial') in ('next_to', 'behind', 'in_front', 'above', 'below', 'between'):
        print(f"[i] Detecting anchor object: '{parsed['anchor']}'...")
        try:
            anchor_boxes = _run_anchor_detection(gdino_model, image_tensor_dev, parsed['anchor'], H, W)
            if anchor_boxes is not None and len(anchor_boxes) > 0:
                print(f"[OK] Found {len(anchor_boxes)} anchor object(s): '{parsed['anchor']}'")
            else:
                print(f"[WARN] Anchor object '{parsed['anchor']}' not found, falling back to closest")
        except Exception as e:
            print(f"[WARN] Anchor detection failed: {e}")

    # Detect second anchor for "between" queries
    anchor2_boxes = None
    if parsed.get('anchor2') and parsed.get('spatial') == 'between':
        print(f"[i] Detecting second anchor object: '{parsed['anchor2']}'...")
        try:
            anchor2_boxes = _run_anchor_detection(gdino_model, image_tensor_dev, parsed['anchor2'], H, W)
            if anchor2_boxes is not None and len(anchor2_boxes) > 0:
                print(f"[OK] Found {len(anchor2_boxes)} second anchor(s): '{parsed['anchor2']}'")
            else:
                print(f"[WARN] Second anchor '{parsed['anchor2']}' not found")
        except Exception as e:
            print(f"[WARN] Second anchor detection failed: {e}")

    if len(boxes_filt) > 0 and clip_verifier is not None:
        print(f"[i] Running CLIP verification + natural-language candidate scoring (threshold={clip_threshold})...")
        for i in range(len(all_boxes_xyxy)):
            box = all_boxes_xyxy[i]
            crop_pil = _safe_crop(image_np, box.int().numpy())
            if crop_pil is None:
                clip_pass_mask.append(False)
                clip_scores_all.append(0.0)
                continue

            clip_score_map = clip_verifier.compute_discriminative_scores(crop_pil, parsed)
            primary_similarity = clip_score_map.get("full_query_score", 0.0)
            clip_scores_all.append(primary_similarity)
            clip_pass_mask.append(primary_similarity >= clip_threshold)

            candidate_scores = _score_candidate(
                box_xyxy=box.numpy() if torch.is_tensor(box) else box,
                det_score=all_det_scores[i],
                clip_scores=clip_score_map,
                parsed=parsed,
                image_shape=(H, W),
                scene_result=scene_result,
                attr_result=attr_result,
                anchor_boxes=anchor_boxes,
                anchor2_boxes=anchor2_boxes,
            )
            match_analysis = summarize_candidate_match(parsed, candidate_scores, attr_result)
            candidate_scores["final_score"] = round(
                max(0.0, float(candidate_scores["final_score"]) - float(match_analysis["ambiguity_penalty"])),
                4,
            )
            candidate_details.append({
                "index": i,
                "label": all_labels[i],
                "det_score": round(float(all_det_scores[i]), 4),
                "clip_scores": {k: round(float(v), 4) for k, v in clip_score_map.items()},
                "scores": candidate_scores,
                "match_analysis": match_analysis,
                "box_xyxy": (box.numpy() if torch.is_tensor(box) else np.array(box)).tolist(),
            })

        n_passed = sum(clip_pass_mask)
        n_rejected = len(clip_pass_mask) - n_passed
        print(f"[OK] CLIP: {n_passed} passed, {n_rejected} rejected")
    else:
        clip_pass_mask = [True] * len(boxes_filt)
        clip_scores_all = [0.0] * len(boxes_filt)
        for i in range(len(all_boxes_xyxy)):
            candidate_scores = _score_candidate(
                box_xyxy=all_boxes_xyxy[i].numpy(),
                det_score=all_det_scores[i],
                clip_scores={"full_query_score": 0.0, "object_score": all_det_scores[i], "attribute_score": 0.0},
                parsed=parsed,
                image_shape=(H, W),
                scene_result=scene_result,
                attr_result=attr_result,
                anchor_boxes=anchor_boxes,
                anchor2_boxes=anchor2_boxes,
            )
            match_analysis = summarize_candidate_match(parsed, candidate_scores, attr_result)
            candidate_scores["final_score"] = round(
                max(0.0, float(candidate_scores["final_score"]) - float(match_analysis["ambiguity_penalty"])),
                4,
            )
            candidate_details.append({
                "index": i,
                "label": all_labels[i],
                "det_score": round(float(all_det_scores[i]), 4),
                "clip_scores": {},
                "scores": candidate_scores,
                "match_analysis": match_analysis,
                "box_xyxy": all_boxes_xyxy[i].numpy().tolist(),
            })

    if candidate_details and not any(clip_pass_mask):
        best_idx = int(np.argmax([item["scores"]["final_score"] for item in candidate_details]))
        clip_pass_mask[best_idx] = True
        print(f"[WARN] All candidates were below CLIP threshold, keeping best-ranked candidate #{best_idx + 1}")

    ranked_candidates = sorted(candidate_details, key=lambda item: item["scores"]["final_score"], reverse=True)
    need_rerank = len(ranked_candidates) > 1 and (
        ranked_candidates[0]["scores"]["final_score"] - ranked_candidates[1]["scores"]["final_score"] < 0.08
    )
    enable_vlm_rerank = os.getenv("ENABLE_VLM_RERANK", "0") == "1"
    if need_rerank and enable_vlm_rerank:
        print("[i] Top candidates are close; running VLM reranking on leading crops...")
        ranked_candidates = _rerank_candidates_with_vlm(image_np, ranked_candidates, user_prompt)
    elif need_rerank:
        print("[i] Top candidates are close, but VLM reranking is disabled (set ENABLE_VLM_RERANK=1 to enable it).")

    match_decision = determine_match_decision(ranked_candidates)
    selected_indices = match_decision["selected_indices"]
    if not selected_indices and ranked_candidates and match_decision["state"] != "no_reliable_match":
        selected_indices = [ranked_candidates[0]["index"]]

    if selected_indices:
        selected_idx = selected_indices if len(selected_indices) > 1 else selected_indices[0]
        print(
            f"[OK] Candidate ranking selected {selected_indices} "
            f"with state '{match_decision['state']}' (confidence={match_decision['confidence']:.2f})"
        )
    else:
        selected_idx = None
        print(f"[WARN] No reliable object selected ({match_decision['reason']})")

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

            # Get SAM device reliably
            sam_device = next(sam_predictor.model.parameters()).device

            # apply_boxes_torch uses deepcopy which can strip CUDA device
            # so we do the transform on CPU, then move result to CUDA
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(
                seg_boxes_xyxy, (H, W)
            ).to(sam_device)

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

    top_ranked = ranked_candidates[:3] if 'ranked_candidates' in locals() else []
    match_state = match_decision["state"] if 'match_decision' in locals() else "unknown"
    match_confidence = match_decision["confidence"] if 'match_decision' in locals() else 0.0

    summary_lines = [
        f"TEXT-GUIDED DETECTION RESULTS (Multi-Agent)",
        f"{'='*40}",
        f"Query: \"{user_prompt}\"",
        f"Detection Prompt: '{parsed['object_prompt']}'",
        f"Target Object: {parsed.get('target_object', 'unknown')}",
        f"Spatial: {parsed.get('spatial', 'None')}",
        f"Priority Order: {', '.join(parsed.get('priority_order', []))}",
        f"Match State: {match_state}",
        f"Match Confidence: {match_confidence:.3f}",
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
        f"STEP 5 - Candidate Ranking: state='{match_state}', confidence={match_confidence:.3f}",
    ])

    if clip_scores_all:
        for i, (label, score, passed) in enumerate(zip(all_labels, clip_scores_all, clip_pass_mask)):
            status = "PASS" if passed else "REJECT"
            summary_lines.append(f"  #{i+1} {label}: CLIP={score:.3f} [{status}]")

    if top_ranked:
        summary_lines.append("")
        summary_lines.append("TOP CANDIDATE SCORES:")
        for candidate in top_ranked:
            rank = ranked_candidates.index(candidate) + 1
            scores = candidate["scores"]
            summary_lines.append(
                f"  Rank {rank} -> candidate #{candidate['index'] + 1}: "
                f"final={scores['final_score']:.3f}, object={scores['object_score']:.3f}, "
                f"attribute={scores['attribute_score']:.3f}, clip={scores['clip_score']:.3f}, "
                f"spatial={scores['spatial_score']:.3f}"
            )
            if candidate.get("match_analysis", {}).get("reason"):
                summary_lines.append(f"    Match Analysis: {candidate['match_analysis']['reason']}")
            if candidate.get("vlm_rerank", {}).get("reason"):
                summary_lines.append(f"    VLM: {candidate['vlm_rerank']['reason']}")

    summary_lines.extend([
        f"",
        f"STEP 6 - Selection Outcome: '{match_state}' -> {n_selected} selected",
        f"STEP 6 - SAM Segmentation: {'Complete' if final_masks is not None else 'Skipped'}",
    ])

    summary = "\n".join(summary_lines)
    print(f"\n{summary}")

    # ---- Step 7: Reasoning Agent (LLaVA text-only) ----
    # Free detection models first to make room for LLaVA
    gc.collect()
    torch.cuda.empty_cache()

    # Build CLIP details string for reasoning context
    clip_detail_str = ""
    if clip_scores_all:
        clip_parts = []
        for i, (score, passed) in enumerate(zip(clip_scores_all, clip_pass_mask)):
            status = "passed" if passed else "rejected"
            clip_parts.append(f"Candidate {i+1}: similarity={score:.3f} ({status})")
        clip_detail_str = "; ".join(clip_parts)

    # Prepare reasoning input
    scene_type = scene_result.get("scene_type", "road scene") if isinstance(scene_result, dict) else "road scene"
    lighting = scene_result.get("lighting", "unknown") if isinstance(scene_result, dict) else "unknown"
    n_scene_objects = len(scene_result.get("objects", [])) if isinstance(scene_result, dict) else 0
    agent_reasoning = attr_result.get("reasoning", "") if isinstance(attr_result, dict) else ""
    agent_ambiguity = attr_result.get("ambiguity", "unknown") if isinstance(attr_result, dict) else "unknown"

    reasoning_data = {
        "query": user_prompt,
        "scene_type": scene_type,
        "lighting": lighting,
        "n_objects": n_scene_objects,
        "reasoning": agent_reasoning,
        "ambiguity": agent_ambiguity,
        "recommended_prompt": parsed['object_prompt'],
        "n_candidates": n_detected,
        "n_verified": n_verified,
        "n_rejected": n_detected - n_verified,
        "clip_details": clip_detail_str,
        "spatial_term": parsed.get('spatial', 'none'),
        "n_selected": n_selected,
        "match_state": match_state,
        "match_confidence": match_confidence,
        "priority_order": parsed.get('priority_order', []),
        "top_candidates": [
            {
                "candidate_index": candidate["index"] + 1,
                "final_score": candidate["scores"]["final_score"],
                "object_score": candidate["scores"]["object_score"],
                "attribute_score": candidate["scores"]["attribute_score"],
                "clip_score": candidate["scores"]["clip_score"],
                "spatial_score": candidate["scores"]["spatial_score"],
                "reason": candidate.get("match_analysis", {}).get("reason", ""),
                "satisfied_constraints": candidate.get("match_analysis", {}).get("satisfied_constraints", []),
                "violated_constraints": candidate.get("match_analysis", {}).get("violated_constraints", []),
            }
            for candidate in top_ranked
        ],
    }

    reasoning_text = reasoning_agent(reasoning_data)
    print(f"\nSTEP 7 - Reasoning Agent:")
    print(f"  {reasoning_text[:200]}..." if len(reasoning_text) > 200 else f"  {reasoning_text}")

    # Free LLaVA again after reasoning
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
        print("[OK] LLaVA freed from GPU memory (after reasoning)")
    except Exception:
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Save Results to JSON ----
    output_dir = os.path.join("outputs", "text_guided")
    os.makedirs(output_dir, exist_ok=True)

    img_basename = os.path.splitext(os.path.basename(image_path))[0] if image_path else "unknown"

    with open(os.path.join(output_dir, f"{img_basename}_scene_agent.json"), "w") as f:
        json.dump(scene_result if isinstance(scene_result, dict) else {"raw": str(scene_result)}, f, indent=2)
    print(f"[OK] Saved: {output_dir}/{img_basename}_scene_agent.json")

    with open(os.path.join(output_dir, f"{img_basename}_attribute_agent.json"), "w") as f:
        json.dump(attr_result if isinstance(attr_result, dict) else {"raw": str(attr_result)}, f, indent=2)
    print(f"[OK] Saved: {output_dir}/{img_basename}_attribute_agent.json")

    with open(os.path.join(output_dir, f"{img_basename}_reasoning.json"), "w") as f:
        json.dump({"reasoning": reasoning_text}, f, indent=2)
    print(f"[OK] Saved: {output_dir}/{img_basename}_reasoning.json")

    with open(os.path.join(output_dir, f"{img_basename}_summary.json"), "w") as f:
        json.dump({
            "query": user_prompt,
            "detection_prompt": parsed['object_prompt'],
            "spatial": parsed.get('spatial'),
            "candidates_found": n_detected,
            "clip_verified": n_verified,
            "selected": n_selected,
            "match_state": match_state,
            "match_confidence": match_confidence,
            "candidate_rankings": ranked_candidates,
            "summary": summary,
            "reasoning": reasoning_text,
        }, f, indent=2)
    print(f"[OK] Saved: {output_dir}/{img_basename}_summary.json")

    return {
        "step_images": step_images,
        "scene_result": scene_result,
        "attr_result": attr_result,
        "parsed": parsed,
        "parsed_query": parsed,
        "n_detected": n_detected,
        "n_verified": n_verified,
        "n_selected": n_selected,
        "final_masks": final_masks,
        "selected_idx": selected_idx,
        "candidate_rankings": ranked_candidates,
        "match_state": match_state,
        "match_confidence": match_confidence,
        "match_reason": match_decision.get("reason"),
        "summary": summary,
        "reasoning": reasoning_text,
    }
