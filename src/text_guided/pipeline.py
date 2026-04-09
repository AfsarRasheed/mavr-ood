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
from PIL import Image as PILImage

from src.text_guided.scene_agent import scene_understanding
from src.text_guided.attribute_agent import attribute_matching_agent
from src.text_guided.query_parser import llava_parse_query
from src.text_guided.visualizer import generate_step_visualizations
from src.text_guided.reasoning_agent import reasoning_agent
from src.text_guided.candidate_reasoner import summarize_candidate_match
from src.text_guided.semantic_controller import build_semantic_plan
from src.text_guided.candidate_judge import judge_candidate_against_plan
from src.text_guided.reliability import determine_match_decision
from src.text_guided.candidate_adapter import cxcywh_normalized_to_xyxy
from src.text_guided.florence2_backend import run_florence2_grounding
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


def _normalize_prompt_tokens(text):
    import re

    return set(re.findall(r"[a-zA-Z][a-zA-Z\-]*", str(text or "").lower()))


def _is_prompt_consistent_with_query(parsed, prompt_text):
    tokens = _normalize_prompt_tokens(prompt_text)
    attrs = parsed.get("attributes") or {}
    semantic_plan = parsed.get("semantic_plan") or {}
    target_object = str(parsed.get("target_object") or "").lower().strip()
    color = str(attrs.get("color") or "").lower().strip()
    condition = str(attrs.get("condition") or "").lower().strip()
    mandatory_kinds = {item.get("kind") for item in semantic_plan.get("mandatory_constraints", [])}

    if target_object and target_object not in tokens:
        return False, f"missing target object '{target_object}'"

    if color:
        if "color" in mandatory_kinds and color not in tokens:
            gray_family = {"grey", "gray", "silver"}
            if not (color in gray_family and tokens.intersection(gray_family)):
                return False, f"missing mandatory color '{color}'"
        competing_colors = {
            token for token in tokens
            if token in {"red", "blue", "green", "yellow", "white", "black", "grey", "gray", "silver",
                         "brown", "orange", "purple", "pink", "gold", "dark", "light", "bright",
                         "beige", "maroon", "navy", "cyan", "teal"}
            and token != color
        }
        gray_family = {"grey", "gray", "silver"}
        if color in gray_family:
            competing_colors = {token for token in competing_colors if token not in gray_family}
        if competing_colors:
            return False, f"conflicts with color '{color}'"

    if condition:
        if "condition" in mandatory_kinds and condition not in tokens:
            return False, f"missing mandatory condition '{condition}'"
        competing_conditions = {
            token for token in tokens
            if token in {"damaged", "broken", "burning", "parked", "moving", "stopped", "crashed",
                         "overturned", "tilted", "fallen", "open", "closed", "old", "new",
                         "dirty", "clean", "blurred"}
            and token != condition
        }
        if competing_conditions:
            return False, f"conflicts with condition '{condition}'"

    return True, "consistent with parsed query"


def _apply_match_penalties(candidate_scores, match_analysis):
    penalty = float(match_analysis.get("ambiguity_penalty", 0.0)) + float(
        match_analysis.get("relation_uncertainty_penalty", 0.0)
    )
    candidate_scores["final_score"] = round(
        max(0.0, float(candidate_scores["final_score"]) - penalty),
        4,
    )
    return candidate_scores


def _apply_semantic_judgment(candidate_scores, semantic_judgment):
    score = float(candidate_scores.get("final_score", 0.0))
    score += float(semantic_judgment.get("semantic_bonus", 0.0))
    score -= float(semantic_judgment.get("contradiction_penalty", 0.0))
    candidate_scores["semantic_bonus"] = round(float(semantic_judgment.get("semantic_bonus", 0.0)), 4)
    candidate_scores["contradiction_penalty"] = round(float(semantic_judgment.get("contradiction_penalty", 0.0)), 4)
    candidate_scores["final_score"] = round(float(np.clip(score, 0.0, 1.0)), 4)
    return candidate_scores


def _format_anchor_confidence(parsed, anchor_info=None, anchor2_info=None):
    if parsed.get("spatial") == "between":
        conf1 = float((anchor_info or {}).get("confidence", 0.0))
        conf2 = float((anchor2_info or {}).get("confidence", 0.0))
        return f"{conf1:.3f}, {conf2:.3f}"
    return f"{float((anchor_info or {}).get('confidence', 0.0 if parsed.get('anchor') else 1.0)):.3f}"


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
        return {
            "boxes": None,
            "confidence": 0.0,
            "count": 0,
            "label": anchor_text,
        }

    anchor_xyxy = torch.zeros_like(anchor_boxes_cxcywh)
    anchor_xyxy[:, 0] = (anchor_boxes_cxcywh[:, 0] - anchor_boxes_cxcywh[:, 2] / 2) * W
    anchor_xyxy[:, 1] = (anchor_boxes_cxcywh[:, 1] - anchor_boxes_cxcywh[:, 3] / 2) * H
    anchor_xyxy[:, 2] = (anchor_boxes_cxcywh[:, 0] + anchor_boxes_cxcywh[:, 2] / 2) * W
    anchor_xyxy[:, 3] = (anchor_boxes_cxcywh[:, 1] + anchor_boxes_cxcywh[:, 3] / 2) * H
    anchor_confidence = float(anchor_logits[anchor_filt].max(dim=1)[0].max().item()) if anchor_filt.any() else 0.0
    return {
        "boxes": anchor_xyxy,
        "confidence": round(anchor_confidence, 4),
        "count": int(len(anchor_boxes_cxcywh)),
        "label": anchor_text,
    }


def _resolve_text_guided_backend(requested_backend, florence2_backend):
    backend = str(requested_backend or os.getenv("TEXT_GUIDED_BACKEND", "florence2")).strip().lower()
    if backend not in {"gdino", "florence2"}:
        print(f"[WARN] Unknown TEXT_GUIDED_BACKEND='{backend}', falling back to gdino")
        return "gdino"
    if backend == "florence2" and not florence2_backend:
        print("[WARN] Florence-2 backend requested but not loaded, falling back to gdino")
        return "gdino"
    return backend


def _run_gdino_candidate_proposal(gdino_model, image_tensor_dev, prompt, image_size, box_threshold):
    from groundingdino.util.utils import get_phrases_from_posmap

    height, width = image_size
    caption = prompt.lower().strip()
    if not caption.endswith("."):
        caption += "."

    with torch.no_grad():
        outputs = gdino_model(image_tensor_dev[None], captions=[caption])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes_cxcywh = outputs["pred_boxes"].cpu()[0]
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes_cxcywh[filt_mask]

    tokenizer = gdino_model.tokenizer
    tokenized = tokenizer(caption)

    labels = []
    scores = []
    for logit in logits_filt:
        pred_phrase = get_phrases_from_posmap(logit > 0.25, tokenized, tokenizer)
        score = float(logit.max().item())
        labels.append(f"{pred_phrase}({score:.2f})")
        scores.append(score)

    return {
        "backend": "gdino",
        "prompt_used": prompt,
        "boxes_cxcywh": boxes_filt,
        "boxes_xyxy": cxcywh_normalized_to_xyxy(boxes_filt, (height, width)),
        "labels": labels,
        "scores": scores,
        "raw_response": outputs,
    }


def _run_text_guided_candidate_proposal(
    *,
    backend_name,
    gdino_model,
    florence2_backend,
    image_pil,
    image_tensor_dev,
    parsed,
    image_size,
    box_threshold,
):
    prompt = parsed["object_prompt"]

    if backend_name == "florence2":
        florence_result = run_florence2_grounding(
            florence_model=florence2_backend["model"],
            florence_processor=florence2_backend["processor"],
            image_pil=image_pil,
            prompt=prompt,
            device=florence2_backend["device"],
            image_size=image_size,
        )
        return {
            "backend": "florence2",
            "prompt_used": prompt,
            "boxes_cxcywh": florence_result["boxes_cxcywh"],
            "boxes_xyxy": florence_result["boxes_xyxy"],
            "labels": florence_result["labels"],
            "scores": florence_result["scores"],
            "raw_response": florence_result["raw_response"],
        }

    return _run_gdino_candidate_proposal(
        gdino_model=gdino_model,
        image_tensor_dev=image_tensor_dev,
        prompt=prompt,
        image_size=image_size,
        box_threshold=box_threshold,
    )


def _compute_spatial_score(box_xyxy, parsed, image_shape, anchor_boxes=None, anchor2_boxes=None,
                           anchor_confidence=1.0, anchor2_confidence=1.0):
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

    is_relational = spatial in ("next_to", "behind", "in_front", "above", "below", "between")
    if is_relational and anchor_boxes is None:
        return 0.15

    if anchor_boxes is not None and len(anchor_boxes) > 0:
        anchor = anchor_boxes.numpy() if torch.is_tensor(anchor_boxes) else np.array(anchor_boxes)
        anchor_center = _box_center(anchor[0])
        dx = abs(center[0] - anchor_center[0]) / max(W, 1)
        dy = abs(center[1] - anchor_center[1]) / max(H, 1)
        dist = float(np.sqrt(dx ** 2 + dy ** 2))
        confidence_scale = 0.55 + 0.45 * float(np.clip(anchor_confidence, 0.0, 1.0))

        if spatial == "next_to":
            return float(np.clip((1.0 - dist * 1.5) * confidence_scale, 0.0, 1.0))
        if spatial == "behind":
            return float(np.clip(((anchor_center[1] - center[1]) / max(H * 0.5, 1.0) + 0.5) * confidence_scale, 0.0, 1.0))
        if spatial == "in_front":
            return float(np.clip(((center[1] - anchor_center[1]) / max(H * 0.5, 1.0) + 0.5) * confidence_scale, 0.0, 1.0))
        if spatial == "above":
            return float(np.clip(((anchor_center[1] - center[1]) / max(H * 0.5, 1.0) + 0.5) * confidence_scale, 0.0, 1.0))
        if spatial == "below":
            return float(np.clip(((center[1] - anchor_center[1]) / max(H * 0.5, 1.0) + 0.5) * confidence_scale, 0.0, 1.0))

    if spatial == "between" and anchor_boxes is not None and anchor2_boxes is not None and len(anchor_boxes) > 0 and len(anchor2_boxes) > 0:
        anchor1 = anchor_boxes.numpy() if torch.is_tensor(anchor_boxes) else np.array(anchor_boxes)
        anchor2 = anchor2_boxes.numpy() if torch.is_tensor(anchor2_boxes) else np.array(anchor2_boxes)
        midpoint = (_box_center(anchor1[0]) + _box_center(anchor2[0])) / 2.0
        dist = np.linalg.norm((center - midpoint) / np.array([max(W, 1), max(H, 1)], dtype=np.float32))
        confidence_scale = 0.45 + 0.55 * min(
            float(np.clip(anchor_confidence, 0.0, 1.0)),
            float(np.clip(anchor2_confidence, 0.0, 1.0)),
        )
        return float(np.clip((1.0 - dist * 2.0) * confidence_scale, 0.0, 1.0))

    return 0.45


def _compute_anchor_confidence(parsed, anchor_info=None, anchor2_info=None):
    spatial = parsed.get("spatial")
    if spatial not in ("next_to", "behind", "in_front", "above", "below", "between"):
        return 1.0

    if spatial == "between":
        conf1 = float((anchor_info or {}).get("confidence", 0.0))
        conf2 = float((anchor2_info or {}).get("confidence", 0.0))
        if conf1 <= 0.0 or conf2 <= 0.0:
            return 0.0
        return round(min(conf1, conf2), 4)

    return round(float((anchor_info or {}).get("confidence", 0.0)), 4)


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
                     attr_result, anchor_boxes=None, anchor2_boxes=None,
                     anchor_confidence=1.0, anchor2_confidence=1.0):
    semantic_plan = parsed.get("semantic_plan") or {}
    query_type = semantic_plan.get("query_type", "object-centric")
    spatial_score = _compute_spatial_score(
        box_xyxy,
        parsed,
        image_shape=image_shape,
        anchor_boxes=anchor_boxes,
        anchor2_boxes=anchor2_boxes,
        anchor_confidence=anchor_confidence,
        anchor2_confidence=anchor2_confidence,
    )
    scene_score = _scene_consistency_score(parsed, scene_result)
    attr_agent_score = _attribute_agent_score(parsed, attr_result)
    anchor_score = _compute_anchor_confidence(
        parsed,
        anchor_info={"confidence": anchor_confidence},
        anchor2_info={"confidence": anchor2_confidence},
    )
    object_score = float(np.clip((det_score + clip_scores.get("object_score", det_score)) / 2.0, 0.0, 1.0))
    detector_prompt_score = float(np.clip(clip_scores.get("detector_prompt_score", clip_scores.get("full_query_score", 0.0)), 0.0, 1.0))
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

    if query_type == "condition-centric":
        final_score = (
            object_score * 0.24 +
            attribute_score * 0.28 +
            clip_score * 0.24 +
            spatial_score * 0.10 +
            anchor_score * 0.04 +
            scene_score * 0.04 +
            attr_agent_score * 0.03 +
            detector_prompt_score * 0.03
        )
    elif query_type == "relation-centric":
        final_score = (
            object_score * 0.25 +
            attribute_score * 0.18 +
            clip_score * 0.18 +
            spatial_score * 0.18 +
            anchor_score * 0.12 +
            scene_score * 0.05 +
            attr_agent_score * 0.04
        )
    else:
        final_score = (
            object_score * 0.27 +
            attribute_score * 0.22 +
            clip_score * 0.20 +
            spatial_score * 0.16 +
            anchor_score * 0.08 +
            scene_score * 0.04 +
            attr_agent_score * 0.03
        )

    return {
        "query_type": query_type,
        "object_score": round(object_score, 4),
        "attribute_score": round(attribute_score, 4),
        "clip_score": round(clip_score, 4),
        "detector_prompt_score": round(detector_prompt_score, 4),
        "color_score": round(float(clip_scores.get("color_score", 0.0)), 4),
        "condition_score": round(float(clip_scores.get("condition_score", 0.0)), 4),
        "spatial_score": round(spatial_score, 4),
        "anchor_confidence_score": round(anchor_score, 4),
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
                              florence2_backend=None, text_guided_backend=None,
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
    semantic_plan = build_semantic_plan(
        user_prompt=user_prompt,
        parsed=parsed,
        attr_result=attr_result,
        scene_result=scene_result,
    )
    parsed["semantic_plan"] = semantic_plan

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

    # If the attribute agent suggests a detector prompt, only accept it when it
    # stays consistent with the parsed user intent.
    if isinstance(attr_result, dict) and attr_result.get('recommended_prompt'):
        agent_prompt = attr_result['recommended_prompt'].strip()
        # Reject if it looks like template text (LLaVA sometimes copies the template)
        bad_keywords = ['groundingdino', 'optimized', 'detection prompt', 'example', 'template']
        is_template = any(kw in agent_prompt.lower() for kw in bad_keywords)
        if agent_prompt and len(agent_prompt) > 2 and not is_template:
            is_consistent, reason = _is_prompt_consistent_with_query(parsed, agent_prompt)
            if is_consistent:
                print(f"[i] Using agent's recommended prompt: '{agent_prompt}'")
                parsed['object_prompt'] = agent_prompt
            else:
                print(
                    f"[i] Agent prompt rejected ({reason}), using parsed: "
                    f"'{parsed['object_prompt']}'"
                )
        else:
            print(f"[i] Agent prompt rejected (template text), using parsed: '{parsed['object_prompt']}'")

    if semantic_plan.get("detector_prompt"):
        semantic_detector_prompt = str(semantic_plan["detector_prompt"]).strip()
        is_consistent, reason = _is_prompt_consistent_with_query(parsed, semantic_detector_prompt)
        if semantic_detector_prompt and is_consistent:
            parsed["object_prompt"] = semantic_detector_prompt
        else:
            print(f"[i] Semantic controller kept detector prompt unchanged ({reason})")

    grounding_backend = _resolve_text_guided_backend(text_guided_backend, florence2_backend)

    # ---- Step 3: Candidate Detection / Grounding ----
    print(f"[i] Running {grounding_backend} grounding with prompt: '{parsed['object_prompt']}'")

    image_pil = PILImage.fromarray(image_np)
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)

    device = next(gdino_model.parameters()).device
    image_tensor_dev = image_tensor.to(device)

    candidate_output = _run_text_guided_candidate_proposal(
        backend_name=grounding_backend,
        gdino_model=gdino_model,
        florence2_backend=florence2_backend,
        image_pil=image_pil,
        image_tensor_dev=image_tensor_dev,
        parsed=parsed,
        image_size=(H, W),
        box_threshold=box_threshold,
    )
    boxes_filt = candidate_output["boxes_cxcywh"]
    all_boxes_xyxy = candidate_output["boxes_xyxy"]
    all_labels = candidate_output["labels"]
    all_det_scores = candidate_output["scores"]

    print(f"[OK] {grounding_backend} found {len(boxes_filt)} candidates")

    # ---- Retry logic if 0 candidates found ----
    if grounding_backend == "gdino" and len(boxes_filt) == 0:
        # Retry 1: lower threshold
        retry_threshold = 0.20
        print(f"[WARN] 0 candidates — retrying with lower threshold ({retry_threshold})...")
        candidate_output = _run_gdino_candidate_proposal(
            gdino_model=gdino_model,
            image_tensor_dev=image_tensor_dev,
            prompt=parsed["object_prompt"],
            image_size=(H, W),
            box_threshold=retry_threshold,
        )
        boxes_filt = candidate_output["boxes_cxcywh"]
        all_boxes_xyxy = candidate_output["boxes_xyxy"]
        all_labels = candidate_output["labels"]
        all_det_scores = candidate_output["scores"]
        print(f"[i] Retry 1: {len(boxes_filt)} candidates at threshold {retry_threshold}")

    if grounding_backend == "gdino" and len(boxes_filt) == 0 and parsed['object_prompt'] != user_prompt.lower().strip():
        # Retry 2: use raw user prompt
        raw_prompt = user_prompt.lower().strip()
        raw_caption = raw_prompt
        print(f"[WARN] Still 0 — retrying with raw prompt: '{raw_caption}'")
        candidate_output = _run_gdino_candidate_proposal(
            gdino_model=gdino_model,
            image_tensor_dev=image_tensor_dev,
            prompt=raw_prompt,
            image_size=(H, W),
            box_threshold=0.20,
        )
        boxes_filt = candidate_output["boxes_cxcywh"]
        all_boxes_xyxy = candidate_output["boxes_xyxy"]
        all_labels = candidate_output["labels"]
        all_det_scores = candidate_output["scores"]
        print(f"[i] Retry 2: {len(boxes_filt)} candidates with raw prompt")


    # ---- Step 4/5: CLIP + Full-Query Candidate Ranking ----
    clip_pass_mask = []
    clip_scores_all = []
    candidate_details = []
    selected_idx = None
    anchor_boxes = None
    anchor_info = None

    # Detect anchor (reference) object if relational query
    if parsed.get('anchor') and parsed.get('spatial') in ('next_to', 'behind', 'in_front', 'above', 'below', 'between'):
        print(f"[i] Detecting anchor object: '{parsed['anchor']}'...")
        try:
            anchor_info = _run_anchor_detection(gdino_model, image_tensor_dev, parsed['anchor'], H, W)
            anchor_boxes = anchor_info.get("boxes")
            if anchor_boxes is not None and len(anchor_boxes) > 0:
                print(
                    f"[OK] Found {len(anchor_boxes)} anchor object(s): '{parsed['anchor']}' "
                    f"(confidence={anchor_info.get('confidence', 0.0):.2f})"
                )
            else:
                print(f"[WARN] Anchor object '{parsed['anchor']}' not found, falling back to closest")
        except Exception as e:
            print(f"[WARN] Anchor detection failed: {e}")

    # Detect second anchor for "between" queries
    anchor2_boxes = None
    anchor2_info = None
    if parsed.get('anchor2') and parsed.get('spatial') == 'between':
        print(f"[i] Detecting second anchor object: '{parsed['anchor2']}'...")
        try:
            anchor2_info = _run_anchor_detection(gdino_model, image_tensor_dev, parsed['anchor2'], H, W)
            anchor2_boxes = anchor2_info.get("boxes")
            if anchor2_boxes is not None and len(anchor2_boxes) > 0:
                print(
                    f"[OK] Found {len(anchor2_boxes)} second anchor(s): '{parsed['anchor2']}' "
                    f"(confidence={anchor2_info.get('confidence', 0.0):.2f})"
                )
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
                anchor_confidence=float((anchor_info or {}).get("confidence", 0.0 if parsed.get('anchor') else 1.0)),
                anchor2_confidence=float((anchor2_info or {}).get("confidence", 0.0 if parsed.get('anchor2') else 1.0)),
            )
            match_analysis = summarize_candidate_match(parsed, candidate_scores, attr_result)
            candidate_scores = _apply_match_penalties(candidate_scores, match_analysis)
            semantic_judgment = judge_candidate_against_plan(
                semantic_plan=semantic_plan,
                parsed=parsed,
                candidate_scores=candidate_scores,
                clip_scores=clip_score_map,
            )
            candidate_scores = _apply_semantic_judgment(candidate_scores, semantic_judgment)
            candidate_details.append({
                "index": i,
                "label": all_labels[i],
                "det_score": round(float(all_det_scores[i]), 4),
                "clip_passed": bool(clip_pass_mask[-1]),
                "clip_scores": {k: round(float(v), 4) for k, v in clip_score_map.items()},
                "scores": candidate_scores,
                "match_analysis": match_analysis,
                "semantic_judgment": semantic_judgment,
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
                anchor_confidence=float((anchor_info or {}).get("confidence", 0.0 if parsed.get('anchor') else 1.0)),
                anchor2_confidence=float((anchor2_info or {}).get("confidence", 0.0 if parsed.get('anchor2') else 1.0)),
            )
            match_analysis = summarize_candidate_match(parsed, candidate_scores, attr_result)
            candidate_scores = _apply_match_penalties(candidate_scores, match_analysis)
            semantic_judgment = judge_candidate_against_plan(
                semantic_plan=semantic_plan,
                parsed=parsed,
                candidate_scores=candidate_scores,
                clip_scores={},
            )
            candidate_scores = _apply_semantic_judgment(candidate_scores, semantic_judgment)
            candidate_details.append({
                "index": i,
                "label": all_labels[i],
                "det_score": round(float(all_det_scores[i]), 4),
                "clip_passed": True,
                "clip_scores": {},
                "scores": candidate_scores,
                "match_analysis": match_analysis,
                "semantic_judgment": semantic_judgment,
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
        selected_idx, final_masks,
        grounding_backend=grounding_backend,
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
        f"Grounding Backend: {grounding_backend}",
        f"Detection Prompt: '{parsed['object_prompt']}'",
        f"Semantic Query Type: {semantic_plan.get('query_type', 'unknown')}",
        f"Target Object: {parsed.get('target_object', 'unknown')}",
        f"Spatial: {parsed.get('spatial', 'None')}",
        f"Anchor Confidence: {_format_anchor_confidence(parsed, anchor_info, anchor2_info)}",
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
        f"STEP 3 - {grounding_backend}: {n_detected} candidates detected",
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
                f"spatial={scores['spatial_score']:.3f}, anchor={scores.get('anchor_confidence_score', 1.0):.3f}"
            )
            if candidate.get("match_analysis", {}).get("reason"):
                summary_lines.append(f"    Match Analysis: {candidate['match_analysis']['reason']}")
            semantic_judgment = candidate.get("semantic_judgment", {})
            if semantic_judgment:
                summary_lines.append(
                    f"    Semantic Judge: mandatory_ok={semantic_judgment.get('mandatory_satisfied', [])}, "
                    f"mandatory_fail={semantic_judgment.get('mandatory_violations', [])}"
                )
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
        "grounding_backend": grounding_backend,
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
        "semantic_query_type": semantic_plan.get("query_type", "unknown"),
        "semantic_plan": semantic_plan,
        "top_candidates": [
            {
                "candidate_index": candidate["index"] + 1,
                "final_score": candidate["scores"]["final_score"],
                "object_score": candidate["scores"]["object_score"],
                "attribute_score": candidate["scores"]["attribute_score"],
                "clip_score": candidate["scores"]["clip_score"],
                "spatial_score": candidate["scores"]["spatial_score"],
                "anchor_confidence_score": candidate["scores"].get("anchor_confidence_score", 1.0),
                "reason": candidate.get("match_analysis", {}).get("reason", ""),
                "satisfied_constraints": candidate.get("match_analysis", {}).get("satisfied_constraints", []),
                "violated_constraints": candidate.get("match_analysis", {}).get("violated_constraints", []),
                "semantic_judgment": candidate.get("semantic_judgment", {}),
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
            "grounding_backend": grounding_backend,
            "detection_prompt": parsed['object_prompt'],
            "semantic_plan": semantic_plan,
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
        "semantic_plan": semantic_plan,
        "grounding_backend": grounding_backend,
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
