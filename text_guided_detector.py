#!/usr/bin/env python3
"""
Text-Guided VLM Detection Module
Multi-agent approach: two LLaVA agents analyze the scene and match
the user's query before detection, verification, and segmentation.

Pipeline:
  Step 1: Scene Understanding Agent (LLaVA) - lists all objects
  Step 2: Attribute Matching Agent (LLaVA) - reasons about which object matches the query
  Step 3: Candidate Detection (GroundingDINO) - find all matches
  Step 4: Semantic Verification (CLIP) - filter false positives
  Step 5: Spatial Filtering (rule-based) - pick correct one
  Step 6: Pixel Segmentation (SAM) - precise mask
"""

import os
import re
import json
import cv2
import numpy as np
import torch
import tempfile
from PIL import Image


# =====================
# Step 1: Scene Understanding
# =====================

SCENE_SYSTEM_PROMPT = """You are an expert scene analyzer. Analyze the given image and list ALL visible objects.
For each object, provide:
1. Object name/type
2. Approximate position (left, center-left, center, center-right, right)
3. Color/appearance
4. Size relative to scene (small, medium, large)

Return your analysis as valid JSON:
{
  "scene_type": "description of the scene",
  "lighting": "bright/dim/dark/night",
  "total_objects": N,
  "objects": [
    {"name": "red car", "position": "left", "color": "red", "size": "medium"},
    {"name": "pedestrian", "position": "center", "color": "dark clothing", "size": "small"}
  ]
}

List EVERY visible object. Be thorough."""


def scene_understanding(image_path):
    """
    Run LLaVA scene analysis to list all objects in the image.
    
    Args:
        image_path: path to the image file
        
    Returns:
        dict with scene analysis results
    """
    from src.agents.vlm_backend import run_vlm
    
    messages = [
        {"role": "system", "content": SCENE_SYSTEM_PROMPT},
        {"role": "user", "content": "Analyze this image and list all visible objects with their positions and attributes. Return valid JSON only."}
    ]
    
    print("[i] Text-Guided: Running scene understanding (LLaVA)...")
    output = run_vlm(messages, image_path=image_path)
    print(f"[OK] Scene analysis complete")
    
    # Parse JSON from LLaVA output
    return _parse_json_response(output)


def _parse_json_response(response):
    """Robustly parse JSON from LLaVA output."""
    # Clean LaTeX-style underscores
    response = response.replace("\\_", "_")
    
    # Try direct parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try extracting JSON block
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{[\s\S]*\}',
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1) if '```' in pattern else match.group(0))
            except (json.JSONDecodeError, IndexError):
                continue
    
    # Fallback
    return {"scene_type": "unknown", "objects": [], "raw_output": response}


# =====================
# Step 2: Attribute Matching Agent
# =====================

ATTRIBUTE_AGENT_PROMPT = """You are an expert object matching agent. You will be given:
1. A scene analysis listing all objects in an image
2. A user's search query describing a specific object

Your task: Analyze which object(s) from the scene best match the user's query.
Consider: color, type, position, size, and any other attributes mentioned.

Return your analysis as valid JSON:
{
  "query": "the user's original query",
  "matched_objects": [
    {"name": "object name", "position": "position", "confidence": "high/medium/low", "reason": "why this matches"}
  ],
  "recommended_prompt": "optimized detection prompt for GroundingDINO",
  "ambiguity": "none/low/high",
  "reasoning": "brief explanation of matching logic"
}"""


def attribute_matching_agent(image_path, scene_result, user_prompt):
    """
    Run LLaVA to reason about which object matches the user's query.
    
    Args:
        image_path: path to image file
        scene_result: dict from scene_understanding()
        user_prompt: original user query
        
    Returns:
        dict with matching analysis
    """
    from src.agents.vlm_backend import run_vlm
    
    # Build context from scene analysis
    scene_text = json.dumps(scene_result, indent=2) if isinstance(scene_result, dict) else str(scene_result)
    
    messages = [
        {"role": "system", "content": ATTRIBUTE_AGENT_PROMPT},
        {"role": "user", "content": f"Scene Analysis:\n{scene_text}\n\nUser Query: \"{user_prompt}\"\n\nWhich object(s) match this query? Return valid JSON only."}
    ]
    
    print("[i] Text-Guided: Running attribute matching agent (LLaVA)...")
    output = run_vlm(messages, image_path=image_path)
    print(f"[OK] Attribute matching complete")
    
    result = _parse_json_response(output)
    
    # Use the agent's recommended prompt if available
    if isinstance(result, dict) and result.get("recommended_prompt"):
        print(f"[i] Agent recommended prompt: '{result['recommended_prompt']}'")
    
    return result


# =====================
# Query Parsing (internal helper)
# =====================

SPATIAL_TERMS = {
    "left": "left",
    "leftmost": "left", 
    "right": "right",
    "rightmost": "right",
    "center": "center",
    "middle": "center",
    "top": "top",
    "bottom": "bottom",
    "upper": "top",
    "lower": "bottom",
    "nearest": "nearest",
    "closest": "nearest",
    "farthest": "farthest",
    "largest": "largest",
    "biggest": "largest",
    "smallest": "smallest",
    "front": "front",
    "back": "back",
    "behind": "behind",
    "near": "near",
}

COLOR_TERMS = [
    "red", "blue", "green", "yellow", "white", "black", "grey", "gray",
    "silver", "brown", "orange", "purple", "pink", "gold", "dark", "light",
    "bright", "beige", "maroon", "navy", "cyan", "teal",
]


def parse_query(user_prompt):
    """
    Parse user text prompt into structured query components.
    
    Args:
        user_prompt: e.g., "the grey car on the left"
        
    Returns:
        dict with keys: original, object_prompt, attribute, spatial, detect_all
    """
    prompt = user_prompt.lower().strip()
    
    # Remove common prefixes
    for prefix in ["find the", "detect the", "locate the", "show the", 
                   "find", "detect", "locate", "show", "get the", "get"]:
        if prompt.startswith(prefix):
            prompt = prompt[len(prefix):].strip()
            break
    
    # Extract spatial term
    spatial = None
    spatial_phrase = None
    for term, normalized in SPATIAL_TERMS.items():
        # Check patterns like "on the left", "at the right", "in the center"
        for pattern in [f"on the {term}", f"at the {term}", f"in the {term}",
                       f"to the {term}", f"the {term} side", f"{term}most",
                       f"{term} side"]:
            if pattern in prompt:
                spatial = normalized
                spatial_phrase = pattern
                break
        # Also check standalone spatial term at the end
        if spatial is None and prompt.endswith(term):
            spatial = normalized
            spatial_phrase = term
        if spatial:
            break
    
    # Remove spatial phrase from prompt to get the object description
    object_desc = prompt
    if spatial_phrase:
        object_desc = prompt.replace(spatial_phrase, "").strip()
    # Clean trailing prepositions
    for prep in [" on", " at", " in", " to", " from"]:
        if object_desc.endswith(prep):
            object_desc = object_desc[:-len(prep)].strip()
    
    # Extract color/attribute
    attribute = None
    for color in COLOR_TERMS:
        if color in object_desc:
            attribute = color
            break
    
    # Determine if user wants all instances or just one
    detect_all = spatial is None  # No spatial term = find all
    
    # Build the GroundingDINO prompt (object + attribute, no spatial)
    gdino_prompt = object_desc.strip()
    if not gdino_prompt:
        gdino_prompt = user_prompt.strip()
    
    result = {
        "original": user_prompt,
        "object_prompt": gdino_prompt,
        "attribute": attribute,
        "spatial": spatial,
        "detect_all": detect_all,
    }
    
    print(f"[i] Query parsed: object='{gdino_prompt}', attribute={attribute}, spatial={spatial}, detect_all={detect_all}")
    return result


# =====================
# Step 5: Spatial Filtering
# =====================

def spatial_filter(boxes_xyxy, spatial_term, image_shape=None):
    """
    Filter bounding boxes by spatial term.
    
    Args:
        boxes_xyxy: tensor (N, 4) in [x1, y1, x2, y2] format
        spatial_term: one of 'left', 'right', 'center', 'largest', etc.
        image_shape: (H, W) tuple for reference
        
    Returns:
        index of the selected box
    """
    if len(boxes_xyxy) == 0:
        return None
    if len(boxes_xyxy) == 1:
        return 0
    
    boxes = boxes_xyxy.numpy() if torch.is_tensor(boxes_xyxy) else np.array(boxes_xyxy)
    
    # Compute box properties
    x_centers = (boxes[:, 0] + boxes[:, 2]) / 2
    y_centers = (boxes[:, 1] + boxes[:, 3]) / 2
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    if spatial_term == "left":
        return int(np.argmin(x_centers))
    elif spatial_term == "right":
        return int(np.argmax(x_centers))
    elif spatial_term == "center":
        if image_shape:
            img_center_x = image_shape[1] / 2
        else:
            img_center_x = np.mean(x_centers)
        return int(np.argmin(np.abs(x_centers - img_center_x)))
    elif spatial_term == "top":
        return int(np.argmin(y_centers))
    elif spatial_term == "bottom":
        return int(np.argmax(y_centers))
    elif spatial_term == "largest":
        return int(np.argmax(areas))
    elif spatial_term == "smallest":
        return int(np.argmin(areas))
    elif spatial_term == "nearest":
        # Nearest to camera = largest y_center (bottom of image)
        return int(np.argmax(y_centers))
    elif spatial_term == "farthest":
        return int(np.argmin(y_centers))
    else:
        # Default: return the first (highest confidence) box
        return 0


# =====================
# Step Visualization
# =====================

def generate_step_visualizations(image_np, scene_result, parsed_query,
                                  all_boxes_xyxy, all_labels, all_scores,
                                  clip_mask, clip_scores,
                                  selected_idx, final_masks):
    """
    Generate visualization images for each pipeline step.
    
    Returns:
        dict of step_name -> numpy RGB image
    """
    H, W = image_np.shape[:2]
    steps = {}
    
    # ---- Step 1: Scene Understanding ----
    step1 = image_np.copy()
    scene_text = []
    if isinstance(scene_result, dict):
        scene_type = scene_result.get("scene_type", "Unknown scene")
        lighting = scene_result.get("lighting", "unknown")
        objects = scene_result.get("objects", [])
        scene_text.append(f"Scene: {scene_type}")
        scene_text.append(f"Lighting: {lighting}")
        scene_text.append(f"Objects found: {len(objects)}")
        for i, obj in enumerate(objects[:10]):  # Show max 10
            name = obj.get("name", "?")
            pos = obj.get("position", "?")
            color = obj.get("color", "")
            scene_text.append(f"  {i+1}. {name} ({color}, {pos})")
    else:
        scene_text.append("Scene analysis unavailable")
    
    # Draw text overlay on semi-transparent background
    overlay = step1.copy()
    box_h = min(30 + len(scene_text) * 25, H - 20)
    cv2.rectangle(overlay, (10, 10), (min(450, W - 10), box_h), (0, 0, 0), -1)
    step1 = cv2.addWeighted(overlay, 0.7, step1, 0.3, 0)
    for i, text in enumerate(scene_text):
        cv2.putText(step1, text, (20, 35 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    # Title
    cv2.putText(step1, "STEP 1: Scene Understanding (LLaVA)", (10, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    steps["step1_scene"] = step1
    
    # ---- Step 2: Attribute Matching Agent ----
    step2 = image_np.copy()
    overlay2 = step2.copy()
    cv2.rectangle(overlay2, (10, 10), (min(500, W - 10), 220), (0, 0, 0), -1)
    step2 = cv2.addWeighted(overlay2, 0.7, step2, 0.3, 0)
    
    agent2_info = [
        f"Query: \"{parsed_query['original']}\"",
        f"Detection Prompt: \"{parsed_query['object_prompt']}\"",
        f"Attribute: {parsed_query.get('attribute', 'None')}",
        f"Spatial: {parsed_query.get('spatial', 'None (detect all)')}",
    ]
    # Add attribute agent reasoning if available
    attr_result = parsed_query.get('attr_agent_result', {})
    if isinstance(attr_result, dict):
        reasoning = attr_result.get('reasoning', '')
        ambiguity = attr_result.get('ambiguity', 'N/A')
        if reasoning:
            agent2_info.append(f"Reasoning: {reasoning[:80]}")
        agent2_info.append(f"Ambiguity: {ambiguity}")
        matched = attr_result.get('matched_objects', [])
        for m in matched[:3]:
            agent2_info.append(f"  Match: {m.get('name','')} ({m.get('position','')}) [{m.get('confidence','')}]")
    
    for i, text in enumerate(agent2_info):
        cv2.putText(step2, text, (20, 35 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)
    cv2.putText(step2, "STEP 2: Attribute Matching Agent (LLaVA)", (10, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    steps["step2_query"] = step2
    
    # ---- Step 3: All Candidates (GroundingDINO) ----
    step3 = image_np.copy()
    if all_boxes_xyxy is not None and len(all_boxes_xyxy) > 0:
        boxes_np = all_boxes_xyxy.numpy() if torch.is_tensor(all_boxes_xyxy) else np.array(all_boxes_xyxy)
        for i, (box, label) in enumerate(zip(boxes_np, all_labels)):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(step3, (x1, y1), (x2, y2), (0, 255, 255), 3)
            score_txt = f"#{i+1} {label}"
            cv2.putText(step3, score_txt, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        n_found = len(all_boxes_xyxy)
    else:
        n_found = 0
    cv2.putText(step3, f"STEP 3: GroundingDINO - {n_found} candidates found", (10, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    steps["step3_candidates"] = step3
    
    # ---- Step 4: CLIP Verification ----
    step4 = image_np.copy()
    if all_boxes_xyxy is not None and len(all_boxes_xyxy) > 0:
        boxes_np = all_boxes_xyxy.numpy() if torch.is_tensor(all_boxes_xyxy) else np.array(all_boxes_xyxy)
        n_passed = 0
        n_rejected = 0
        for i, box in enumerate(boxes_np):
            x1, y1, x2, y2 = box.astype(int)
            passed = clip_mask[i] if clip_mask is not None and i < len(clip_mask) else True
            if passed:
                color = (0, 255, 0)  # Green = passed
                status = "PASS"
                score = clip_scores[i] if clip_scores and i < len(clip_scores) else 0
                n_passed += 1
            else:
                color = (0, 0, 255)  # Red = rejected
                status = "REJECT"
                score = clip_scores[i] if clip_scores and i < len(clip_scores) else 0
                n_rejected += 1
            cv2.rectangle(step4, (x1, y1), (x2, y2), color, 3)
            cv2.putText(step4, f"#{i+1} {status} ({score:.2f})", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        summary = f"STEP 4: CLIP Verification - {n_passed} passed, {n_rejected} rejected"
    else:
        summary = "STEP 4: CLIP Verification - No candidates"
    cv2.putText(step4, summary, (10, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    steps["step4_clip"] = step4
    
    # ---- Step 5: Spatial Selection ----
    step5 = image_np.copy()
    if all_boxes_xyxy is not None and len(all_boxes_xyxy) > 0 and selected_idx is not None:
        boxes_np = all_boxes_xyxy.numpy() if torch.is_tensor(all_boxes_xyxy) else np.array(all_boxes_xyxy)
        # Draw all boxes in grey (not selected)
        for i, box in enumerate(boxes_np):
            if clip_mask is not None and i < len(clip_mask) and not clip_mask[i]:
                continue  # Skip rejected boxes
            x1, y1, x2, y2 = box.astype(int)
            if isinstance(selected_idx, list):
                is_selected = i in selected_idx
            else:
                is_selected = (i == selected_idx)
            
            if is_selected:
                cv2.rectangle(step5, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(step5, "SELECTED", (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(step5, (x1, y1), (x2, y2), (128, 128, 128), 2)
                cv2.putText(step5, "skipped", (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA)
    
    spatial_text = parsed_query.get('spatial', 'all')
    if spatial_text:
        cv2.putText(step5, f"STEP 5: Spatial Filter - '{spatial_text}'", (10, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(step5, "STEP 5: No spatial filter - keeping all", (10, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    steps["step5_spatial"] = step5
    
    # ---- Step 6: Final Segmentation ----
    step6 = image_np.copy().astype(np.float32)
    if final_masks is not None and len(final_masks) > 0:
        colors = [
            [0, 255, 100],   # green
            [255, 0, 100],   # pink
            [255, 165, 0],   # orange
            [0, 255, 255],   # cyan
        ]
        for i, mask in enumerate(final_masks):
            mask_np = mask.squeeze().numpy().astype(bool) if torch.is_tensor(mask) else mask.astype(bool)
            color = np.array(colors[i % len(colors)], dtype=np.float32)
            step6[mask_np] = step6[mask_np] * 0.35 + color * 0.65
    step6 = step6.astype(np.uint8)
    
    # Draw final box(es)
    if all_boxes_xyxy is not None and selected_idx is not None:
        boxes_np = all_boxes_xyxy.numpy() if torch.is_tensor(all_boxes_xyxy) else np.array(all_boxes_xyxy)
        indices = selected_idx if isinstance(selected_idx, list) else [selected_idx]
        for idx in indices:
            if idx < len(boxes_np):
                x1, y1, x2, y2 = boxes_np[idx].astype(int)
                cv2.rectangle(step6, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    cv2.putText(step6, "STEP 6: SAM Segmentation - Final Result", (10, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    steps["step6_final"] = step6
    
    return steps


# =====================
# Main Pipeline
# =====================

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
        
    Returns:
        dict with keys:
            step_images: dict of step name -> RGB numpy image
            scene_result: scene analysis JSON
            parsed_query: parsed query dict
            final_boxes: selected bounding boxes
            final_masks: SAM segmentation masks
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
    import gc
    try:
        from src.agents.vlm_backend import _model, _processor
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
    
    # If agent recommended a better prompt, use it
    if isinstance(attr_result, dict) and attr_result.get('recommended_prompt'):
        agent_prompt = attr_result['recommended_prompt'].strip()
        if agent_prompt and len(agent_prompt) > 2:
            print(f"[i] Using agent's recommended prompt: '{agent_prompt}'")
            parsed['object_prompt'] = agent_prompt
    
    # ---- Step 3: Candidate Detection (GroundingDINO) ----
    print(f"[i] Running GroundingDINO with prompt: '{parsed['object_prompt']}'")
    
    # Preprocess image for GroundingDINO
    image_pil = PILImage.fromarray(image_np)
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)
    
    device = next(gdino_model.parameters()).device
    
    # Get all detections
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
    
    # Convert to xyxy for visualization and CLIP
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
            
            similarity = clip_verifier.compute_similarity(crop, parsed['object_prompt'])
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
    
    # Get only passed boxes
    passed_indices = [i for i, passed in enumerate(clip_pass_mask) if passed]
    
    # ---- Step 5: Spatial Filtering ----
    selected_idx = None
    if len(passed_indices) > 0:
        if parsed['spatial'] and not parsed['detect_all']:
            # Filter by spatial term among passed boxes
            passed_boxes = all_boxes_xyxy[passed_indices]
            local_idx = spatial_filter(passed_boxes, parsed['spatial'], image_shape=(H, W))
            if local_idx is not None:
                selected_idx = passed_indices[local_idx]
                print(f"[OK] Spatial filter '{parsed['spatial']}' selected candidate #{selected_idx+1}")
            else:
                selected_idx = passed_indices[0]
        else:
            # Keep all passed boxes
            selected_idx = passed_indices
            print(f"[OK] Keeping all {len(passed_indices)} verified candidates")
    
    # ---- Step 6: SAM Segmentation ----
    final_masks = None
    final_boxes_for_sam = None
    
    if selected_idx is not None:
        # Get the boxes to segment
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
    
    # Attribute Agent summary
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
