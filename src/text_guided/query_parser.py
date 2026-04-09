"""
Query Parser and Spatial Filter
Parses user text prompts into structured components and filters
bounding boxes by spatial terms.

Supports two modes:
1. Structured rules parsing (fast, reliable fallback)
2. LLaVA-assisted parsing (advanced natural language understanding)
"""

import math
import re
import json
import numpy as np
import torch


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

# Relational phrases that require an anchor (reference) object
RELATIONAL_PHRASES = [
    "next to the", "beside the", "near the", "close to the",
    "next to a", "beside a", "near a", "close to a",
    "behind the", "in front of the", "above the", "below the",
    "behind a", "in front of a", "above a", "below a",
]

COLOR_TERMS = [
    "red", "blue", "green", "yellow", "white", "black", "grey", "gray",
    "silver", "brown", "orange", "purple", "pink", "gold", "dark", "light",
    "bright", "beige", "maroon", "navy", "cyan", "teal",
]

CONDITION_TERMS = [
    "damaged", "broken", "burning", "parked", "moving", "stopped",
    "crashed", "overturned", "tilted", "fallen", "open", "closed",
    "old", "new", "dirty", "clean", "blurred", "far", "nearby",
]

COUNT_TERMS = {
    "single": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
}

UNIQUENESS_HINTS = [
    "the only",
    "single",
    "unique",
    "exact",
]


def _extract_terms(text, term_list):
    found = []
    for term in term_list:
        if re.search(rf"\b{re.escape(term)}\b", text):
            found.append(term)
    return found


def _normalize_anchor(anchor_text):
    if not anchor_text:
        return None

    anchor_text = anchor_text.strip(" .,")
    anchor_text = re.sub(r"^(the|a|an)\s+", "", anchor_text)
    return anchor_text.strip() or None


def _guess_target_object(object_desc):
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]*", object_desc.lower())
    stopwords = {
        "the", "a", "an", "on", "at", "in", "to", "from", "of", "with",
        "near", "next", "side", "left", "right", "top", "bottom", "middle",
        "center", "between", "behind", "front",
    }
    filtered = [
        token for token in tokens
        if token not in stopwords and token not in COLOR_TERMS and token not in CONDITION_TERMS
    ]
    return filtered[-1] if filtered else object_desc.strip() or None


def _build_priority_order(target_object, attributes, spatial, anchor, count):
    priority = []
    if target_object:
        priority.append("object")
    if attributes:
        if attributes.get("color"):
            priority.append("color")
        if attributes.get("condition"):
            priority.append("condition")
        extras = attributes.get("descriptors") or []
        if extras:
            priority.append("attributes")
    if anchor:
        priority.append("anchor_relation")
    if spatial and spatial not in ("next_to", "behind", "in_front", "above", "below", "between"):
        priority.append("region")
    if count:
        priority.append("count")
    return priority or ["object"]


def _build_query_result(user_prompt, detection_prompt, target_object=None, color=None,
                        conditions=None, descriptors=None, spatial=None, anchor=None,
                        anchor2=None, ordinal=None, ordinal_direction=None,
                        parser_mode="structured rules"):
    target_object = target_object or _guess_target_object(detection_prompt or user_prompt)
    conditions = [c for c in (conditions or []) if c]
    descriptors = [d for d in (descriptors or []) if d]
    color = color or None
    attribute = color or (conditions[0] if conditions else (descriptors[0] if descriptors else None))

    count = None
    for term, value in COUNT_TERMS.items():
        if re.search(rf"\b{re.escape(term)}\b", user_prompt.lower()):
            count = value
            break

    uniqueness = any(hint in user_prompt.lower() for hint in UNIQUENESS_HINTS) or bool(ordinal)
    attributes = {
        "color": color,
        "condition": conditions[0] if conditions else None,
        "conditions": conditions,
        "descriptors": descriptors,
    }
    spatial_constraints = {
        "relation": spatial if spatial in ("next_to", "behind", "in_front", "above", "below", "between") else None,
        "anchor": _normalize_anchor(anchor),
        "anchor2": _normalize_anchor(anchor2),
        "region_bias": spatial if spatial in ("left", "right", "center", "top", "bottom") else None,
        "size_bias": spatial if spatial in ("largest", "smallest") else None,
        "depth_bias": spatial if spatial in ("nearest", "farthest", "ahead") else None,
        "ordinal": ordinal,
        "ordinal_direction": ordinal_direction,
    }
    priority_order = _build_priority_order(
        target_object,
        attributes,
        spatial,
        anchor,
        count,
    )

    result = {
        "original": user_prompt,
        "object_prompt": (detection_prompt or user_prompt).strip(),
        "target_object": target_object,
        "attribute": attribute,
        "attributes": attributes,
        "spatial": spatial,
        "spatial_constraints": spatial_constraints,
        "detect_all": spatial is None,
        "anchor": _normalize_anchor(anchor),
        "anchor2": _normalize_anchor(anchor2),
        "ordinal": ordinal,
        "ordinal_direction": ordinal_direction,
        "count": count,
        "unique_expected": uniqueness,
        "match_type": "single_best_match" if uniqueness or count == 1 or count is None else "multi_match",
        "priority_order": priority_order,
        "parser_mode": parser_mode,
        "full_query_prompt": user_prompt.strip(),
    }
    return result


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
        for pattern in [f"on the {term}", f"at the {term}", f"in the {term}",
                       f"to the {term}", f"the {term} side", f"{term}most",
                       f"{term} side"]:
            if pattern in prompt:
                spatial = normalized
                spatial_phrase = pattern
                break
        if spatial is None and prompt.endswith(term):
            spatial = normalized
            spatial_phrase = term
        if spatial:
            break

    # Remove spatial phrase from prompt to get the object description
    object_desc = prompt
    if spatial_phrase:
        object_desc = prompt.replace(spatial_phrase, "").strip()
    for prep in [" on", " at", " in", " to", " from"]:
        if object_desc.endswith(prep):
            object_desc = object_desc[:-len(prep)].strip()

    # Extract relational anchor object (e.g. "the car next to the truck" → anchor="truck")
    anchor = None
    anchor_phrase = None
    for rel_phrase in RELATIONAL_PHRASES:
        if rel_phrase in prompt:
            # Everything after the relational phrase is the anchor object
            anchor_part = prompt.split(rel_phrase, 1)[1].strip()
            # Clean up anchor
            for prep in [" on", " at", " in", " to"]:
                if anchor_part.endswith(prep):
                    anchor_part = anchor_part[:-len(prep)].strip()
            if anchor_part:
                anchor = anchor_part
                anchor_phrase = rel_phrase + anchor_part
                # Set spatial to relational type
                if "next to" in rel_phrase or "beside" in rel_phrase or "near" in rel_phrase or "close to" in rel_phrase:
                    spatial = "next_to"
                elif "behind" in rel_phrase:
                    spatial = "behind"
                elif "in front of" in rel_phrase:
                    spatial = "in_front"
                elif "above" in rel_phrase:
                    spatial = "above"
                elif "below" in rel_phrase:
                    spatial = "below"
                break

    # Remove anchor phrase from object description
    if anchor_phrase:
        object_desc = prompt.replace(anchor_phrase, "").strip()
        # Clean trailing prepositions
        for prep in [" on", " at", " in", " to", " from"]:
            if object_desc.endswith(prep):
                object_desc = object_desc[:-len(prep)].strip()

    colors = _extract_terms(object_desc, COLOR_TERMS)
    conditions = _extract_terms(object_desc, CONDITION_TERMS)

    detection_prompt = object_desc.strip()
    if not detection_prompt:
        detection_prompt = user_prompt.strip()

    result = _build_query_result(
        user_prompt=user_prompt,
        detection_prompt=detection_prompt,
        target_object=_guess_target_object(object_desc),
        color=colors[0] if colors else None,
        conditions=conditions,
        descriptors=[d for d in object_desc.split() if d not in colors and d not in conditions],
        spatial=spatial,
        anchor=anchor,
        parser_mode="structured rules",
    )

    anchor_info = f", anchor='{anchor}'" if anchor else ""
    print(
        f"[i] Query parsed: object='{detection_prompt}', target='{result.get('target_object')}', "
        f"attribute={result.get('attribute')}, spatial={spatial}{anchor_info}, "
        f"priorities={result.get('priority_order')}"
    )
    return result


def llava_parse_query(user_prompt, image_path=None):
    """
    Advanced query parser using LLaVA to understand more natural language
    than the rule parser can safely cover.

    Falls back to the structured rules parser if the LLaVA response is not
    usable, so the UI keeps working even when the VLM parse is imperfect.
    """
    print(f"[i] LLaVA Query Parser: parsing '{user_prompt}'...")

    parse_prompt = f"""You are a query parser for an object detection system.
Given the user's search query, extract structured information as JSON.

User query: "{user_prompt}"

Return ONLY valid JSON with these fields:
{{
  "object": "main object to detect (e.g. car, truck, pedestrian)",
  "color": "color if mentioned, else null",
  "spatial": "spatial relationship: left/right/center/between/ahead/behind/above/below/nearest/farthest/largest/smallest/next_to/in_front, else null",
  "anchor": "reference object if relational query (e.g. 'next to the truck' -> 'truck'), else null",
  "anchor2": "second reference object if between query (e.g. 'between truck and bus' -> 'bus'), else null",
  "ordinal": "ordinal position if mentioned (e.g. 'second from right' -> 2), else null",
  "ordinal_direction": "direction for ordinal: 'left'/'right', else null",
  "attribute": "other descriptor if only one is obvious (parked, damaged, moving, large, small), else null",
  "attributes": ["list of all meaningful descriptors beyond the object category"],
  "condition": "main state/condition descriptor if present, else null",
  "count": "number requested if explicit, else null",
  "unique_expected": "true if the user implies a single exact object, else false",
  "priority_order": ["ordered list of constraints by importance such as object, color, relation, region, condition"],
  "detection_prompt": "short phrase for object detector (combine object + the most salient appearance words)"
}}

Examples:
- "the red car on the left" -> {{"object": "car", "color": "red", "spatial": "left", "anchor": null, "anchor2": null, "ordinal": null, "ordinal_direction": null, "attribute": null, "detection_prompt": "red car"}}
- "the truck between the car and the bus" -> {{"object": "truck", "color": null, "spatial": "between", "anchor": "car", "anchor2": "bus", "ordinal": null, "ordinal_direction": null, "attribute": null, "detection_prompt": "truck"}}
- "the second vehicle from the right" -> {{"object": "vehicle", "color": null, "spatial": "right", "anchor": null, "anchor2": null, "ordinal": 2, "ordinal_direction": "right", "attribute": null, "detection_prompt": "vehicle"}}
- "the damaged car near the traffic signal" -> {{"object": "car", "color": null, "spatial": "next_to", "anchor": "traffic signal", "anchor2": null, "ordinal": null, "ordinal_direction": null, "attribute": "damaged", "detection_prompt": "damaged car"}}

Return ONLY the JSON, nothing else."""

    try:
        from src.agents.vlm_backend import ask_vlm_text_only

        raw = ask_vlm_text_only(parse_prompt)
        text = raw.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)

        parsed_llm = json.loads(text)

        spatial_map = {
            "left": "left",
            "right": "right",
            "center": "center",
            "middle": "center",
            "top": "top",
            "bottom": "bottom",
            "largest": "largest",
            "smallest": "smallest",
            "nearest": "nearest",
            "closest": "nearest",
            "farthest": "farthest",
            "between": "between",
            "ahead": "ahead",
            "behind": "behind",
            "above": "above",
            "below": "below",
            "next_to": "next_to",
            "next to": "next_to",
            "near": "next_to",
            "in front": "in_front",
            "in_front": "in_front",
        }

        llm_spatial = parsed_llm.get("spatial")
        spatial = spatial_map.get(llm_spatial, llm_spatial) if llm_spatial else None

        detection_prompt = (parsed_llm.get("detection_prompt") or "").strip()
        if not detection_prompt:
            parts = []
            if parsed_llm.get("color"):
                parts.append(parsed_llm["color"])
            if parsed_llm.get("attribute"):
                parts.append(parsed_llm["attribute"])
            if parsed_llm.get("object"):
                parts.append(parsed_llm["object"])
            detection_prompt = " ".join(parts) if parts else user_prompt

        ordinal = parsed_llm.get("ordinal")
        if ordinal is not None:
            try:
                ordinal = int(ordinal)
            except (TypeError, ValueError):
                ordinal = None

        extra_attributes = parsed_llm.get("attributes") or []
        if isinstance(extra_attributes, str):
            extra_attributes = [extra_attributes]
        color = parsed_llm.get("color")
        attribute = parsed_llm.get("attribute")
        condition = parsed_llm.get("condition") or (
            attribute if attribute and attribute not in COLOR_TERMS else None
        )
        descriptors = [item for item in extra_attributes if item and item != condition and item != color]

        result = _build_query_result(
            user_prompt=user_prompt,
            detection_prompt=detection_prompt,
            target_object=parsed_llm.get("object"),
            color=color,
            conditions=[condition] if condition else [],
            descriptors=descriptors,
            spatial=spatial,
            anchor=parsed_llm.get("anchor"),
            anchor2=parsed_llm.get("anchor2"),
            ordinal=ordinal,
            ordinal_direction=parsed_llm.get("ordinal_direction"),
            parser_mode="llava-assisted",
        )
        if parsed_llm.get("count") not in (None, ""):
            try:
                result["count"] = int(parsed_llm.get("count"))
            except (TypeError, ValueError):
                pass
        if isinstance(parsed_llm.get("unique_expected"), bool):
            result["unique_expected"] = parsed_llm["unique_expected"]
        if parsed_llm.get("priority_order"):
            result["priority_order"] = parsed_llm["priority_order"]

        anchor_info = f", anchor='{result['anchor']}'" if result.get("anchor") else ""
        anchor2_info = f", anchor2='{result['anchor2']}'" if result.get("anchor2") else ""
        ordinal_info = f", ordinal={ordinal}" if ordinal else ""
        print(
            f"[OK] LLaVA parsed: object='{detection_prompt}', spatial={spatial}"
            f"{anchor_info}{anchor2_info}{ordinal_info}"
        )
        return result

    except Exception as e:
        print(f"[WARN] LLaVA parser failed ({e}), falling back to structured rules")
        result = parse_query(user_prompt)
        result["parser_mode"] = "structured rules (fallback)"
        return result


def spatial_filter(boxes_xyxy, spatial_term, image_shape=None, anchor_boxes=None,
                   anchor2_boxes=None, ordinal=None, ordinal_direction=None):
    """
    Filter bounding boxes by spatial term.

    Args:
        boxes_xyxy: tensor (N, 4) in [x1, y1, x2, y2] format
        spatial_term: one of 'left', 'right', 'center', 'largest', 'next_to', etc.
        image_shape: (H, W) tuple for reference
        anchor_boxes: tensor (M, 4) of anchor/reference object boxes (for relational queries)

    Returns:
        index of the selected box
    """
    if len(boxes_xyxy) == 0:
        return None
    if len(boxes_xyxy) == 1:
        return 0

    boxes = boxes_xyxy.numpy() if torch.is_tensor(boxes_xyxy) else np.array(boxes_xyxy)

    x_centers = (boxes[:, 0] + boxes[:, 2]) / 2
    y_centers = (boxes[:, 1] + boxes[:, 3]) / 2
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # --- Relational spatial terms (require anchor) ---
    if spatial_term in ("next_to", "behind", "in_front", "above", "below") and anchor_boxes is not None:
        anchor = anchor_boxes.numpy() if torch.is_tensor(anchor_boxes) else np.array(anchor_boxes)
        if len(anchor) > 0:
            # Use the center of the first (best) anchor box
            anchor_cx = (anchor[0, 0] + anchor[0, 2]) / 2
            anchor_cy = (anchor[0, 1] + anchor[0, 3]) / 2

            if spatial_term == "next_to":
                # Pick the target closest to the anchor (Euclidean distance)
                distances = np.sqrt((x_centers - anchor_cx)**2 + (y_centers - anchor_cy)**2)
                return int(np.argmin(distances))
            elif spatial_term == "behind":
                # "Behind" in road scenes = further up in image (smaller y)
                above_mask = y_centers < anchor_cy
                if above_mask.any():
                    candidates = np.where(above_mask)[0]
                    dists = np.sqrt((x_centers[candidates] - anchor_cx)**2 + (y_centers[candidates] - anchor_cy)**2)
                    return int(candidates[np.argmin(dists)])
            elif spatial_term == "in_front":
                # "In front" = closer to camera (larger y)
                below_mask = y_centers > anchor_cy
                if below_mask.any():
                    candidates = np.where(below_mask)[0]
                    dists = np.sqrt((x_centers[candidates] - anchor_cx)**2 + (y_centers[candidates] - anchor_cy)**2)
                    return int(candidates[np.argmin(dists)])
            elif spatial_term == "above":
                above_mask = y_centers < anchor_cy
                if above_mask.any():
                    candidates = np.where(above_mask)[0]
                    dists = np.abs(x_centers[candidates] - anchor_cx)
                    return int(candidates[np.argmin(dists)])
            elif spatial_term == "below":
                below_mask = y_centers > anchor_cy
                if below_mask.any():
                    candidates = np.where(below_mask)[0]
                    dists = np.abs(x_centers[candidates] - anchor_cx)
                    return int(candidates[np.argmin(dists)])

            # Fallback for relational: pick closest
            distances = np.sqrt((x_centers - anchor_cx)**2 + (y_centers - anchor_cy)**2)
            return int(np.argmin(distances))

    # --- Absolute spatial terms (existing logic, unchanged) ---
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
        return int(np.argmax(y_centers))
    elif spatial_term == "farthest":
        return int(np.argmin(y_centers))
    elif spatial_term == "between" and anchor_boxes is not None and anchor2_boxes is not None:
        anchor1 = anchor_boxes.numpy() if torch.is_tensor(anchor_boxes) else np.array(anchor_boxes)
        anchor2 = anchor2_boxes.numpy() if torch.is_tensor(anchor2_boxes) else np.array(anchor2_boxes)
        if len(anchor1) > 0 and len(anchor2) > 0:
            mid_x = ((anchor1[0, 0] + anchor1[0, 2]) / 2 + (anchor2[0, 0] + anchor2[0, 2]) / 2) / 2
            mid_y = ((anchor1[0, 1] + anchor1[0, 3]) / 2 + (anchor2[0, 1] + anchor2[0, 3]) / 2) / 2
            distances = np.sqrt((x_centers - mid_x) ** 2 + (y_centers - mid_y) ** 2)
            return int(np.argmin(distances))
        return 0
    elif spatial_term == "ahead":
        return int(np.argmax(y_centers))
    elif spatial_term in ("left", "right") and ordinal is not None and ordinal > 0:
        if spatial_term == "left":
            sorted_indices = np.argsort(x_centers)
        else:
            sorted_indices = np.argsort(x_centers)[::-1]
        idx = min(ordinal - 1, len(sorted_indices) - 1)
        return int(sorted_indices[idx])
    else:
        return 0
