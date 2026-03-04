"""
Query Parser and Spatial Filter
Parses user text prompts into structured components and filters
bounding boxes by spatial terms.
"""

import math
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

    # Extract color/attribute
    attribute = None
    for color in COLOR_TERMS:
        if color in object_desc:
            attribute = color
            break

    # No spatial term = find all
    detect_all = spatial is None

    # Build the GroundingDINO prompt
    gdino_prompt = object_desc.strip()
    if not gdino_prompt:
        gdino_prompt = user_prompt.strip()

    result = {
        "original": user_prompt,
        "object_prompt": gdino_prompt,
        "attribute": attribute,
        "spatial": spatial,
        "detect_all": detect_all,
        "anchor": anchor,  # NEW: reference object for relational queries
    }

    anchor_info = f", anchor='{anchor}'" if anchor else ""
    print(f"[i] Query parsed: object='{gdino_prompt}', attribute={attribute}, spatial={spatial}{anchor_info}, detect_all={detect_all}")
    return result


def spatial_filter(boxes_xyxy, spatial_term, image_shape=None, anchor_boxes=None):
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
    else:
        return 0
