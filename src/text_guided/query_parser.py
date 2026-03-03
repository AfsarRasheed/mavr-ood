"""
Query Parser and Spatial Filter
Parses user text prompts into structured components and filters
bounding boxes by spatial terms.
"""

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
    }

    print(f"[i] Query parsed: object='{gdino_prompt}', attribute={attribute}, spatial={spatial}, detect_all={detect_all}")
    return result


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
        return int(np.argmax(y_centers))
    elif spatial_term == "farthest":
        return int(np.argmin(y_centers))
    else:
        return 0
