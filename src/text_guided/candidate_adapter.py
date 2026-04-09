"""
Helpers for adapting different grounding backends into the text-guided
pipeline's shared candidate representation.
"""

from __future__ import annotations

from typing import Iterable, List

import torch


def xyxy_to_cxcywh_normalized(boxes_xyxy: Iterable[Iterable[float]], image_size: tuple[int, int]) -> torch.Tensor:
    """Convert pixel-space xyxy boxes to normalized cxcywh tensors."""
    height, width = image_size
    boxes_xyxy = list(boxes_xyxy or [])
    if not boxes_xyxy:
        return torch.zeros((0, 4), dtype=torch.float32)

    boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
    cxcywh = torch.zeros_like(boxes_tensor)
    cxcywh[:, 0] = ((boxes_tensor[:, 0] + boxes_tensor[:, 2]) / 2.0) / max(width, 1)
    cxcywh[:, 1] = ((boxes_tensor[:, 1] + boxes_tensor[:, 3]) / 2.0) / max(height, 1)
    cxcywh[:, 2] = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) / max(width, 1)
    cxcywh[:, 3] = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) / max(height, 1)
    return cxcywh


def cxcywh_normalized_to_xyxy(boxes_cxcywh: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """Convert normalized cxcywh tensors to pixel-space xyxy tensors."""
    height, width = image_size
    if boxes_cxcywh is None or len(boxes_cxcywh) == 0:
        return torch.zeros((0, 4), dtype=torch.float32)

    boxes_xyxy = torch.zeros((len(boxes_cxcywh), 4), dtype=torch.float32)
    scaled = boxes_cxcywh.clone().to(dtype=torch.float32)
    scaled[:, 0] *= width
    scaled[:, 1] *= height
    scaled[:, 2] *= width
    scaled[:, 3] *= height
    boxes_xyxy[:, 0] = scaled[:, 0] - scaled[:, 2] / 2.0
    boxes_xyxy[:, 1] = scaled[:, 1] - scaled[:, 3] / 2.0
    boxes_xyxy[:, 2] = scaled[:, 0] + scaled[:, 2] / 2.0
    boxes_xyxy[:, 3] = scaled[:, 1] + scaled[:, 3] / 2.0
    return boxes_xyxy


def normalize_candidate_output(
    *,
    backend_name: str,
    boxes_xyxy: Iterable[Iterable[float]],
    image_size: tuple[int, int],
    labels: Iterable[str] | None = None,
    scores: Iterable[float] | None = None,
    raw_response: object | None = None,
) -> dict:
    """
    Return the unified candidate payload expected by the text-guided pipeline.
    """
    boxes_xyxy_list: List[List[float]] = [list(map(float, box)) for box in (boxes_xyxy or [])]
    labels_list = [str(label) for label in (labels or [])]
    scores_list = [float(score) for score in (scores or [])]

    while len(labels_list) < len(boxes_xyxy_list):
        idx = len(labels_list) + 1
        labels_list.append(f"{backend_name}-candidate-{idx}")

    while len(scores_list) < len(boxes_xyxy_list):
        scores_list.append(0.0)

    boxes_cxcywh = xyxy_to_cxcywh_normalized(boxes_xyxy_list, image_size)

    return {
        "backend": backend_name,
        "boxes_xyxy": torch.tensor(boxes_xyxy_list, dtype=torch.float32) if boxes_xyxy_list else torch.zeros((0, 4), dtype=torch.float32),
        "boxes_cxcywh": boxes_cxcywh,
        "labels": labels_list,
        "scores": scores_list,
        "raw_response": raw_response,
    }
