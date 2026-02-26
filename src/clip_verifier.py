#!/usr/bin/env python3
"""
CLIP Semantic Verifier

Verifies GroundingDINO detections using CLIP similarity scoring.
Filters out false positive detections by checking if cropped regions
actually match the text prompt semantically.

Used between GroundingDINO detection and SAM segmentation in Stage 2.
"""

import torch
import clip
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional


class CLIPVerifier:
    """
    CLIP-based semantic verification for OOD detection.

    Crops each detected bounding box from the image, encodes it with
    CLIP's vision encoder, and compares against the text prompt using
    cosine similarity. Only detections above the threshold are kept.
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda",
                 similarity_threshold: float = 0.20):
        """
        Args:
            model_name: CLIP model variant (ViT-B/32 is lightweight ~1GB)
            device: cuda or cpu
            similarity_threshold: minimum cosine similarity to keep a detection
        """
        self.device = device
        self.similarity_threshold = similarity_threshold

        print(f"🔍 Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        print(f"   ✅ CLIP loaded on {device}")

    @torch.no_grad()
    def compute_similarity(self, image_crop: Image.Image, text_prompt: str) -> float:
        """
        Compute CLIP cosine similarity between an image crop and text prompt.

        Args:
            image_crop: PIL Image of the cropped detection region
            text_prompt: text description to match against

        Returns:
            float: cosine similarity score (0.0 to 1.0)
        """
        # Encode image
        image_input = self.preprocess(image_crop).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Encode text
        text_input = clip.tokenize([text_prompt]).to(self.device)
        text_features = self.model.encode_text(text_input)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarity = (image_features @ text_features.T).item()
        return similarity

    def crop_detection(self, image: np.ndarray, box: torch.Tensor,
                       padding: int = 10) -> Optional[Image.Image]:
        """
        Crop a detection region from the image with optional padding.

        Args:
            image: numpy array (H, W, 3) RGB image
            box: tensor [x1, y1, x2, y2] in pixel coordinates
            padding: extra pixels around the box

        Returns:
            PIL Image of cropped region, or None if invalid
        """
        H, W = image.shape[:2]

        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()

        x1 = max(0, int(box[0]) - padding)
        y1 = max(0, int(box[1]) - padding)
        x2 = min(W, int(box[2]) + padding)
        y2 = min(H, int(box[3]) + padding)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]
        return Image.fromarray(crop)

    def verify_detections(
        self,
        image: np.ndarray,
        boxes: torch.Tensor,
        phrases: List[str],
        text_prompt: str,
        scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[str], List[float], Optional[torch.Tensor]]:
        """
        Verify detected boxes using CLIP semantic similarity.

        Args:
            image: numpy array (H, W, 3) RGB image
            boxes: tensor (N, 4) bounding boxes [x1, y1, x2, y2]
            phrases: list of detected phrase labels
            text_prompt: the OOD text prompt to verify against
            scores: optional detection confidence scores

        Returns:
            Tuple of (filtered_boxes, filtered_phrases, clip_scores, filtered_det_scores)
        """
        if len(boxes) == 0:
            return boxes, phrases, [], scores

        verified_boxes = []
        verified_phrases = []
        clip_scores = []
        verified_det_scores = []

        for i in range(len(boxes)):
            crop = self.crop_detection(image, boxes[i])
            if crop is None:
                continue

            # Compute CLIP similarity
            similarity = self.compute_similarity(crop, text_prompt)

            if similarity >= self.similarity_threshold:
                verified_boxes.append(boxes[i])
                verified_phrases.append(phrases[i])
                clip_scores.append(similarity)
                if scores is not None:
                    verified_det_scores.append(scores[i])

        if len(verified_boxes) == 0:
            return (torch.zeros(0, 4), [], [], 
                    torch.zeros(0) if scores is not None else None)

        filtered_boxes = torch.stack(verified_boxes)
        filtered_det_scores = (torch.stack(verified_det_scores) 
                               if verified_det_scores else None)

        n_filtered = len(boxes) - len(verified_boxes)
        if n_filtered > 0:
            print(f"   🔍 CLIP filtered {n_filtered}/{len(boxes)} detections "
                  f"(threshold={self.similarity_threshold:.2f})")

        return filtered_boxes, verified_phrases, clip_scores, filtered_det_scores
