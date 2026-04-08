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
from typing import Dict, List, Tuple, Optional


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

        print(f"[*] Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        print(f"   [OK] CLIP loaded on {device}")

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

    @torch.no_grad()
    def compare_prompts(self, image_crop: Image.Image, text_prompts: List[str]) -> Dict[str, float]:
        """
        Compare one crop against multiple prompt variants in a single CLIP pass.

        Args:
            image_crop: PIL Image of the cropped detection region
            text_prompts: prompt variants to score

        Returns:
            dict mapping prompt -> cosine similarity
        """
        prompts = [prompt for prompt in text_prompts if prompt and prompt.strip()]
        if not prompts:
            return {}

        image_input = self.preprocess(image_crop).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_input = clip.tokenize(prompts).to(self.device)
        text_features = self.model.encode_text(text_input)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarities = (image_features @ text_features.T).squeeze(0).tolist()
        if isinstance(similarities, float):
            similarities = [similarities]

        return {prompt: float(score) for prompt, score in zip(prompts, similarities)}

    def compute_discriminative_scores(self, image_crop: Image.Image, query_info: dict) -> Dict[str, float]:
        """
        Score a crop against positive and contrastive prompts derived from the query.

        Returns a compact score dictionary used by the natural-language grounding
        ranker in the text-guided pipeline.
        """
        target_object = (query_info.get("target_object") or query_info.get("object_prompt") or "object").strip()
        attrs = query_info.get("attributes") or {}
        color = attrs.get("color")
        condition = attrs.get("condition")
        descriptors = attrs.get("descriptors") or []
        semantic_plan = query_info.get("semantic_plan") or {}
        query_type = semantic_plan.get("query_type", "object-centric")

        prompts = []
        full_query_prompt = (query_info.get("full_query_prompt") or query_info.get("original") or query_info.get("object_prompt") or target_object).strip()
        detector_prompt = (query_info.get("object_prompt") or target_object).strip()
        object_prompt = target_object
        prompts.extend([full_query_prompt, detector_prompt, object_prompt])

        if color and target_object:
            prompts.append(f"{color} {target_object}")
            contrast_colors = [c for c in ["white", "black", "red", "blue", "yellow", "silver", "gray"] if c != color]
            prompts.extend(f"{c} {target_object}" for c in contrast_colors[:2])

        if condition and target_object:
            prompts.append(f"{condition} {target_object}")
            prompts.append(f"{target_object} that is {condition}")
            negative_condition = "normal" if condition != "normal" else "damaged"
            prompts.append(f"{negative_condition} {target_object}")

        for descriptor in descriptors[:2]:
            if target_object:
                prompts.append(f"{descriptor} {target_object}")

        scores = self.compare_prompts(image_crop, prompts)

        full_query_score = scores.get(full_query_prompt, 0.0)
        detector_score = scores.get(detector_prompt, full_query_score)
        object_score = scores.get(object_prompt, detector_score)
        color_score = scores.get(f"{color} {target_object}", detector_score) if color and target_object else detector_score
        condition_prompt_score = scores.get(f"{condition} {target_object}", detector_score) if condition and target_object else detector_score
        condition_phrase_score = scores.get(f"{target_object} that is {condition}", condition_prompt_score) if condition and target_object else condition_prompt_score
        condition_score = max(condition_prompt_score, condition_phrase_score)

        color_contrast = 0.0
        if color and target_object:
            competing = [
                score for prompt, score in scores.items()
                if prompt.endswith(target_object) and prompt != f"{color} {target_object}" and prompt != detector_prompt
            ]
            if competing:
                color_contrast = color_score - max(competing)

        condition_contrast = 0.0
        if condition and target_object:
            alt_prompt = f"{'normal' if condition != 'normal' else 'damaged'} {target_object}"
            condition_contrast = condition_score - scores.get(alt_prompt, 0.0)

        if query_type == "condition-centric" and condition:
            full_query_score = max(full_query_score, condition_score)

        attribute_score = max(color_score, condition_score, detector_score)

        return {
            "full_query_score": float(full_query_score),
            "detector_prompt_score": float(detector_score),
            "object_score": float(object_score),
            "attribute_score": float(attribute_score),
            "color_score": float(color_score),
            "condition_score": float(condition_score),
            "color_contrast": float(color_contrast),
            "condition_contrast": float(condition_contrast),
        }

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
        
        # Track ALL scores so we can fallback to the best one if all are rejected
        all_scores_info = []

        for i in range(len(boxes)):
            crop = self.crop_detection(image, boxes[i])
            if crop is None:
                continue

            # Compute CLIP similarity
            similarity = self.compute_similarity(crop, text_prompt)
            all_scores_info.append((i, similarity))

            if similarity >= self.similarity_threshold:
                verified_boxes.append(boxes[i])
                verified_phrases.append(phrases[i])
                clip_scores.append(similarity)
                if scores is not None:
                    verified_det_scores.append(scores[i])

        # FALLBACK: If ALL boxes were rejected by CLIP, keep the best-scoring one
        # This prevents total detection failure for valid but low-confidence matches
        if len(verified_boxes) == 0 and len(all_scores_info) > 0:
            best_idx, best_score = max(all_scores_info, key=lambda x: x[1])
            print(f"   [WARN] CLIP rejected all {len(all_scores_info)} detections "
                  f"(best={best_score:.3f}, threshold={self.similarity_threshold:.2f}). "
                  f"Keeping best detection as fallback.")
            verified_boxes.append(boxes[best_idx])
            verified_phrases.append(phrases[best_idx])
            clip_scores.append(best_score)
            if scores is not None:
                verified_det_scores.append(scores[best_idx])

        if len(verified_boxes) == 0:
            return (torch.zeros(0, 4), [], [], 
                    torch.zeros(0) if scores is not None else None)

        filtered_boxes = torch.stack(verified_boxes)
        filtered_det_scores = (torch.stack(verified_det_scores) 
                               if verified_det_scores else None)

        n_filtered = len(boxes) - len(verified_boxes)
        if n_filtered > 0:
            print(f"   [*] CLIP filtered {n_filtered}/{len(boxes)} detections "
                  f"(threshold={self.similarity_threshold:.2f})")

        return filtered_boxes, verified_phrases, clip_scores, filtered_det_scores

    @torch.no_grad()
    def generate_heatmap(self, image: np.ndarray, text_prompt: str) -> np.ndarray:
        """
        Generates a dense semantic similarity heatmap across the entire image
        by extracting spatial tokens from the Vision Transformer before pooling.
        
        Args:
            image: numpy array (H, W, 3) RGB image
            text_prompt: text description to map
            
        Returns:
            numpy array (H, W) heatmap normalized to [0, 1]
        """
        import cv2
        
        # 1. Prepare image
        pil_img = Image.fromarray(image)
        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        
        # 2. Encode text normally
        text_input = clip.tokenize([text_prompt]).to(self.device)
        text_features = self.model.encode_text(text_input)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 3. Extract dense image features (bypass final pooling)
        # CLIP ViT-B/32 processes images in 32x32 patches. 224/32 = 7, so it outputs 7x7 spatial tokens + 1 CLS token
        visual_model = self.model.visual
        
        x = image_input.type(visual_model.conv1.weight.dtype)
        x = visual_model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        # Add class token and positional embeddings
        x = torch.cat([visual_model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + visual_model.positional_embedding.to(x.dtype)
        x = visual_model.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = visual_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # We only want the spatial tokens (ignore the [CLS] token at index 0)
        spatial_tokens = x[:, 1:, :] 
        spatial_tokens = visual_model.ln_post(spatial_tokens)
        
        # Project tokens to the joint multimodal space
        dense_image_features = spatial_tokens @ visual_model.proj
        
        # Normalize features
        dense_image_features = dense_image_features / dense_image_features.norm(dim=-1, keepdim=True)
        
        # 4. Compute similarity per spatial patch
        # dense_image_features shape: [1, 49, 512]
        # text_features shape: [1, 512]
        similarity_map = (dense_image_features[0] @ text_features[0]).cpu().numpy()
        
        # 5. Reshape and Normalize
        # The 49 tokens represent a 7x7 grid (for 224x224 input)
        grid_size = int(np.sqrt(similarity_map.shape[0]))
        similarity_map = similarity_map.reshape(grid_size, grid_size)
        
        # Min-max scale the map to [0, 1] for visualization
        similarity_map = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min() + 1e-8)
        
        # Resize to match original image dimensions
        H, W = image.shape[:2]
        heatmap = cv2.resize(similarity_map.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        
        return heatmap
