#!/usr/bin/env python3
"""
Model Loader — Central model loading for MAVR-OOD.
Both app.py (Gradio) and text_guided_detector.py import from here.
No Gradio dependency.

Usage:
    from src.model_loader import load_gdino_model, load_sam_predictor, load_clip_verifier
"""

import os
import sys
import torch

# ============================================================
# CRITICAL: Monkey-patch transformers BEFORE importing GroundingDINO
# transformers 5.0 changed get_extended_attention_mask(mask, shape, device)
# to get_extended_attention_mask(mask, shape, dtype). GroundingDINO passes
# device, causing TypeError. This makes it work with both.
# ============================================================
import transformers
_orig_fn = getattr(transformers.PreTrainedModel, 'get_extended_attention_mask', None)
if _orig_fn is not None:
    def _safe_get_extended_attention_mask(self, attention_mask, input_shape, device_or_dtype=None):
        if attention_mask.dim() == 3:
            extended = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong attention_mask shape: {attention_mask.shape}")
        extended = extended.to(dtype=torch.float32)
        extended = (1.0 - extended) * torch.finfo(torch.float32).min
        return extended
    transformers.PreTrainedModel.get_extended_attention_mask = _safe_get_extended_attention_mask

# Add paths for GroundingDINO and SAM submodules
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_gdino_path = os.path.join(_project_root, "GroundingDINO")
_sam_path = os.path.join(_project_root, "segment_anything")

if _gdino_path not in sys.path:
    sys.path.insert(0, _gdino_path)
if _sam_path not in sys.path:
    sys.path.insert(0, _sam_path)


# =====================
# Default paths and device
# =====================
DEFAULT_GDINO_CONFIG = os.path.join(_project_root, "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
DEFAULT_GDINO_CKPT = os.path.join(_project_root, "weights", "groundingdino_swint_ogc.pth")
DEFAULT_SAM_CKPT = os.path.join(_project_root, "weights", "sam_vit_h_4b8939.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================
# Singleton model holders
# =====================
_gdino_model = None
_sam_predictor = None
_clip_verifier = None


def load_gdino_model(config_path=None, checkpoint_path=None, device=None):
    """
    Load GroundingDINO model (singleton — loads once, reuses after).

    Args:
        config_path: path to GroundingDINO config .py (default: project default)
        checkpoint_path: path to .pth checkpoint (default: project default)
        device: 'cuda' or 'cpu' (default: auto-detect)

    Returns:
        loaded GroundingDINO model
    """
    global _gdino_model
    if _gdino_model is not None:
        return _gdino_model

    config_path = config_path or DEFAULT_GDINO_CONFIG
    checkpoint_path = checkpoint_path or DEFAULT_GDINO_CKPT
    device = device or DEVICE

    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict

    print("[i] Loading GroundingDINO...")
    args = SLConfig.fromfile(config_path)
    args.device = device
    args.bert_base_uncased_path = None
    _gdino_model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _gdino_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _gdino_model = _gdino_model.to(device)
    _gdino_model.eval()
    print("[OK] GroundingDINO loaded")
    return _gdino_model


def load_sam_predictor(checkpoint_path=None, device=None):
    """
    Load SAM predictor (singleton — loads once, reuses after).

    Args:
        checkpoint_path: path to SAM .pth checkpoint (default: project default)
        device: 'cuda' or 'cpu' (default: auto-detect)

    Returns:
        SamPredictor instance
    """
    global _sam_predictor
    if _sam_predictor is not None:
        return _sam_predictor

    checkpoint_path = checkpoint_path or DEFAULT_SAM_CKPT
    device = device or DEVICE

    from segment_anything import sam_model_registry, SamPredictor

    print("[i] Loading SAM...")
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam = sam.to(device)
    _sam_predictor = SamPredictor(sam)
    print("[OK] SAM loaded")
    return _sam_predictor


def load_clip_verifier(device=None):
    """
    Load CLIP verifier (singleton — loads once, reuses after).

    Args:
        device: 'cuda' or 'cpu' (default: auto-detect)

    Returns:
        CLIPVerifier instance
    """
    global _clip_verifier
    if _clip_verifier is not None:
        return _clip_verifier

    device = device or DEVICE

    from src.clip_verifier import CLIPVerifier

    print("[i] Loading CLIP...")
    _clip_verifier = CLIPVerifier(device=device)
    print("[OK] CLIP loaded")
    return _clip_verifier


def get_device():
    """Return the detected device string."""
    return DEVICE
