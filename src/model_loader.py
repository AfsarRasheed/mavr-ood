#!/usr/bin/env python3
"""
Model Loader — Central model loading for MAVR-OOD.
Both app.py (Gradio) and text_guided_detector.py import from here.
No Gradio dependency.

Usage:
    from src.model_loader import (
        load_gdino_model,
        load_sam_predictor,
        load_clip_verifier,
        load_florence2_model,
    )
"""

import os
import sys
import json
import traceback
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
DEFAULT_FLORENCE2_MODEL_ID = os.getenv("FLORENCE2_MODEL_ID", "microsoft/Florence-2-large")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================
# Singleton model holders
# =====================
_gdino_model = None
_sam_predictor = None
_clip_verifier = None
_florence2_bundle = None


def _ensure_florence2_config_fields(config):
    """
    Fill missing Florence-2 config attributes before model construction.
    """
    if config is None:
        return config

    fallback_bos = getattr(config, "bos_token_id", 0)
    fallback_eos = getattr(config, "eos_token_id", 2)
    fallback_pad = getattr(config, "pad_token_id", 1)

    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        text_config_cls = type(text_config)
        if not hasattr(text_config_cls, "forced_bos_token_id"):
            setattr(text_config_cls, "forced_bos_token_id", fallback_bos)
        if not hasattr(text_config_cls, "bos_token_id"):
            setattr(text_config_cls, "bos_token_id", fallback_bos)
        if not hasattr(text_config_cls, "eos_token_id"):
            setattr(text_config_cls, "eos_token_id", fallback_eos)
        if not hasattr(text_config_cls, "pad_token_id"):
            setattr(text_config_cls, "pad_token_id", fallback_pad)
        fallback_bos = getattr(text_config, "bos_token_id", fallback_bos)
        fallback_eos = getattr(text_config, "eos_token_id", fallback_eos)
        fallback_pad = getattr(text_config, "pad_token_id", fallback_pad)

    for target in [obj for obj in (config, text_config) if obj is not None]:
        if getattr(target, "forced_bos_token_id", None) is None:
            setattr(target, "forced_bos_token_id", fallback_bos)
        if getattr(target, "bos_token_id", None) is None:
            setattr(target, "bos_token_id", fallback_bos)
        if getattr(target, "eos_token_id", None) is None:
            setattr(target, "eos_token_id", fallback_eos)
        if getattr(target, "pad_token_id", None) is None:
            setattr(target, "pad_token_id", fallback_pad)

    return config


def _load_florence2_config_with_defaults(model_id):
    """
    Load Florence-2 config with trust_remote_code and ensure all required
    token ID fields are present.

    This runs AFTER _patch_florence2_remote_code has fixed the config file
    on disk, so AutoConfig.from_pretrained will succeed.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return _ensure_florence2_config_fields(config)


def _ensure_florence2_generation_config(model):
    """
    Florence-2 remote code can expect generation/config attributes that may be
    absent in some Colab package combinations. Fill in safe defaults.
    """
    config = getattr(model, "config", None)
    generation_config = getattr(model, "generation_config", None)

    fallback_bos = 0
    fallback_eos = 2
    fallback_pad = 1

    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        fallback_bos = getattr(text_config, "bos_token_id", fallback_bos)
        fallback_eos = getattr(text_config, "eos_token_id", fallback_eos)
        fallback_pad = getattr(text_config, "pad_token_id", fallback_pad)

    fallback_bos = getattr(config, "bos_token_id", fallback_bos)
    fallback_eos = getattr(config, "eos_token_id", fallback_eos)
    fallback_pad = getattr(config, "pad_token_id", fallback_pad)

    targets = [obj for obj in (config, text_config, generation_config) if obj is not None]
    for target in targets:
        if getattr(target, "forced_bos_token_id", None) is None:
            setattr(target, "forced_bos_token_id", fallback_bos)
        if getattr(target, "bos_token_id", None) is None:
            setattr(target, "bos_token_id", fallback_bos)
        if getattr(target, "eos_token_id", None) is None:
            setattr(target, "eos_token_id", fallback_eos)
        if getattr(target, "pad_token_id", None) is None:
            setattr(target, "pad_token_id", fallback_pad)


def _ensure_florence2_tokenizer_padding(processor, model=None):
    """
    Ensure Florence-2 has a usable pad token for processor tokenization and generation.
    """
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return processor

    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if model is not None:
                model.resize_token_embeddings(len(tokenizer))

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            tokenizer.pad_token_id = eos_token_id
            pad_token_id = eos_token_id

    for target in (getattr(model, "config", None), getattr(model, "generation_config", None)):
        if target is not None and getattr(target, "pad_token_id", None) is None and pad_token_id is not None:
            setattr(target, "pad_token_id", pad_token_id)

    return processor


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


def _patch_florence2_remote_code(model_id):
    """
    Download and patch Florence-2's remote code on disk before loading.

    Microsoft's remote code has two known compatibility bugs with recent
    transformers versions:

    1. ``configuration_florence2.py``: accesses ``self.forced_bos_token_id``
       before ``super().__init__()`` sets it → ``AttributeError``
    2. ``processing_florence2.py``: accesses ``tokenizer.additional_special_tokens``
       which doesn't exist on newer ``RobertaTokenizer`` → ``AttributeError``

    This function downloads the remote files (triggering cache), patches them
    on disk, and clears stale module imports.
    """
    import os

    # Step 1: Ensure ALL remote code files are cached on disk.
    # We use huggingface_hub to download the full model repo, which caches
    # all .py files (config, processing, modeling) at once. Individual
    # Auto*.from_pretrained calls are used as fallbacks since they each
    # only download a subset of files.
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(model_id, local_files_only=False)
    except Exception:
        pass

    from transformers import AutoConfig, AutoProcessor
    try:
        AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        pass
    try:
        AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        pass

    # Step 2: Find and patch all Florence-2 remote code files
    cache_base = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    patched_files = []

    # Patches for configuration_florence2.py
    config_repairs = {
        "if self.forced_bos_token_id is None":
            "if getattr(self, 'forced_bos_token_id', None) is None",
        "if self.forced_eos_token_id is None":
            "if getattr(self, 'forced_eos_token_id', None) is None",
    }

    # Patches for processing_florence2.py
    processing_repairs = {
        "tokenizer.additional_special_tokens + ":
            "getattr(tokenizer, 'additional_special_tokens', []) + ",
        "tokenizer.additional_special_tokens +\\":
            "getattr(tokenizer, 'additional_special_tokens', []) +\\",
    }

    # Files to patch with simple string replacement
    simple_patch_map = {
        "configuration_florence2.py": config_repairs,
        "processing_florence2.py": processing_repairs,
    }

    for root, dirs, files in os.walk(cache_base):
        if "florence" not in root.lower():
            continue
        for fname in files:
            filepath = os.path.join(root, fname)

            # --- Simple string replacements for config & processing ---
            if fname in simple_patch_map:
                repairs = simple_patch_map[fname]
                with open(filepath, "r") as fh:
                    content = fh.read()

                changed = False
                for old, new in repairs.items():
                    if old in content:
                        content = content.replace(old, new)
                        changed = True

                if changed:
                    with open(filepath, "w") as fh:
                        fh.write(content)
                    patched_files.append(fname)
                    print(f"[OK] Patched: {filepath}")

            # --- Comment out flash_attn imports in modeling file ---
            elif fname == "modeling_florence2.py":
                with open(filepath, "r") as fh:
                    lines = fh.readlines()

                changed = False
                new_lines = []
                for line in lines:
                    stripped = line.strip()
                    # Comment out any line that imports from flash_attn
                    if (stripped.startswith("from flash_attn") or
                            stripped.startswith("import flash_attn")):
                        new_lines.append(
                            f"# {stripped}  # patched out — eager attention\n")
                        changed = True
                    else:
                        new_lines.append(line)

                if changed:
                    with open(filepath, "w") as fh:
                        fh.writelines(new_lines)
                    patched_files.append(fname)
                    print(f"[OK] Patched flash_attn imports: {filepath}")

    # Step 3: Clear stale modules so Python reloads the patched files
    stale = [k for k in sys.modules if "florence" in k.lower()]
    for k in stale:
        del sys.modules[k]

    if patched_files:
        print(f"[OK] Florence-2 remote code patched: {', '.join(patched_files)}")
    else:
        print("[i] Florence-2 remote code — no patches needed")


def load_florence2_model(model_id=None, device=None):
    """
    Load Florence-2 model + processor as a singleton bundle.

    Handles compatibility issues with Florence-2's remote config code that
    expects certain token ID attributes during __init__ before they're set
    by the parent class.
    """
    global _florence2_bundle
    if _florence2_bundle is not None:
        return _florence2_bundle

    device = device or DEVICE
    model_id = model_id or DEFAULT_FLORENCE2_MODEL_ID

    from transformers import AutoModelForCausalLM, AutoProcessor

    try:
        print(f"[i] Loading Florence-2 ({model_id})...")

        # Step 1: Download remote code and patch the config bug on disk
        _patch_florence2_remote_code(model_id)

        # Step 2: Load processor (now uses the patched config code)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # Step 3: Build config with injected defaults
        config = _load_florence2_config_with_defaults(model_id)

        # Step 4: Load model with patched config
        # attn_implementation="eager" bypasses SDPA check — Florence-2's remote
        # code doesn't define _supports_sdpa which newer transformers expects.
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
        )
        _ensure_florence2_generation_config(model)
        processor = _ensure_florence2_tokenizer_padding(processor, model=model)
        model = model.to(device)
        model.eval()
    except Exception:
        print("[ERR] Florence-2 load traceback:")
        traceback.print_exc()
        raise

    _florence2_bundle = {
        "model_id": model_id,
        "device": device,
        "model": model,
        "processor": processor,
    }
    print("[OK] Florence-2 loaded")
    return _florence2_bundle


def get_device():
    """Return the detected device string."""
    return DEVICE
