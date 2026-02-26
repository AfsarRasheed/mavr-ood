#!/usr/bin/env python3
"""
VLM Backend: LLaVA-7B via Hugging Face Transformers

Replaces gemini_llm.py. Provides a unified interface for all agents
to run vision-language inference locally using LLaVA-1.5-7B.

Supports:
  - Image + text input (Agents 1-4)
  - Text-only input (Agent 5)
  - 4-bit quantization (Linux/Colab — fastest, ~4GB VRAM)
  - FP16 with automatic GPU/CPU split (Windows fallback)
"""

import gc
import os
import torch
from dotenv import load_dotenv
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

load_dotenv()

# =========================
# Configuration
# =========================
MODEL_NAME = os.getenv("VLM_MODEL_NAME", "llava-hf/llava-1.5-7b-hf")
TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.2"))
MAX_NEW_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "4096"))
DEVICE = os.getenv("VLM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Auto-detect 4-bit support
# =========================
def _can_use_4bit():
    """Check if bitsandbytes 4-bit quantization is available."""
    try:
        import bitsandbytes as bnb
        # bitsandbytes only works on Linux (Colab)
        import platform
        if platform.system() == "Linux":
            print("   ✅ bitsandbytes available — using 4-bit quantization")
            return True
        else:
            print("   ⚠️ bitsandbytes not supported on Windows — using FP16")
            return False
    except ImportError:
        print("   ⚠️ bitsandbytes not installed — using FP16")
        return False

# =========================
# Singleton Model Loader
# =========================
_model = None
_processor = None


def _load_model():
    """Load LLaVA model and processor (singleton — loaded once, reused by all agents)."""
    global _model, _processor

    if _model is not None:
        return _model, _processor

    print(f"🔄 Loading LLaVA model: {MODEL_NAME}")
    print(f"   Device: {DEVICE}")

    use_4bit = _can_use_4bit()

    if use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print("   📦 Loading with 4-bit NF4 quantization (~4GB VRAM)")
        _model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        print("   📦 Loading in FP16 with auto device_map (GPU + CPU split)")
        _model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    # Load processor
    _processor = AutoProcessor.from_pretrained(MODEL_NAME)

    print(f"   ✅ Model loaded successfully")
    if hasattr(_model, 'hf_device_map'):
        # Show summary instead of full map
        devices = set(_model.hf_device_map.values())
        print(f"   📊 Model distributed across: {devices}")
    return _model, _processor


def run_vlm(messages, image_path=None):
    """
    Run LLaVA inference.

    Args:
        messages: list of dicts [{"role": "system/user", "content": "..."}]
        image_path: optional path to image file (for Agents 1-4)

    Returns:
        str: model output text
    """
    model, processor = _load_model()

    # Build the prompt from messages
    system_prompt = ""
    user_prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        elif msg["role"] == "user":
            user_prompt = msg["content"]

    # Construct conversation for LLaVA
    if image_path and os.path.exists(image_path):
        # Vision + Language mode (Agents 1-4)
        image = Image.open(image_path).convert("RGB")

        # LLaVA prompt format
        if system_prompt:
            full_prompt = f"USER: <image>\n{system_prompt}\n\n{user_prompt}\nASSISTANT:"
        else:
            full_prompt = f"USER: <image>\n{user_prompt}\nASSISTANT:"

        inputs = processor(
            text=full_prompt,
            images=image,
            return_tensors="pt",
        )
    else:
        # Text-only mode (Agent 5)
        if system_prompt:
            full_prompt = f"USER: {system_prompt}\n\n{user_prompt}\nASSISTANT:"
        else:
            full_prompt = f"USER: {user_prompt}\nASSISTANT:"

        inputs = processor(
            text=full_prompt,
            return_tensors="pt",
        )

    # Move inputs to the same device as the model's first parameter
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=TEMPERATURE > 0,
            top_p=0.9,
        )

    # Decode only the generated tokens (skip the input prompt)
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True)

    # Clean up GPU memory after each inference
    del inputs, output_ids, generated_ids
    if image_path:
        del image
    gc.collect()
    torch.cuda.empty_cache()

    return response.strip()
