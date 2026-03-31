"""
MAVR-OOD Web Application — FastAPI Backend
Modern web interface for the Multi-Agent Vision-Language System.
"""

import os
import sys
import gc
import io
import time
import json
import base64
import tempfile
import numpy as np
import torch
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Add paths
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "GroundingDINO"))
sys.path.insert(0, str(ROOT / "segment_anything"))

# Monkey-patch for transformers compatibility
import transformers
_orig_fn = getattr(transformers.PreTrainedModel, 'get_extended_attention_mask', None)
if _orig_fn is not None:
    def _safe_get_extended_attention_mask(self, attention_mask, input_shape, device_or_dtype=None):
        if attention_mask.dim() == 3:
            extended = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape: {attention_mask.shape}")
        extended = extended.to(dtype=torch.float32)
        extended = (1.0 - extended) * torch.finfo(torch.float32).min
        return extended
    transformers.PreTrainedModel.get_extended_attention_mask = _safe_get_extended_attention_mask

# ── FastAPI App ──────────────────────────────────────────────
app = FastAPI(title="MAVR-OOD", description="Multi-Agent Vision-Language System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler — ALWAYS return JSON, never HTML
from starlette.requests import Request
from starlette.responses import JSONResponse as StarletteJSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    traceback.print_exc()
    return StarletteJSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc)},
    )

# Serve static files
static_dir = ROOT / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ── Model Storage ────────────────────────────────────────────
models = {}


def load_all_models():
    """Load detection models (not LLaVA — it loads on demand)."""
    from src.model_loader import load_gdino_model, load_sam_predictor, load_clip_verifier
    print("[i] Loading GroundingDINO...")
    models['gdino'] = load_gdino_model()
    print("[i] Loading SAM...")
    models['sam'] = load_sam_predictor()
    print("[i] Loading CLIP verifier...")
    models['clip'] = load_clip_verifier()
    print("[OK] All detection models loaded")


@app.on_event("startup")
async def startup():
    load_all_models()


# ── Helper: numpy image → base64 JPEG ───────────────────────
def img_to_b64(img_np):
    if img_np is None:
        return None
    pil = Image.fromarray(img_np.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ── Routes ───────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home():
    html_path = static_dir / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>MAVR-OOD</h1><p>static/index.html not found</p>")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A"
    return {
        "status": "ok",
        "gpu": gpu,
        "vram": vram,
        "models_loaded": list(models.keys()),
    }


@app.post("/api/detect")
async def detect(
    image: UploadFile = File(...),
    query: str = Form(...),
):
    """Run the full MAVR pipeline."""
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    # Read & decode image
    img_bytes = await image.read()
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img_pil)

    # Save temp file for LLaVA
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img_pil.save(tmp.name)
    tmp_path = tmp.name

    try:
        from src.text_guided import run_text_guided_pipeline

        t0 = time.time()
        results = run_text_guided_pipeline(
            image_np=img_np,
            user_prompt=query,
            image_path=tmp_path,
            gdino_model=models['gdino'],
            sam_predictor=models['sam'],
            clip_verifier=models['clip'],
            box_threshold=0.35,
            clip_threshold=0.25,
        )
        elapsed = round(time.time() - t0, 1)

        # Encode step images
        step_images_b64 = {}
        for key, img in results.get('step_images', {}).items():
            step_images_b64[key] = img_to_b64(img)

        # Encode final overlay
        final_overlay = None
        final_masks = results.get('final_masks')
        if final_masks is not None:
            H, W = img_np.shape[:2]
            overlay = img_np.copy().astype(np.float32)
            for i in range(len(final_masks)):
                m = final_masks[i]
                if hasattr(m, 'cpu'):
                    m = m.cpu().numpy()
                if m.ndim == 3:
                    m = m[0]
                mask = (m > 0)
                overlay[mask] = overlay[mask] * 0.4 + np.array([100, 180, 255]) * 0.6
            final_overlay = img_to_b64(overlay.astype(np.uint8))

        # Parse query info
        parsed = results.get('parsed_query', {})

        return JSONResponse({
            "success": True,
            "time": elapsed,
            "query": query,
            "parsed": {
                "object_prompt": parsed.get('object_prompt', query),
                "spatial": parsed.get('spatial'),
                "attribute": parsed.get('attribute'),
                "parser_mode": parsed.get('parser_mode', 'rule-based'),
                "anchor": parsed.get('anchor'),
            },
            "step_images": step_images_b64,
            "final_overlay": final_overlay,
            "original_image": img_to_b64(img_np),
            "reasoning": results.get('reasoning', 'No reasoning available.'),
            "summary": results.get('summary', ''),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False,
            "error": str(e),
        }, status_code=500)

    finally:
        os.unlink(tmp_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── OOD Detection Endpoint ───────────────────────────────────
@app.post("/api/ood_detect")
async def ood_detect(
    image: UploadFile = File(...),
    gt_mask: UploadFile = File(None),
):
    """Run the full OOD (out-of-distribution) detection pipeline."""
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    img_bytes = await image.read()
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img_pil)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img_pil.save(tmp.name)
    tmp_path = tmp.name

    try:
        from app import (
            run_agents_on_image, extract_prompts,
            preprocess_image, get_grounding_output,
            run_sam_segmentation,
            create_detection_visualization, create_mask_visualization,
            create_binary_mask_visualization,
        )

        # Stage 1: Run 5 LLaVA agents
        agent_results = run_agents_on_image(tmp_path)
        prompt_v1, prompt_v2 = extract_prompts(agent_results)

        # Free LLaVA from GPU
        try:
            import src.agents.vlm_backend as vlm_mod
            if getattr(vlm_mod, '_model', None) is not None:
                del vlm_mod._model; vlm_mod._model = None
            if getattr(vlm_mod, '_processor', None) is not None:
                del vlm_mod._processor; vlm_mod._processor = None
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            gc.collect(); torch.cuda.empty_cache()

        # Stage 2: GroundingDINO detection
        gdino = models['gdino']
        image_tensor = preprocess_image(img_pil)
        boxes, labels, scores = get_grounding_output(
            gdino, image_tensor, prompt_v1,
            box_threshold=0.35, text_threshold=0.25
        )

        if len(boxes) == 0 and prompt_v2 != prompt_v1:
            boxes, labels, scores = get_grounding_output(
                gdino, image_tensor, prompt_v2,
                box_threshold=0.35, text_threshold=0.25
            )

        # CLIP verification
        if len(boxes) > 0 and models.get('clip'):
            try:
                H, W = img_np.shape[:2]
                clip_boxes = boxes.clone()
                clip_boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * W
                clip_boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * H
                clip_boxes[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * W
                clip_boxes[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * H
                models['clip'].similarity_threshold = 0.25
                filtered_boxes, filtered_phrases, _, _ = models['clip'].verify_detections(
                    img_np, clip_boxes, labels, prompt_v1
                )
                if len(filtered_boxes) > 0:
                    boxes_back = torch.zeros(len(filtered_boxes), 4)
                    boxes_back[:, 0] = ((filtered_boxes[:, 0] + filtered_boxes[:, 2]) / 2) / W
                    boxes_back[:, 1] = ((filtered_boxes[:, 1] + filtered_boxes[:, 3]) / 2) / H
                    boxes_back[:, 2] = (filtered_boxes[:, 2] - filtered_boxes[:, 0]) / W
                    boxes_back[:, 3] = (filtered_boxes[:, 3] - filtered_boxes[:, 1]) / H
                    boxes = boxes_back
                    labels = filtered_phrases
            except Exception:
                pass

        # SAM segmentation
        masks = None
        boxes_xyxy = None
        if len(boxes) > 0:
            predictor = models['sam']
            masks, boxes_xyxy = run_sam_segmentation(predictor, img_np, boxes)

        # Generate visualizations
        result_images = {}
        if masks is not None and boxes_xyxy is not None:
            det_img = create_detection_visualization(img_np, boxes_xyxy, labels)
            mask_img = create_mask_visualization(img_np, masks)
            binary_img = create_binary_mask_visualization(img_np, masks)
            result_images['detection'] = img_to_b64(det_img)
            result_images['masks'] = img_to_b64(mask_img)
            result_images['binary_mask'] = img_to_b64(binary_img)

        # Compute metrics if ground truth provided
        metrics = None
        if gt_mask is not None and masks is not None:
            import cv2
            gt_bytes = await gt_mask.read()
            gt_pil = Image.open(io.BytesIO(gt_bytes)).convert("L")
            gt_np = np.array(gt_pil)
            gt_binary = (gt_np > 0).astype(np.float32)

            pred_binary = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.float32)
            for m in masks:
                m_np = m.cpu().numpy() if isinstance(m, torch.Tensor) else m
                if m_np.ndim == 3:
                    m_np = m_np.squeeze(0)
                if m_np.shape != pred_binary.shape:
                    m_np = cv2.resize(m_np.astype(np.float32), (pred_binary.shape[1], pred_binary.shape[0]))
                pred_binary = np.maximum(pred_binary, m_np)
            pred_binary = (pred_binary > 0.5).astype(np.float32)

            if gt_binary.shape != pred_binary.shape:
                gt_binary = cv2.resize(gt_binary, (pred_binary.shape[1], pred_binary.shape[0]))

            intersection = (pred_binary * gt_binary).sum()
            union = ((pred_binary + gt_binary) > 0).sum()
            iou = float(intersection / (union + 1e-8))
            tp = float(intersection)
            fp = float((pred_binary * (1 - gt_binary)).sum())
            fn = float(((1 - pred_binary) * gt_binary).sum())
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            metrics = {
                "iou": round(iou, 4),
                "f1": round(f1, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
            }

        return JSONResponse({
            "success": True,
            "detections": len(boxes) if boxes is not None else 0,
            "prompt_v1": prompt_v1,
            "prompt_v2": prompt_v2,
            "images": result_images,
            "metrics": metrics,
            "agents": {k: v for k, v in agent_results.items()},
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    finally:
        os.unlink(tmp_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
