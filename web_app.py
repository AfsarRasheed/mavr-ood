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

# ── OOD Helper Functions (self-contained, no gradio dependency) ──

import cv2
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.util.utils import get_phrases_from_posmap

def _preprocess_image(image_pil):
    """Preprocess image for GroundingDINO."""
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(image_pil, None)
    return image_tensor


def _get_grounding_output(model, image_tensor, caption, box_threshold=0.3, text_threshold=0.25):
    """Run GroundingDINO detection."""
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        score = logit.max().item()
        pred_phrases.append(f"{pred_phrase}({score:.2f})")
        scores.append(score)
    return boxes_filt, pred_phrases, scores


def _run_sam_segmentation(predictor, image_np, boxes):
    """Run SAM segmentation on detected boxes."""
    predictor.set_image(image_np)
    H, W = image_np.shape[:2]
    boxes_scaled = boxes.clone()
    boxes_scaled[:, 0] *= W
    boxes_scaled[:, 1] *= H
    boxes_scaled[:, 2] *= W
    boxes_scaled[:, 3] *= H
    boxes_xyxy = torch.zeros_like(boxes_scaled)
    boxes_xyxy[:, 0] = boxes_scaled[:, 0] - boxes_scaled[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_scaled[:, 1] - boxes_scaled[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_scaled[:, 0] + boxes_scaled[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_scaled[:, 1] + boxes_scaled[:, 3] / 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy.to(device), (H, W))
    masks, _, _ = predictor.predict_torch(
        point_coords=None, point_labels=None,
        boxes=transformed_boxes, multimask_output=False,
    )
    return masks.cpu(), boxes_xyxy


def _create_detection_vis(image_np, boxes_xyxy, labels):
    vis = image_np.copy()
    for box, label in zip(boxes_xyxy, labels):
        x1, y1, x2, y2 = box.int().numpy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(vis, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return vis


def _create_mask_vis(image_np, masks):
    vis = image_np.copy().astype(np.float32)
    colors = [[255, 0, 100], [255, 165, 0], [0, 255, 255], [255, 255, 0], [128, 0, 255]]
    for i, mask in enumerate(masks):
        mask_np = mask.squeeze().numpy().astype(bool)
        color = np.array(colors[i % len(colors)], dtype=np.float32)
        vis[mask_np] = vis[mask_np] * 0.4 + color * 0.6
    return vis.astype(np.uint8)


def _create_binary_mask_vis(image_np, masks):
    vis = image_np.copy().astype(np.float32)
    combined = np.zeros(image_np.shape[:2], dtype=bool)
    for mask in masks:
        combined |= mask.squeeze().numpy().astype(bool)
    pink = np.array([255, 105, 180], dtype=np.float32)
    vis[combined] = vis[combined] * 0.35 + pink * 0.65
    return vis.astype(np.uint8)


def _run_agents_on_image(image_path):
    """Run all 5 OOD agents."""
    results = {}
    try:
        from src.agents.agent1 import SceneContextAnalyzer
        results["agent1"] = SceneContextAnalyzer().analyze_image(image_path)
    except Exception as e:
        results["agent1"] = {"error": str(e)}
    try:
        from src.agents.agent2 import SpatialAnomalyDetector
        results["agent2"] = SpatialAnomalyDetector().analyze_image(image_path)
    except Exception as e:
        results["agent2"] = {"error": str(e)}
    try:
        from src.agents.agent3 import SemanticInconsistencyAnalyzer
        results["agent3"] = SemanticInconsistencyAnalyzer().analyze_image(image_path)
    except Exception as e:
        results["agent3"] = {"error": str(e)}
    try:
        from src.agents.agent4 import VisualAppearanceEvaluator
        results["agent4"] = VisualAppearanceEvaluator().analyze_image(image_path)
    except Exception as e:
        results["agent4"] = {"error": str(e)}
    try:
        from src.agents.agent5 import ReasoningSynthesizer
        combined = {
            "agent1_scene_context": results.get("agent1", {}),
            "agent2_spatial_anomaly": results.get("agent2", {}),
            "agent3_semantic_inconsistency": results.get("agent3", {}),
            "agent4_visual_appearance": results.get("agent4", {}),
        }
        results["agent5"] = ReasoningSynthesizer().synthesize_analysis(combined)
    except Exception as e:
        results["agent5"] = {"error": str(e)}
    return results


def _extract_prompts(agent_results):
    a5 = agent_results.get("agent5", {})
    prompts = a5.get("grounded_sam_prompts", {})
    prompt_v1 = prompts.get("prompt_v1", a5.get("detailed_prompt", "unusual object on road"))
    prompt_v2 = prompts.get("prompt_v2", a5.get("simple_prompt", "anomaly"))
    return prompt_v1, prompt_v2


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
        # Stage 1: Run 5 LLaVA agents
        agent_results = _run_agents_on_image(tmp_path)
        prompt_v1, prompt_v2 = _extract_prompts(agent_results)

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
        image_tensor = _preprocess_image(img_pil)
        boxes, labels, scores = _get_grounding_output(
            gdino, image_tensor, prompt_v1,
            box_threshold=0.35, text_threshold=0.25
        )

        if len(boxes) == 0 and prompt_v2 != prompt_v1:
            boxes, labels, scores = _get_grounding_output(
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
            masks, boxes_xyxy = _run_sam_segmentation(models['sam'], img_np, boxes)

        # Generate visualizations
        result_images = {}
        if masks is not None and boxes_xyxy is not None:
            result_images['detection'] = img_to_b64(_create_detection_vis(img_np, boxes_xyxy, labels))
            result_images['masks'] = img_to_b64(_create_mask_vis(img_np, masks))
            result_images['binary_mask'] = img_to_b64(_create_binary_mask_vis(img_np, masks))

        # Compute metrics if ground truth provided
        metrics = None
        if gt_mask is not None and masks is not None:
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

