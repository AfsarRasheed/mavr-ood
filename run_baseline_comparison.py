"""
Baseline Comparison: GroundingDINO-only vs Full MAVR Pipeline
Proves that the multi-agent pipeline improves over a single-model baseline.

Usage (Colab):
    !python run_baseline_comparison.py
"""

import matplotlib
matplotlib.use('Agg')

import os, sys, gc, json, time
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Monkey-patch for GroundingDINO compatibility
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

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

from run_evaluate_vlm import (
    QUERY_MAP, load_ground_truth_mask, compute_metrics, create_predicted_mask
)


def run_baseline_gdino_only(image_np, query, gdino_model, sam_predictor, box_threshold=0.35):
    """
    Baseline: GroundingDINO + SAM only (no LLaVA, no CLIP, no spatial filter).
    Takes the highest-confidence detection and segments it.
    """
    from groundingdino.util.inference import predict as gdino_predict
    from PIL import Image as PILImage
    import groundingdino.datasets.transforms as T

    H, W = image_np.shape[:2]

    # Prepare image for GroundingDINO
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_pil = PILImage.fromarray(image_np)
    image_tensor, _ = transform(image_pil, None)

    # Run GroundingDINO
    caption = query.lower().strip()
    if not caption.endswith("."):
        caption += "."

    device = next(gdino_model.parameters()).device
    image_tensor_dev = image_tensor.to(device)

    with torch.no_grad():
        outputs = gdino_model(image_tensor_dev[None], captions=[caption])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]
    filt = logits.max(dim=1)[0] > box_threshold

    boxes_filt = boxes[filt]
    scores_filt = logits.max(dim=1)[0][filt]

    if len(boxes_filt) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    # Take the BEST scoring detection (baseline = no verification)
    best_idx = scores_filt.argmax()
    best_box = boxes_filt[best_idx]

    # Convert cxcywh to xyxy
    cx, cy, w, h = best_box
    x1 = int((cx - w/2) * W)
    y1 = int((cy - h/2) * H)
    x2 = int((cx + w/2) * W)
    y2 = int((cy + h/2) * H)

    # Run SAM
    sam_predictor.set_image(image_np)
    input_box = np.array([[x1, y1, x2, y2]])
    masks, _, _ = sam_predictor.predict(
        point_coords=None, point_labels=None,
        box=input_box, multimask_output=False
    )

    pred_mask = np.zeros((H, W), dtype=np.uint8)
    if masks is not None and len(masks) > 0:
        pred_mask = (masks[0] > 0).astype(np.uint8)

    return pred_mask


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/challenging_subset")
    parser.add_argument("--output-dir", default="./outputs/baseline_comparison")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--clip-threshold", type=float, default=0.25)
    args = parser.parse_args()

    image_dir = os.path.join(args.data_dir, "original")
    label_dir = os.path.join(args.data_dir, "labels")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  BASELINE COMPARISON: GroundingDINO-only vs Full MAVR Pipeline")
    print("=" * 70)

    # Load models
    print("\n>> Loading models...")
    from src.model_loader import load_gdino_model, load_sam_predictor, load_clip_verifier
    gdino = load_gdino_model()
    sam = load_sam_predictor()
    clip_v = load_clip_verifier()
    print("[OK] All models loaded\n")

    from src.text_guided import run_text_guided_pipeline

    baseline_results = []
    mavr_results = []

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    for i, img_file in enumerate(image_files):
        query = QUERY_MAP.get(img_file)
        if query is None:
            continue

        print(f"\n{'─' * 70}")
        print(f"[{i+1}/{len(image_files)}] {img_file}")
        print(f"  Query: \"{query}\"")
        print(f"{'─' * 70}")

        img_path = os.path.join(image_dir, img_file)
        image_pil = Image.open(img_path).convert("RGB")
        image_np = np.array(image_pil)

        label_file = os.path.splitext(img_file)[0] + ".png"
        label_path = os.path.join(label_dir, label_file)
        if not os.path.exists(label_path):
            continue

        gt_mask = load_ground_truth_mask(label_path, image_np.shape)

        # ---- Baseline: GroundingDINO + SAM only ----
        print("  [Baseline] Running GroundingDINO + SAM only...")
        t0 = time.time()
        baseline_mask = run_baseline_gdino_only(
            image_np, query, gdino, sam, box_threshold=args.box_threshold
        )
        baseline_time = time.time() - t0
        baseline_metrics = compute_metrics(baseline_mask, gt_mask)
        baseline_metrics["image"] = img_file
        baseline_metrics["time"] = baseline_time
        baseline_results.append(baseline_metrics)
        print(f"  [Baseline] IoU={baseline_metrics['iou']:.4f}  F1={baseline_metrics['f1']:.4f}  ({baseline_time:.1f}s)")

        # ---- MAVR: Full Pipeline ----
        print("  [MAVR] Running full 7-step pipeline...")
        t0 = time.time()
        try:
            results = run_text_guided_pipeline(
                image_np=image_np, user_prompt=query, image_path=img_path,
                gdino_model=gdino, sam_predictor=sam, clip_verifier=clip_v,
                box_threshold=args.box_threshold, clip_threshold=args.clip_threshold,
            )
            mavr_time = time.time() - t0
            final_masks = results.get("final_masks")
            selected_idx = results.get("selected_idx")
            mavr_mask = create_predicted_mask(image_np, final_masks, selected_idx)
            mavr_metrics = compute_metrics(mavr_mask, gt_mask)
        except Exception as e:
            mavr_time = time.time() - t0
            print(f"  [MAVR ERROR] {e}")
            mavr_mask = np.zeros_like(gt_mask)
            mavr_metrics = {"iou": 0, "f1": 0, "precision": 0, "recall": 0}

        mavr_metrics["image"] = img_file
        mavr_metrics["time"] = mavr_time
        mavr_results.append(mavr_metrics)
        print(f"  [MAVR]     IoU={mavr_metrics['iou']:.4f}  F1={mavr_metrics['f1']:.4f}  ({mavr_time:.1f}s)")

        # Improvement
        iou_diff = mavr_metrics['iou'] - baseline_metrics['iou']
        symbol = "▲" if iou_diff > 0 else "▼" if iou_diff < 0 else "="
        print(f"  [{symbol}] IoU change: {iou_diff:+.4f}")

        # Free LLaVA
        try:
            import src.agents.vlm_backend as vlm_mod
            if hasattr(vlm_mod, '_model') and vlm_mod._model is not None:
                del vlm_mod._model; vlm_mod._model = None
                del vlm_mod._processor; vlm_mod._processor = None
                gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass

    # ============================================================
    # Summary Table
    # ============================================================
    print("\n" + "=" * 90)
    print("  COMPARISON RESULTS")
    print("=" * 90)
    header = f"{'Image':<40} | {'Baseline IoU':>12} | {'MAVR IoU':>10} | {'Improvement':>12}"
    print(header)
    print("─" * len(header))

    for b, m in zip(baseline_results, mavr_results):
        name = b['image'][:37] + "..." if len(b['image']) > 40 else b['image']
        diff = m['iou'] - b['iou']
        symbol = "▲" if diff > 0 else "▼" if diff < 0 else "="
        print(f"{name:<40} | {b['iou']:>12.4f} | {m['iou']:>10.4f} | {symbol} {diff:>+10.4f}")

    print("─" * len(header))

    avg_b_iou = np.mean([r['iou'] for r in baseline_results])
    avg_m_iou = np.mean([r['iou'] for r in mavr_results])
    avg_b_f1 = np.mean([r['f1'] for r in baseline_results])
    avg_m_f1 = np.mean([r['f1'] for r in mavr_results])
    avg_diff = avg_m_iou - avg_b_iou

    print(f"{'AVERAGE':<40} | {avg_b_iou:>12.4f} | {avg_m_iou:>10.4f} | {'▲' if avg_diff>0 else '▼'} {avg_diff:>+10.4f}")
    print()
    print(f"  Baseline Avg F1: {avg_b_f1:.4f}")
    print(f"  MAVR Avg F1:     {avg_m_f1:.4f}")
    print(f"  F1 Improvement:  {avg_m_f1 - avg_b_f1:+.4f}")

    n_better = sum(1 for b, m in zip(baseline_results, mavr_results) if m['iou'] > b['iou'])
    n_same = sum(1 for b, m in zip(baseline_results, mavr_results) if m['iou'] == b['iou'])
    n_worse = sum(1 for b, m in zip(baseline_results, mavr_results) if m['iou'] < b['iou'])
    print(f"\n  MAVR better: {n_better}/{len(baseline_results)}")
    print(f"  Same:        {n_same}/{len(baseline_results)}")
    print(f"  MAVR worse:  {n_worse}/{len(baseline_results)}")

    # ============================================================
    # Save comparison chart
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    names = [r['image'].replace('.jpg','').replace('animals','')[:15] for r in baseline_results]
    b_ious = [r['iou'] for r in baseline_results]
    m_ious = [r['iou'] for r in mavr_results]
    x = np.arange(len(names))
    width = 0.35

    # Chart 1: IoU comparison bars
    axes[0].bar(x - width/2, b_ious, width, label='Baseline (GDino-only)', color='#F44336', alpha=0.8)
    axes[0].bar(x + width/2, m_ious, width, label='MAVR (Full Pipeline)', color='#4CAF50', alpha=0.8)
    axes[0].axhline(y=avg_b_iou, color='#F44336', linestyle='--', alpha=0.6, label=f'Baseline Avg: {avg_b_iou:.3f}')
    axes[0].axhline(y=avg_m_iou, color='#4CAF50', linestyle='--', alpha=0.6, label=f'MAVR Avg: {avg_m_iou:.3f}')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('IoU Score')
    axes[0].set_title('IoU: Baseline vs MAVR', fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1)

    # Chart 2: Improvement per image
    diffs = [m - b for b, m in zip(b_ious, m_ious)]
    colors = ['#4CAF50' if d > 0 else '#F44336' for d in diffs]
    axes[1].bar(x, diffs, color=colors, edgecolor='white')
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('IoU Improvement')
    axes[1].set_title('MAVR Improvement over Baseline', fontweight='bold')

    # Chart 3: Summary metrics
    metrics_names = ['Avg IoU', 'Avg F1']
    baseline_vals = [avg_b_iou, avg_b_f1]
    mavr_vals = [avg_m_iou, avg_m_f1]
    x2 = np.arange(len(metrics_names))
    axes[2].bar(x2 - width/2, baseline_vals, width, label='Baseline', color='#F44336', alpha=0.8)
    axes[2].bar(x2 + width/2, mavr_vals, width, label='MAVR', color='#4CAF50', alpha=0.8)
    for j, (bv, mv) in enumerate(zip(baseline_vals, mavr_vals)):
        axes[2].text(j - width/2, bv + 0.02, f'{bv:.3f}', ha='center', fontsize=11, fontweight='bold')
        axes[2].text(j + width/2, mv + 0.02, f'{mv:.3f}', ha='center', fontsize=11, fontweight='bold')
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(metrics_names, fontsize=12)
    axes[2].set_title('Overall Comparison', fontweight='bold')
    axes[2].legend()
    axes[2].set_ylim(0, 1.15)

    plt.suptitle('Baseline (GroundingDINO-only) vs MAVR (Full Multi-Agent Pipeline)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    chart_path = os.path.join(args.output_dir, "baseline_comparison.jpg")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Chart saved: {chart_path}")

    # Save JSON
    results_json = {
        "baseline": {"per_image": baseline_results, "avg_iou": float(avg_b_iou), "avg_f1": float(avg_b_f1)},
        "mavr": {"per_image": mavr_results, "avg_iou": float(avg_m_iou), "avg_f1": float(avg_m_f1)},
        "improvement": {"iou": float(avg_diff), "f1": float(avg_m_f1 - avg_b_f1),
                        "n_better": n_better, "n_same": n_same, "n_worse": n_worse}
    }
    json_path = os.path.join(args.output_dir, "baseline_comparison.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"[OK] JSON saved: {json_path}")
    print("\n[DONE] Baseline comparison complete!")


if __name__ == "__main__":
    main()
