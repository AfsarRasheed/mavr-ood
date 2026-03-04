"""
MAVR Evaluation Script
Multi-Agent Vision-Language Reasoning for Reliable Object Localization
Runs the VLM pipeline on the challenging subset and computes IoU/F1 metrics
against ground truth masks.

Usage (Colab):
    !python run_evaluate_vlm.py

Usage (Local):
    python run_evaluate_vlm.py --no-cleanup
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
import gc
import json
import time
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# Monkey-patch transformers for GroundingDINO compatibility
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

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# ============================================================
# Image → Query Mapping
# Each image is mapped to a natural language query describing
# the anomalous object that should be detected.
# ============================================================
QUERY_MAP = {
    "animals03_Zebras_in_the_road.jpg": "the zebra",
    "animals05_cows_near_Phonsavan_Laos.jpg": "the cow",
    "animals06_sheep_roads_lambs.jpg": "the sheep",
    "animals07_Aihole_Pattadakal.jpg": "the cow",
    "animals09_working_together.jpg": "the donkey",
    "animals14_Dekemhare_Road.jpg": "the donkey",
    "animals15_Doebeln_Pferdebahn.jpg": "the horse",
    "animals16_Donkeys_of_Morocco.jpg": "the donkey",
    "animals21_On_The_Road_to_Gondar_Ethiopia.jpg": "the donkey",
    "animals23_rhino_crossing_road.jpg": "the rhino",
    "animals24_Sheep_Herders_Block_the_Road_Kaleybar_to_Eskanlu.jpg": "the sheep",
    "animals25_jablonna_dziki.jpg": "the wild boar",
    "animals26_Unnamed_Road_Kazakhstan.jpg": "the cow",
}


def load_ground_truth_mask(label_path, image_shape):
    """Load ground truth mask and resize to match image dimensions."""
    gt = np.array(Image.open(label_path).convert("L"))
    # Binary: anything > 0 is anomaly
    gt_binary = (gt > 0).astype(np.uint8)
    # Resize if needed
    if gt_binary.shape[:2] != image_shape[:2]:
        from PIL import Image as PILImage
        gt_pil = PILImage.fromarray(gt_binary * 255)
        gt_pil = gt_pil.resize((image_shape[1], image_shape[0]), PILImage.NEAREST)
        gt_binary = (np.array(gt_pil) > 127).astype(np.uint8)
    return gt_binary


def compute_metrics(pred_mask, gt_mask):
    """Compute IoU, F1, Precision, Recall between predicted and ground truth masks."""
    pred = pred_mask.astype(bool).flatten()
    gt = gt_mask.astype(bool).flatten()

    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    tn = np.sum(~pred & ~gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def create_predicted_mask(image_np, masks, selected_idx):
    """Create a binary mask from SAM's output masks."""
    H, W = image_np.shape[:2]
    pred_mask = np.zeros((H, W), dtype=np.uint8)

    if masks is None:
        return pred_mask

    # Handle single or multiple selected indices
    if isinstance(selected_idx, list):
        indices = selected_idx
    elif selected_idx is not None:
        indices = [selected_idx]
    else:
        return pred_mask

    for idx in indices:
        if idx < len(masks):
            mask = masks[idx]
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            if mask.ndim == 3:
                mask = mask[0]
            pred_mask = np.maximum(pred_mask, (mask > 0).astype(np.uint8))

    return pred_mask


def save_comparison_visualization(image_np, pred_mask, gt_mask, metrics, query, save_path):
    """Save a side-by-side visualization of prediction vs ground truth."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Original image with query
    axes[0].imshow(image_np)
    axes[0].set_title(f'Query: "{query}"', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Ground Truth mask
    gt_overlay = image_np.copy()
    gt_overlay[gt_mask > 0] = gt_overlay[gt_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    axes[1].imshow(gt_overlay.astype(np.uint8))
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Panel 3: Predicted mask
    pred_overlay = image_np.copy()
    pred_overlay[pred_mask > 0] = pred_overlay[pred_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    axes[2].imshow(pred_overlay.astype(np.uint8))
    axes[2].set_title(f'Predicted (IoU: {metrics["iou"]:.3f}, F1: {metrics["f1"]:.3f})',
                      fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MAVR VLM Pipeline Evaluation")
    parser.add_argument("--data-dir", default="./data/challenging_subset",
                        help="Path to challenging subset directory")
    parser.add_argument("--output-dir", default="./outputs/vlm_evaluation",
                        help="Output directory for results")
    parser.add_argument("--box-threshold", type=float, default=0.35,
                        help="GroundingDINO box threshold")
    parser.add_argument("--clip-threshold", type=float, default=0.25,
                        help="CLIP verification threshold")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Don't free LLaVA between images (faster but uses more VRAM)")
    args = parser.parse_args()

    image_dir = os.path.join(args.data_dir, "original")
    label_dir = os.path.join(args.data_dir, "labels")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("MAVR: Multi-Agent VLM Pipeline Evaluation")
    print("=" * 60)
    print(f"Images: {image_dir}")
    print(f"Labels: {label_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Thresholds: box={args.box_threshold}, clip={args.clip_threshold}")
    print("=" * 60)

    # ---- Load Models ----
    print("\n>> Loading models...")
    from src.model_loader import load_gdino_model, load_sam_predictor, load_clip_verifier

    gdino = load_gdino_model()
    sam = load_sam_predictor()
    clip_v = load_clip_verifier()
    print("[OK] All models loaded")

    from src.text_guided import run_text_guided_pipeline

    # ---- Run Evaluation ----
    all_results = []
    total_time = 0

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    for i, img_file in enumerate(image_files):
        print(f"\n{'─' * 60}")
        print(f"[{i+1}/{len(image_files)}] {img_file}")
        print(f"{'─' * 60}")

        # Get query for this image
        query = QUERY_MAP.get(img_file)
        if query is None:
            print(f"[SKIP] No query mapping for {img_file}")
            continue

        # Load image
        img_path = os.path.join(image_dir, img_file)
        image_pil = Image.open(img_path).convert("RGB")
        image_np = np.array(image_pil)

        # Load ground truth mask
        label_file = os.path.splitext(img_file)[0] + ".png"
        label_path = os.path.join(label_dir, label_file)
        if not os.path.exists(label_path):
            print(f"[SKIP] No ground truth: {label_path}")
            continue

        gt_mask = load_ground_truth_mask(label_path, image_np.shape)
        print(f"[i] Query: \"{query}\"")
        print(f"[i] Image size: {image_np.shape[:2]}")
        print(f"[i] GT mask pixels: {gt_mask.sum()} / {gt_mask.size}")

        # Run pipeline
        start = time.time()
        try:
            results = run_text_guided_pipeline(
                image_np=image_np,
                user_prompt=query,
                image_path=img_path,
                gdino_model=gdino,
                sam_predictor=sam,
                clip_verifier=clip_v,
                box_threshold=args.box_threshold,
                clip_threshold=args.clip_threshold,
            )
            elapsed = time.time() - start
            total_time += elapsed

            # Extract predicted mask
            final_masks = results.get("final_masks")
            selected_idx = results.get("selected_idx")
            pred_mask = create_predicted_mask(image_np, final_masks, selected_idx)

            # Compute metrics
            metrics = compute_metrics(pred_mask, gt_mask)
            metrics["image"] = img_file
            metrics["query"] = query
            metrics["time"] = elapsed

            print(f"\n[RESULT] IoU={metrics['iou']:.4f}  F1={metrics['f1']:.4f}  "
                  f"Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}  "
                  f"Time={elapsed:.1f}s")

            # Save visualization
            viz_path = os.path.join(args.output_dir, f"{os.path.splitext(img_file)[0]}_comparison.jpg")
            save_comparison_visualization(image_np, pred_mask, gt_mask, metrics, query, viz_path)
            print(f"[OK] Saved: {viz_path}")

            # Save step images if available
            step_images = results.get("step_images", {})
            for key, step_img in step_images.items():
                if step_img is not None:
                    step_path = os.path.join(args.output_dir, f"{os.path.splitext(img_file)[0]}_{key}.jpg")
                    plt.imsave(step_path, step_img)

            all_results.append(metrics)

        except Exception as e:
            elapsed = time.time() - start
            print(f"[ERROR] Pipeline failed: {e}")
            all_results.append({
                "image": img_file,
                "query": query,
                "iou": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "time": elapsed,
                "error": str(e),
            })

        # Free LLaVA memory between images
        if not args.no_cleanup:
            try:
                import src.agents.vlm_backend as vlm_mod
                if hasattr(vlm_mod, '_model') and vlm_mod._model is not None:
                    del vlm_mod._model
                    vlm_mod._model = None
                if hasattr(vlm_mod, '_processor') and vlm_mod._processor is not None:
                    del vlm_mod._processor
                    vlm_mod._processor = None
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)

    # Results table
    header = f"{'Image':<55} | {'IoU':>6} | {'F1':>6} | {'Prec':>6} | {'Rec':>6} | {'Time':>6}"
    print(header)
    print("─" * len(header))

    for r in all_results:
        name = r['image'][:52] + "..." if len(r['image']) > 55 else r['image']
        print(f"{name:<55} | {r['iou']:>6.4f} | {r['f1']:>6.4f} | "
              f"{r['precision']:>6.4f} | {r['recall']:>6.4f} | {r.get('time', 0):>5.1f}s")

    print("─" * len(header))

    # Compute averages
    if all_results:
        avg_iou = np.mean([r['iou'] for r in all_results])
        avg_f1 = np.mean([r['f1'] for r in all_results])
        avg_prec = np.mean([r['precision'] for r in all_results])
        avg_rec = np.mean([r['recall'] for r in all_results])
        avg_time = np.mean([r.get('time', 0) for r in all_results])

        print(f"{'AVERAGE':<55} | {avg_iou:>6.4f} | {avg_f1:>6.4f} | "
              f"{avg_prec:>6.4f} | {avg_rec:>6.4f} | {avg_time:>5.1f}s")

        # Count successes (IoU > 0.1)
        n_success = sum(1 for r in all_results if r['iou'] > 0.1)
        print(f"\nLocalization Success Rate: {n_success}/{len(all_results)} "
              f"({100*n_success/len(all_results):.1f}%)")
        print(f"Total Time: {total_time:.1f}s ({total_time/len(all_results):.1f}s per image)")

    # Save JSON results
    results_json = {
        "config": {
            "box_threshold": args.box_threshold,
            "clip_threshold": args.clip_threshold,
            "data_dir": args.data_dir,
        },
        "per_image": all_results,
        "summary": {
            "avg_iou": float(avg_iou) if all_results else 0,
            "avg_f1": float(avg_f1) if all_results else 0,
            "avg_precision": float(avg_prec) if all_results else 0,
            "avg_recall": float(avg_rec) if all_results else 0,
            "n_images": len(all_results),
            "n_success": n_success if all_results else 0,
            "total_time": total_time,
        }
    }

    json_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n[OK] Results saved: {json_path}")

    # Save summary plot
    if all_results:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Bar chart: IoU per image
        names = [r['image'].replace('.jpg', '').replace('animals', '').replace('_', ' ')[:20]
                 for r in all_results]
        ious = [r['iou'] for r in all_results]
        f1s = [r['f1'] for r in all_results]

        x = np.arange(len(names))
        width = 0.35

        axes[0].bar(x - width/2, ious, width, label='IoU', color='#4CAF50')
        axes[0].bar(x + width/2, f1s, width, label='F1', color='#2196F3')
        axes[0].axhline(y=avg_iou, color='#4CAF50', linestyle='--', alpha=0.7, label=f'Avg IoU: {avg_iou:.3f}')
        axes[0].axhline(y=avg_f1, color='#2196F3', linestyle='--', alpha=0.7, label=f'Avg F1: {avg_f1:.3f}')
        axes[0].set_xlabel('Image')
        axes[0].set_ylabel('Score')
        axes[0].set_title('IoU and F1 per Image', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        axes[0].legend(fontsize=8)
        axes[0].set_ylim(0, 1)

        # Pie chart: Success rate
        axes[1].pie([n_success, len(all_results) - n_success],
                    labels=['Success (IoU>0.1)', 'Failed'],
                    colors=['#4CAF50', '#F44336'],
                    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
        axes[1].set_title('Localization Success Rate', fontweight='bold')

        plt.tight_layout()
        summary_path = os.path.join(args.output_dir, "evaluation_summary.jpg")
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] Summary plot: {summary_path}")

    print("\n[DONE] Evaluation complete!")


if __name__ == "__main__":
    main()
