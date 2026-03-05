"""
Evaluate Text-Guided Pipeline on a Custom Image
Usage:
    python evaluate_custom_image.py \
        --image path/to/image.jpg \
        --mask path/to/mask.png \
        --query "the red car" \
        --out_dir ./outputs/custom_eval
"""

import matplotlib
matplotlib.use('Agg')

import os
import argparse
import time
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

# Reuse functions from the existing evaluation script
from run_evaluate_vlm import (
    load_ground_truth_mask,
    compute_metrics,
    create_predicted_mask,
    save_comparison_visualization
)

from src.model_loader import load_gdino_model, load_sam_predictor, load_clip_verifier
from src.text_guided.pipeline import run_text_guided_pipeline

def main():
    parser = argparse.ArgumentParser(description="Evaluate MAVR on a custom image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--mask", required=True, help="Path to ground truth binary mask")
    parser.add_argument("--query", required=True, help="Object to detect (e.g., 'the red car')")
    parser.add_argument("--out_dir", default="./outputs/custom_eval", help="Output directory")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="GroundingDINO threshold")
    parser.add_argument("--clip-threshold", type=float, default=0.25, help="CLIP threshold")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("MAVR Custom Image Evaluation")
    print(f"Image: {args.image}")
    print(f"Mask: {args.mask}")
    print(f"Query: '{args.query}'")
    print("=" * 60)

    # 1. Load models
    print("\n[i] Loading models...")
    gdino = load_gdino_model()
    sam = load_sam_predictor()
    clip_v = load_clip_verifier()
    print("[OK] Models loaded")

    # 2. Load data
    image_pil = Image.open(args.image).convert("RGB")
    image_np = np.array(image_pil)
    
    gt_mask = load_ground_truth_mask(args.mask, image_np.shape)
    print(f"[i] Image dimensions: {image_np.shape}")
    print(f"[i] GT mask target pixels: {gt_mask.sum()}")

    # 3. Run pipeline
    print("\n[i] Running Text-Guided Pipeline...")
    start_time = time.time()
    
    try:
        results = run_text_guided_pipeline(
            image_np=image_np,
            user_prompt=args.query,
            image_path=args.image,
            gdino_model=gdino,
            sam_predictor=sam,
            clip_verifier=clip_v,
            box_threshold=args.box_threshold,
            clip_threshold=args.clip_threshold
        )
        elapsed = time.time() - start_time

        # 4. Extract masks and compute metrics
        final_masks = results.get("final_masks")
        selected_idx = results.get("selected_idx")
        pred_mask = create_predicted_mask(image_np, final_masks, selected_idx)

        metrics = compute_metrics(pred_mask, gt_mask)
        metrics["image"] = os.path.basename(args.image)
        metrics["query"] = args.query
        metrics["time"] = elapsed

        print(f"\n[RESULT] IoU: {metrics['iou']:.4f} | F1: {metrics['f1']:.4f}")
        print(f"[RESULT] Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
        print(f"[RESULT] Time elapsed: {elapsed:.1f}s")
        print("-" * 60)
        print(f"Reasoning:\n{results.get('reasoning', 'N/A')}")
        print("-" * 60)

        # 5. Save visualizations
        img_name = os.path.splitext(os.path.basename(args.image))[0]
        
        # Save comparison
        viz_path = os.path.join(args.out_dir, f"{img_name}_eval_comparison.jpg")
        save_comparison_visualization(image_np, pred_mask, gt_mask, metrics, args.query, viz_path)
        print(f"[OK] Saved comparison image: {viz_path}")

        # Save pipeline step images
        step_images = results.get("step_images", {})
        for key, step_img in step_images.items():
            if step_img is not None:
                step_path = os.path.join(args.out_dir, f"{img_name}_{key}.jpg")
                plt.imsave(step_path, step_img)
        
        # Save JSON results
        json_path = os.path.join(args.out_dir, f"{img_name}_metrics.json")
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n[DONE] Evaluation complete! Check '{args.out_dir}' for results.")

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {str(e)}")


if __name__ == "__main__":
    main()
