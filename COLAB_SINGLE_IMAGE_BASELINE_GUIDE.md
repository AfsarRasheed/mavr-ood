# Colab Guide: Single-Image Baseline vs MAVR Comparison

This guide explains how to run a **single-image comparison** in Google Colab between:

- **Baseline**: GroundingDINO + SAM
- **MAVR**: full text-guided MAVR pipeline

This is useful before running full dataset-wide baseline comparison.

---

## What This Notebook Flow Does

For one selected image, it will:

1. load the image
2. load the ground-truth mask
3. run the baseline
4. run the full MAVR pipeline
5. compute metrics for both
6. visualize:
   - original image
   - ground truth
   - baseline prediction
   - MAVR prediction

---

## Cell 1: Clone the Repo

```python
!git clone https://github.com/AfsarRasheed/mavr-ood.git
%cd /content/mavr-ood
!git checkout improvement/web-ui
```

---

## Cell 2: Install Dependencies

```python
!pip install -q -r requirements.txt
!pip install -q addict yapf
!pip install -q -e segment_anything/
!pip install -q git+https://github.com/openai/CLIP.git
```

---

## Cell 3: Build GroundingDINO for Python 3.12

```python
%cd /content/mavr-ood/GroundingDINO
!python setup.py build_ext --inplace
%cd /content/mavr-ood
```

---

## Cell 4: Fix Python Import Path

```python
import sys
sys.path.insert(0, "/content/mavr-ood/GroundingDINO")
```

---

## Cell 5: Download Model Weights

```python
import os
os.makedirs("weights", exist_ok=True)

if not os.path.exists("weights/groundingdino_swint_ogc.pth"):
    !wget -q -P weights/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

if not os.path.exists("weights/sam_vit_h_4b8939.pth"):
    !wget -q -P weights/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

---

## Cell 6: Optional Hugging Face Token

This is optional, but recommended to reduce rate-limit issues during model download.

```python
import os
os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"
```

If you do not have a token, you can skip this step.

---

## Cell 7: Make Sure You Are at Repo Root

```python
%cd /content/mavr-ood
```

---

## Cell 8: Run Single-Image Comparison

```python
import os
import sys
import time
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, "/content/mavr-ood/GroundingDINO")

# Choose one image and query
img_file = "animals03_Zebras_in_the_road.jpg"
query = "the zebra"

data_dir = "/content/mavr-ood/data/challenging_subset"
img_path = os.path.join(data_dir, "original", img_file)
label_path = os.path.join(data_dir, "labels", os.path.splitext(img_file)[0] + ".png")

from src.model_loader import load_gdino_model, load_sam_predictor, load_clip_verifier
from src.text_guided import run_text_guided_pipeline
from run_baseline_comparison import run_baseline_gdino_only
from run_evaluate_vlm import load_ground_truth_mask, compute_metrics, create_predicted_mask

# Load models
gdino = load_gdino_model()
sam = load_sam_predictor()
clip_v = load_clip_verifier()

# Load image and GT mask
image_pil = Image.open(img_path).convert("RGB")
image_np = np.array(image_pil)
gt_mask = load_ground_truth_mask(label_path, image_np.shape)

# Baseline
print("Running baseline...")
t0 = time.time()
baseline_mask = run_baseline_gdino_only(
    image_np=image_np,
    query=query,
    gdino_model=gdino,
    sam_predictor=sam,
    box_threshold=0.35
)
baseline_time = time.time() - t0
baseline_metrics = compute_metrics(baseline_mask, gt_mask)

# MAVR
print("Running MAVR...")
t0 = time.time()
results = run_text_guided_pipeline(
    image_np=image_np,
    user_prompt=query,
    image_path=img_path,
    gdino_model=gdino,
    sam_predictor=sam,
    clip_verifier=clip_v,
    box_threshold=0.35,
    clip_threshold=0.25,
)
mavr_time = time.time() - t0

final_masks = results.get("final_masks")
selected_idx = results.get("selected_idx")
mavr_mask = create_predicted_mask(image_np, final_masks, selected_idx)
mavr_metrics = compute_metrics(mavr_mask, gt_mask)

# Print results
print("\n=== RESULTS ===")
print(f"Image: {img_file}")
print(f"Query: {query}")
print()
print(f"Baseline Time: {baseline_time:.1f}s")
print(f"Baseline IoU: {baseline_metrics['iou']:.4f}")
print(f"Baseline F1: {baseline_metrics['f1']:.4f}")
print(f"Baseline Precision: {baseline_metrics['precision']:.4f}")
print(f"Baseline Recall: {baseline_metrics['recall']:.4f}")
print()
print(f"MAVR Time: {mavr_time:.1f}s")
print(f"MAVR IoU: {mavr_metrics['iou']:.4f}")
print(f"MAVR F1: {mavr_metrics['f1']:.4f}")
print(f"MAVR Precision: {mavr_metrics['precision']:.4f}")
print(f"MAVR Recall: {mavr_metrics['recall']:.4f}")
print()
print(f"IoU Improvement: {mavr_metrics['iou'] - baseline_metrics['iou']:+.4f}")

# Visualize
fig, axes = plt.subplots(1, 4, figsize=(22, 6))

axes[0].imshow(image_np)
axes[0].set_title("Original", fontsize=12, fontweight="bold")
axes[0].axis("off")

gt_overlay = image_np.copy()
gt_overlay[gt_mask > 0] = gt_overlay[gt_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
axes[1].imshow(gt_overlay.astype(np.uint8))
axes[1].set_title("Ground Truth", fontsize=12, fontweight="bold")
axes[1].axis("off")

baseline_overlay = image_np.copy()
baseline_overlay[baseline_mask > 0] = baseline_overlay[baseline_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
axes[2].imshow(baseline_overlay.astype(np.uint8))
axes[2].set_title(
    f"Baseline\nIoU={baseline_metrics['iou']:.3f}, F1={baseline_metrics['f1']:.3f}",
    fontsize=12,
    fontweight="bold"
)
axes[2].axis("off")

mavr_overlay = image_np.copy()
mavr_overlay[mavr_mask > 0] = mavr_overlay[mavr_mask > 0] * 0.5 + np.array([0, 0, 255]) * 0.5
axes[3].imshow(mavr_overlay.astype(np.uint8))
axes[3].set_title(
    f"MAVR\nIoU={mavr_metrics['iou']:.3f}, F1={mavr_metrics['f1']:.3f}",
    fontsize=12,
    fontweight="bold"
)
axes[3].axis("off")

plt.suptitle(f"Single Image Comparison: {img_file}", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
```

---

## To Test Another Image

In Cell 8, change these two lines:

```python
img_file = "animals03_Zebras_in_the_road.jpg"
query = "the zebra"
```

Example:

```python
img_file = "animals15_Doebeln_Pferdebahn.jpg"
query = "the horse"
```

---

## Expected Output

The notebook will display:

- original image
- ground-truth overlay
- baseline prediction overlay
- MAVR prediction overlay

It will also print:

- baseline runtime
- baseline IoU / F1 / Precision / Recall
- MAVR runtime
- MAVR IoU / F1 / Precision / Recall
- IoU improvement

---

## Notes

- This flow is for one image only
- It uses the built-in challenging subset already present in the repo
- It is a good first check before running `run_baseline_comparison.py` over the full dataset
