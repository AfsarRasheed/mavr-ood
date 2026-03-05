# Text-Guided Pipeline — Custom Image Evaluation (Colab)

Run these cells **in order** in Google Colab to evaluate the Text-Guided pipeline on your own image + mask.

---

## Cell 1 — Upload Image & Mask + Set Query

```python
from google.colab import files
import shutil, os

# === Upload your image ===
print("Upload your IMAGE file:")
uploaded_img = files.upload()
img_name = list(uploaded_img.keys())[0]
img_path = f"/content/mavr-ood/{img_name}"
shutil.move(img_name, img_path)
print(f"Image saved: {img_path}")

# === Upload your ground truth mask ===
print("\nUpload your MASK file (binary/grayscale PNG where object = white):")
uploaded_mask = files.upload()
mask_name = list(uploaded_mask.keys())[0]
mask_path = f"/content/mavr-ood/{mask_name}"
shutil.move(mask_name, mask_path)
print(f"Mask saved: {mask_path}")

# === Set your query ===
QUERY = "the red car on the left"  # <-- CHANGE THIS to your query
print(f"\nQuery: '{QUERY}'")
```

---

## Cell 2 — Run Pipeline + Compute Metrics

```python
%cd /content/mavr-ood

%matplotlib inline
import matplotlib.pyplot as plt

import os, gc, time, json, re
import numpy as np
import torch
from PIL import Image

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

import sys
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# ---- Load Models ----
print("Loading models...")
from src.model_loader import load_gdino_model, load_sam_predictor, load_clip_verifier
gdino = load_gdino_model()
sam = load_sam_predictor()
clip_v = load_clip_verifier()
print("[OK] All models loaded\n")

# ---- Load Image & Mask ----
image_pil = Image.open(img_path).convert("RGB")
image_np = np.array(image_pil)

gt_pil = Image.open(mask_path).convert("L")
gt_np = np.array(gt_pil)
gt_binary = (gt_np > 0).astype(np.uint8)
if gt_binary.shape[:2] != image_np.shape[:2]:
    gt_pil_resized = gt_pil.resize((image_np.shape[1], image_np.shape[0]), Image.NEAREST)
    gt_binary = (np.array(gt_pil_resized) > 127).astype(np.uint8)

print(f"Image: {image_np.shape}")
print(f"GT mask pixels: {gt_binary.sum()} / {gt_binary.size}")
print(f"Query: \"{QUERY}\"\n")

# ---- Run Pipeline ----
from src.text_guided import run_text_guided_pipeline

start = time.time()
results = run_text_guided_pipeline(
    image_np=image_np,
    user_prompt=QUERY,
    image_path=img_path,
    gdino_model=gdino,
    sam_predictor=sam,
    clip_verifier=clip_v,
    box_threshold=0.35,
    clip_threshold=0.25,
)
elapsed = time.time() - start

# ---- Extract Predicted Mask ----
final_masks = results.get("final_masks")
H, W = image_np.shape[:2]
pred_mask = np.zeros((H, W), dtype=np.uint8)
if final_masks is not None:
    for i in range(len(final_masks)):
        m = final_masks[i]
        if hasattr(m, 'cpu'): m = m.cpu().numpy()
        if m.ndim == 3: m = m[0]
        pred_mask = np.maximum(pred_mask, (m > 0).astype(np.uint8))

# ---- Compute Metrics ----
pred_flat = pred_mask.astype(bool).flatten()
gt_flat = gt_binary.astype(bool).flatten()
tp = np.sum(pred_flat & gt_flat)
fp = np.sum(pred_flat & ~gt_flat)
fn = np.sum(~pred_flat & gt_flat)
tn = np.sum(~pred_flat & ~gt_flat)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
f1 = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0

print("\n" + "=" * 50)
print("         EVALUATION RESULTS")
print("=" * 50)
print(f"  IoU:       {iou:.4f}")
print(f"  F1:        {f1:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  Time:      {elapsed:.1f}s")
print("=" * 50)
print(f"\nStep 7 — Reasoning Agent Output:")
print(results.get('reasoning', 'N/A'))
```

---

## Cell 3 — 4-Panel Comparison (Original, GT, Prediction, Overlap)

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

axes[0].imshow(image_np)
axes[0].set_title(f'Query: "{QUERY}"', fontsize=11, fontweight='bold')
axes[0].axis('off')

gt_overlay = image_np.copy().astype(np.float32)
gt_overlay[gt_binary > 0] = gt_overlay[gt_binary > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
axes[1].imshow(gt_overlay.astype(np.uint8))
axes[1].set_title('Ground Truth (green)', fontsize=11, fontweight='bold')
axes[1].axis('off')

pred_overlay = image_np.copy().astype(np.float32)
pred_overlay[pred_mask > 0] = pred_overlay[pred_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
axes[2].imshow(pred_overlay.astype(np.uint8))
axes[2].set_title(f'Prediction (red) -- IoU: {iou:.3f}', fontsize=11, fontweight='bold')
axes[2].axis('off')

overlap = image_np.copy().astype(np.float32)
only_gt = (gt_binary > 0) & (pred_mask == 0)
only_pred = (pred_mask > 0) & (gt_binary == 0)
both = (gt_binary > 0) & (pred_mask > 0)
overlap[only_gt] = overlap[only_gt] * 0.4 + np.array([0, 255, 0]) * 0.6
overlap[only_pred] = overlap[only_pred] * 0.4 + np.array([255, 0, 0]) * 0.6
overlap[both] = overlap[both] * 0.3 + np.array([255, 255, 0]) * 0.7
axes[3].imshow(overlap.astype(np.uint8))
axes[3].set_title(f'Overlap -- F1: {f1:.3f}', fontsize=11, fontweight='bold')
axes[3].axis('off')

plt.suptitle(f"IoU={iou:.4f}  |  F1={f1:.4f}  |  Precision={precision:.4f}  |  Recall={recall:.4f}",
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
```

---

## Cell 4 — Pipeline Step Images (2x3 Grid)

```python
import matplotlib.pyplot as plt

step_images = results.get("step_images", {})
if step_images:
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    step_order = ["step1_scene", "step2_query", "step3_candidates",
                  "step4_clip", "step5_spatial", "step6_final"]
    titles = ["Step 1: Scene Understanding", "Step 2: Attribute Matching",
              "Step 3: GroundingDINO Candidates", "Step 4: CLIP Verification",
              "Step 5: Spatial Filter", "Step 6: SAM Segmentation"]
    for ax, key, title in zip(axes.flat, step_order, titles):
        img = step_images.get(key)
        if img is not None:
            ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("No step images available.")
```

---

## Cell 5 — Metrics Pie Charts

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

metrics_data = [("IoU", iou), ("F1 Score", f1), ("Precision", precision), ("Recall", recall)]
colors_pass = ['#4CAF50', '#E0E0E0']
colors_fail = ['#F44336', '#E0E0E0']

for ax, (name, value) in zip(axes, metrics_data):
    color = colors_pass if value >= 0.5 else colors_fail
    ax.pie([value, 1 - value],
           labels=[f"{value:.2%}", f"{1-value:.2%}"],
           colors=color, startangle=90,
           textprops={'fontsize': 13, 'fontweight': 'bold'},
           wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax.set_title(f"{name}\n{value:.4f}", fontsize=14, fontweight='bold', pad=15)

plt.suptitle("Evaluation Metrics", fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.show()
```

---

## Cell 6 — Bar Chart (All Metrics)

```python
import matplotlib.pyplot as plt

metrics_names = ['IoU', 'F1', 'Precision', 'Recall']
metrics_values = [iou, f1, precision, recall]
colors = ['#4CAF50' if v >= 0.5 else '#F44336' for v in metrics_values]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(metrics_names, metrics_values, color=colors, edgecolor='white', linewidth=2, width=0.6)
for bar, val in zip(bars, metrics_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.4f}', ha='center', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.set_ylabel('Score', fontsize=12)
ax.set_title(f'Evaluation Metrics — "{QUERY}"', fontsize=14, fontweight='bold')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Cell 7 — Radar / Spider Chart

```python
import matplotlib.pyplot as plt
import numpy as np

categories = ['IoU', 'F1', 'Precision', 'Recall']
values = [iou, f1, precision, recall]
values += values[:1]

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, values, color='#2196F3', alpha=0.25)
ax.plot(angles, values, color='#2196F3', linewidth=2, marker='o', markersize=8)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax.set_title('Detection Performance Radar', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()
```

---

## Cell 8 — Confusion Matrix Heatmap

```python
import matplotlib.pyplot as plt
import numpy as np

conf_matrix = np.array([[int(tn), int(fp)],
                         [int(fn), int(tp)]])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(conf_matrix, cmap='Blues')

labels = [['TN', 'FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        color = 'white' if conf_matrix[i, j] > conf_matrix.max() * 0.5 else 'black'
        ax.text(j, i, f'{labels[i][j]}\n{conf_matrix[i][j]:,}',
                ha='center', va='center', fontsize=14, fontweight='bold', color=color)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Negative', 'Positive'], fontsize=11)
ax.set_yticklabels(['Negative', 'Positive'], fontsize=11)
ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
ax.set_title('Pixel-Level Confusion Matrix', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()
```

---

## Cell 9 — CLIP Scores Per Candidate

```python
import matplotlib.pyplot as plt
import re

summary = results.get("summary", "")
clip_lines = re.findall(r'#(\d+).*?CLIP=([\d.]+)\s*\[(PASS|REJECT)\]', summary)

if clip_lines:
    fig, ax = plt.subplots(figsize=(8, 5))
    candidates = [f'Candidate #{c[0]}' for c in clip_lines]
    scores = [float(c[1]) for c in clip_lines]
    colors = ['#4CAF50' if c[2] == 'PASS' else '#F44336' for c in clip_lines]

    bars = ax.bar(candidates, scores, color=colors, edgecolor='white', linewidth=2)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

    ax.axhline(y=0.25, color='orange', linestyle='--', linewidth=2, label='CLIP threshold (0.25)')
    ax.set_ylabel('CLIP Similarity Score', fontsize=12)
    ax.set_title('CLIP Verification Scores per Candidate', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(scores) + 0.15)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
else:
    print("No CLIP score data found in summary.")
```

---

## Cell 10 — Mask Contour Overlay

```python
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.patches import Patch

fig, ax = plt.subplots(figsize=(10, 7))
display_img = image_np.copy()

gt_contours, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(display_img, gt_contours, -1, (0, 255, 0), 3)

pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(display_img, pred_contours, -1, (255, 0, 0), 3)

ax.imshow(display_img)
ax.set_title(f'Contour Overlay — Green: GT, Red: Prediction\nIoU: {iou:.4f}',
             fontsize=13, fontweight='bold')
ax.axis('off')

legend_elements = [Patch(facecolor='green', label='Ground Truth'),
                   Patch(facecolor='red', label='Prediction')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
plt.tight_layout()
plt.show()
```

---

## Cell 11 — Mask Area Comparison

```python
import matplotlib.pyplot as plt
import numpy as np

gt_area = gt_binary.sum()
pred_area = pred_mask.sum()
overlap_area = int(tp)
total_pixels = gt_binary.size

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

labels = ['Ground Truth', 'Prediction', 'Overlap']
areas = [gt_area, pred_area, overlap_area]
colors = ['#4CAF50', '#F44336', '#FFC107']
bars = axes[0].bar(labels, areas, color=colors, edgecolor='white', linewidth=2)
for bar, val in zip(bars, areas):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(areas)*0.02,
                 f'{val:,} px', ha='center', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Pixel Count', fontsize=12)
axes[0].set_title('Mask Area Comparison', fontsize=14, fontweight='bold')

sizes = [overlap_area, gt_area - overlap_area, pred_area - overlap_area,
         total_pixels - gt_area - pred_area + overlap_area]
labels2 = [f'TP\n{overlap_area:,}', f'FN\n{gt_area-overlap_area:,}',
           f'FP\n{pred_area-overlap_area:,}', f'TN']
colors2 = ['#FFC107', '#4CAF50', '#F44336', '#E0E0E0']
wedges, texts = axes[1].pie(sizes, labels=labels2, colors=colors2, startangle=90,
                             textprops={'fontsize': 10, 'fontweight': 'bold'},
                             wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Pixel Classification Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
```

---

## Cell 12 — Pipeline Summary Dashboard

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(image_np)
ax1.set_title(f'Input Image\nQuery: "{QUERY}"', fontsize=10, fontweight='bold')
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(gt_binary, cmap='Greens')
ax2.set_title('Ground Truth Mask', fontsize=10, fontweight='bold')
ax2.axis('off')

ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(pred_mask, cmap='Reds')
ax3.set_title('Predicted Mask', fontsize=10, fontweight='bold')
ax3.axis('off')

ax4 = fig.add_subplot(gs[1, 0])
m_names = ['IoU', 'F1', 'Prec', 'Rec']
m_vals = [iou, f1, precision, recall]
bars = ax4.barh(m_names, m_vals, color=['#2196F3','#4CAF50','#FF9800','#9C27B0'])
for bar, val in zip(bars, m_vals):
    ax4.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=11, fontweight='bold')
ax4.set_xlim(0, 1.2)
ax4.set_title('Metrics', fontsize=10, fontweight='bold')

ax5 = fig.add_subplot(gs[1, 1])
ax5.axis('off')
info_lines = [
    f"Scene: {results.get('scene_result', {}).get('scene_type', 'N/A')}",
    f"Lighting: {results.get('scene_result', {}).get('lighting', 'N/A')}",
    f"Candidates: {results.get('n_detected', 'N/A')}",
    f"CLIP Verified: {results.get('n_verified', 'N/A')}",
    f"Selected: {results.get('n_selected', 'N/A')}",
    f"Time: {elapsed:.1f}s",
]
ax5.text(0.1, 0.9, "\n".join(info_lines), transform=ax5.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))
ax5.set_title('Pipeline Summary', fontsize=10, fontweight='bold')

ax6 = fig.add_subplot(gs[1, 2])
overlap = image_np.copy().astype(np.float32)
both = (gt_binary > 0) & (pred_mask > 0)
only_gt = (gt_binary > 0) & (pred_mask == 0)
only_pred = (pred_mask > 0) & (gt_binary == 0)
overlap[both] = overlap[both] * 0.3 + np.array([255, 255, 0]) * 0.7
overlap[only_gt] = overlap[only_gt] * 0.4 + np.array([0, 255, 0]) * 0.6
overlap[only_pred] = overlap[only_pred] * 0.4 + np.array([255, 0, 0]) * 0.6
ax6.imshow(overlap.astype(np.uint8))
ax6.set_title('Overlap View', fontsize=10, fontweight='bold')
ax6.axis('off')

plt.suptitle('MAVR Text-Guided Pipeline — Evaluation Dashboard',
             fontsize=15, fontweight='bold', y=1.02)
plt.show()
```

---

## Cell 13 — Difference Map (TP / FN / FP)

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(gt_binary, cmap='Greens', vmin=0, vmax=1)
axes[0].set_title(f'Ground Truth\n({gt_binary.sum():,} pixels)', fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(pred_mask, cmap='Reds', vmin=0, vmax=1)
axes[1].set_title(f'Prediction\n({pred_mask.sum():,} pixels)', fontsize=12, fontweight='bold')
axes[1].axis('off')

diff = np.zeros((*gt_binary.shape, 3), dtype=np.uint8)
diff[(gt_binary > 0) & (pred_mask > 0)] = [255, 255, 0]
diff[(gt_binary > 0) & (pred_mask == 0)] = [0, 255, 0]
diff[(gt_binary == 0) & (pred_mask > 0)] = [255, 0, 0]
axes[2].imshow(diff)
axes[2].set_title(f'Difference Map\nYellow=TP, Green=FN, Red=FP', fontsize=12, fontweight='bold')
axes[2].axis('off')

plt.suptitle(f'IoU: {iou:.4f}  |  F1: {f1:.4f}', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
```

---

## Cell 14 — Metrics Table

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

table_data = [
    ['IoU', f'{iou:.4f}', 'Overlap / Union of masks'],
    ['F1 Score', f'{f1:.4f}', 'Harmonic mean of Precision & Recall'],
    ['Precision', f'{precision:.4f}', 'Correct pixels / Total predicted pixels'],
    ['Recall', f'{recall:.4f}', 'Correct pixels / Total GT pixels'],
    ['True Positives', f'{int(tp):,}', 'Correctly detected pixels'],
    ['False Positives', f'{int(fp):,}', 'Wrongly detected pixels'],
    ['False Negatives', f'{int(fn):,}', 'Missed pixels'],
    ['Processing Time', f'{elapsed:.1f}s', 'Total pipeline runtime'],
]

table = ax.table(
    cellText=table_data,
    colLabels=['Metric', 'Value', 'Description'],
    cellLoc='center',
    loc='center',
    colWidths=[0.25, 0.15, 0.5]
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)

for j in range(3):
    table[0, j].set_facecolor('#1565C0')
    table[0, j].set_text_props(color='white', fontweight='bold')

for i in range(1, 5):
    val = float(table_data[i-1][1])
    color = '#E8F5E9' if val >= 0.5 else '#FFEBEE'
    for j in range(3):
        table[i, j].set_facecolor(color)

for i in range(5, 9):
    for j in range(3):
        table[i, j].set_facecolor('#F5F5F5')

plt.title(f'Evaluation Metrics — "{QUERY}"', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()
```
