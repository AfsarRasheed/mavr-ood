# MAVR-OOD: Google Colab Implementation Guide
> Complete step-by-step cells to run the multi-agent OOD detection pipeline on Google Colab (T4 GPU)
>
> **Tested:** Feb 2026 | **Runtime:** T4 GPU | **Results:** mIoU 0.97, F1 0.99 on test image

---

## Phase 1: Setup (Run once per session)

### Cell 1 — Clone Repository (first time only)
```python
# First time: clone the repo
!git clone https://github.com/AfsarRasheed/mavr-ood.git
%cd mavr-ood

# Returning session: just pull latest
# %cd mavr-ood
# !git checkout -- .
# !git pull origin main
```

### Cell 2 — Install Dependencies
> Run this every new Colab session — Colab resets packages on runtime restart.
```python
!pip install -q -r requirements.txt
!pip install -q gradio>=4.0.0 addict yapf bitsandbytes>=0.41.0
!pip install -q -e segment_anything/
!cd GroundingDINO && pip install -q -e . && cd ..

# Verify critical packages
import torch, bitsandbytes
print(f"[OK] torch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"[OK] bitsandbytes installed (4-bit quantization enabled)")
print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
print(f"[OK] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("[OK] All dependencies installed!")
```

### Cell 3 — Download Weights (~3 min, first time only)
```python
import os
if not os.path.exists("weights/groundingdino_swint_ogc.pth"):
    !mkdir -p weights
    !wget -q -P weights/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    !wget -q -P weights/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    print("[OK] Weights downloaded!")
else:
    print("[OK] Weights already exist!")
```

---

## Phase 2A: Single Image Test (Quick validation ~15 min)

### Cell 4A — Prepare Single Image Test
```python
!mkdir -p ./data/test_single/original
!mkdir -p ./data/test_single/labels
!cp ./data/challenging_subset/original/animals03_Zebras_in_the_road.jpg ./data/test_single/original/
!cp ./data/challenging_subset/labels/animals03_Zebras_in_the_road.png ./data/test_single/labels/
print("[OK] Test folder ready (1 image)")
```

### Cell 5A — Run All 5 Agents (~12 min)
> Each agent loads LLaVA-7B (4-bit quantized, ~4GB VRAM) and analyzes the image.
```python
!python src/agents/run_all_agents.py \
    --image_dir ./data/test_single/original \
    --output_dir ./outputs/test_single_prompts \
    --delay 2
```

### Cell 6A — Run Evaluation (GroundingDINO + CLIP + SAM)
```python
!python run_evaluate.py \
    --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint weights/groundingdino_swint_ogc.pth \
    --sam_checkpoint weights/sam_vit_h_4b8939.pth \
    --dataset_dir ./data/test_single \
    --dataset_type road_anomaly \
    --multiagent_prompts ./outputs/test_single_prompts/agent5_final_synthesis_results.json \
    --output_dir ./outputs/test_single_results \
    --clip_threshold 0.20 \
    --device cuda
```

### Cell 7A — View Results & Agent Reasoning
```python
from IPython.display import display, Image as IPImage
import glob, os, json

# Show metrics
with open("outputs/test_single_results/multiagent_evaluation_results.json") as f:
    results = json.load(f)
    m = results["average_metrics"]
    print(f"mIoU: {m['mIoU']:.4f} | F1: {m['F1']:.4f} | Precision: {m['Precision']:.4f} | Recall: {m['Recall']:.4f}")
    print(f"Detection Rate: {results['detection_rate']}%\n")

# Show agent reasoning
with open("outputs/test_single_prompts/agent5_final_synthesis_results.json") as f:
    agent5 = json.load(f)
    for r in agent5.get("results", []):
        img = r.get("image", "")
        syn = r.get("synthesis_result", {})
        prompts = syn.get("grounded_sam_prompts", {})
        print(f"\n[>>] {img}")
        print(f"   Prompt V1: {prompts.get('prompt_v1', 'N/A')}")
        print(f"   Prompt V2: {prompts.get('prompt_v2', 'N/A')}")
        print(f"   Confidence: {syn.get('overall_confidence', 'N/A')}")
        print(f"   Anomaly Type: {syn.get('anomaly_type', 'N/A')}")
        print(f"   Reasoning: {syn.get('reasoning', 'N/A')}")

# Show visualizations
print("\n" + "="*60)
print("PIPELINE REASONING & VISUALIZATION")
for r in sorted(glob.glob("outputs/test_single_results/*_pipeline_vis.jpg")):
    print(f"\n>> {os.path.basename(r)}")
    display(IPImage(r, width=1200))

print("\n" + "="*60)
print("MULTI-AGENT ANALYSIS DASHBOARD")
for r in sorted(glob.glob("outputs/test_single_results/*_agent_summary.jpg")):
    print(f"\n>> {os.path.basename(r)}")
    display(IPImage(r, width=1200))

print("\n" + "="*60)
print("SYSTEM BALANCE (RADAR CHART)")
for r in sorted(glob.glob("outputs/test_single_results/*_spider_chart.jpg")):
    print(f"\n>> {os.path.basename(r)}")
    display(IPImage(r, width=600))

print("\n" + "="*60)
print("METRICS BAR CHART")
for r in sorted(glob.glob("outputs/test_single_results/*_metrics_bar.jpg")):
    print(f"\n>> {os.path.basename(r)}")
    display(IPImage(r, width=600))

print("\n" + "="*60)
print("CLIP VERIFIER HEATMAP")
for r in sorted(glob.glob("outputs/test_single_results/*_clip_heatmap.jpg")):
    print(f"\n>> {os.path.basename(r)}")
    display(IPImage(r, width=800))
```

---

## Phase 2C: Text-Guided Detection (Run directly in Colab)

### Cell 4C -- Setup Test Image
```python
# Use any image -- here we use the existing zebra image
import os
TEST_IMAGE = "./data/challenging_subset/original/animals03_Zebras_in_the_road.jpg"
USER_PROMPT = "the zebra"  # Change this to any prompt

# Or upload your own image:
# from google.colab import files
# uploaded = files.upload()
# TEST_IMAGE = list(uploaded.keys())[0]

print(f"[OK] Image: {TEST_IMAGE}")
print(f"[OK] Prompt: {USER_PROMPT}")
```

### Cell 5C -- Run Text-Guided Detection Pipeline
> Memory-managed: LLaVA agents run first, then freed, then detection models load.
```python
import sys, gc
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "GroundingDINO")

# Load image
image_pil = Image.open(TEST_IMAGE).convert("RGB")
image_np = np.array(image_pil)

# --- Phase 1: Run LLaVA agents FIRST (before detection models) ---
gc.collect()
torch.cuda.empty_cache()

from text_guided_detector import scene_understanding, attribute_matching_agent, parse_query

print(">> Phase 1: Running LLaVA agents...")
scene_result = scene_understanding(TEST_IMAGE)
attr_result = attribute_matching_agent(TEST_IMAGE, scene_result, USER_PROMPT)

# Free LLaVA from GPU
import src.agents.vlm_backend as vlm_mod
if hasattr(vlm_mod, '_model') and vlm_mod._model is not None:
    del vlm_mod._model
    vlm_mod._model = None
if hasattr(vlm_mod, '_processor') and vlm_mod._processor is not None:
    del vlm_mod._processor
    vlm_mod._processor = None
gc.collect()
torch.cuda.empty_cache()
print("[OK] LLaVA freed from GPU")

# --- Phase 2: Load detection models and run pipeline ---
print("\n>> Phase 2: Running detection pipeline...")
from app import load_gdino_model, load_sam_predictor, load_clip_verifier
from text_guided_detector import run_text_guided_pipeline

gdino = load_gdino_model()
sam = load_sam_predictor()
clip_v = load_clip_verifier()

# Run pipeline (LLaVA steps already done, will reuse cached results)
results = run_text_guided_pipeline(
    image_np=image_np,
    user_prompt=USER_PROMPT,
    image_path=TEST_IMAGE,
    gdino_model=gdino,
    sam_predictor=sam,
    clip_verifier=clip_v,
    box_threshold=0.25,
    clip_threshold=0.20,
    # Pass pre-computed agent results to skip re-loading LLaVA
    precomputed_scene=scene_result,
    precomputed_attr=attr_result,
)

print("\n[OK] Text-guided detection complete!")
```

### Cell 6C -- View Step-by-Step Results
```python
from IPython.display import display, Image as IPImage
import matplotlib.pyplot as plt

step_images = results.get("step_images", {})
step_names = [
    ("step1_scene", "Step 1: Scene Understanding (LLaVA)"),
    ("step2_query", "Step 2: Query Parsing"),
    ("step3_candidates", "Step 3: Candidates (GroundingDINO)"),
    ("step4_clip", "Step 4: CLIP Verification"),
    ("step5_spatial", "Step 5: Spatial Selection"),
    ("step6_final", "Step 6: Final Segmentation (SAM)"),
]

fig, axes = plt.subplots(2, 3, figsize=(24, 12))
for idx, (key, title) in enumerate(step_names):
    ax = axes[idx // 3][idx % 3]
    img = step_images.get(key)
    if img is not None:
        ax.imshow(img)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

fig.suptitle(f'Text-Guided Detection: "{USER_PROMPT}"', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/text_guided_pipeline.jpg", dpi=150, bbox_inches='tight')
plt.show()

# Print summary log
print("\n" + results.get("summary", ""))
```

### Cell 7C -- Try Different Prompts (Optional)
```python
# Try more complex queries on the same image or different images
test_cases = [
    ("./data/challenging_subset/original/animals03_Zebras_in_the_road.jpg", "the largest zebra"),
    ("./data/challenging_subset/original/animals01_Horse_in_the_road.jpg", "the horse"),
]

for img_path, prompt in test_cases:
    image_np = np.array(Image.open(img_path).convert("RGB"))
    results = run_text_guided_pipeline(
        image_np=image_np,
        user_prompt=prompt,
        image_path=img_path,
        gdino_model=gdino,
        sam_predictor=sam,
        clip_verifier=clip_v,
    )
    # Show final result
    final = results["step_images"].get("step6_final")
    if final is not None:
        plt.figure(figsize=(10, 6))
        plt.imshow(final)
        plt.title(f'"{prompt}"', fontsize=14)
        plt.axis('off')
        plt.show()
    print(f'[OK] "{prompt}" -- done\n')
```

---

## Phase 2B: Full Dataset Run (13 images, ~60 min)

### Cell 4B — Run All Agents on Full Dataset (~45 min)
```python
!python src/agents/run_all_agents.py \
    --image_dir ./data/challenging_subset/original \
    --output_dir ./outputs/challenging_subset_prompts \
    --delay 2
```

### Cell 5B — Evaluate Full Dataset
```python
!python run_evaluate.py \
    --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint weights/groundingdino_swint_ogc.pth \
    --sam_checkpoint weights/sam_vit_h_4b8939.pth \
    --dataset_dir ./data/challenging_subset \
    --dataset_type road_anomaly \
    --multiagent_prompts ./outputs/challenging_subset_prompts/agent5_final_synthesis_results.json \
    --output_dir ./outputs/evaluation_results \
    --clip_threshold 0.20 \
    --device cuda
```

### Cell 6B — View All Results & Agent Reasoning
```python
from IPython.display import display, Image as IPImage
import glob, os, json

# Metrics
with open("outputs/evaluation_results/multiagent_evaluation_results.json") as f:
    results = json.load(f)
    m = results["average_metrics"]
    print(f"mIoU: {m['mIoU']:.4f} | F1: {m['F1']:.4f} | Precision: {m['Precision']:.4f} | Recall: {m['Recall']:.4f}")
    print(f"Detection Rate: {results['detection_rate']}%")

# Agent reasoning per image
with open("outputs/challenging_subset_prompts/agent5_final_synthesis_results.json") as f:
    agent5 = json.load(f)
    print(f"\n{'='*60}")
    print("AGENT REASONING")
    print(f"{'='*60}")
    for r in agent5.get("results", []):
        img = r.get("image", "")
        syn = r.get("synthesis_result", {})
        prompts = syn.get("grounded_sam_prompts", {})
        print(f"\n>> {img}")
        print(f"   V1: {prompts.get('prompt_v1', 'N/A')} | V2: {prompts.get('prompt_v2', 'N/A')}")
        print(f"   Confidence: {syn.get('overall_confidence', 'N/A')} | Type: {syn.get('anomaly_type', 'N/A')}")
        print(f"   Reasoning: {syn.get('reasoning', 'N/A')[:200]}")
# Visualizations
print(f"\n{'='*60}")
print("PIPELINE REASONING & VISUALIZATION")
for r in sorted(glob.glob("outputs/evaluation_results/*_pipeline_vis.jpg")):
    print(f"\n>> {os.path.basename(r)}")
    display(IPImage(r, width=1200))

print("\n" + "="*60)
print("MULTI-AGENT ANALYSIS DASHBOARD")
for r in sorted(glob.glob("outputs/evaluation_results/*_agent_summary.jpg")):
    print(f"\n>> {os.path.basename(r)}")
    display(IPImage(r, width=1200))

print("\n" + "="*60)
print("SYSTEM BALANCE (RADAR CHART)")
for r in sorted(glob.glob("outputs/evaluation_results/*_spider_chart.jpg")):
    print(f"\n>> {os.path.basename(r)}")
    display(IPImage(r, width=600))

print("\n" + "="*60)
print("METRICS BAR CHART")
for r in sorted(glob.glob("outputs/evaluation_results/*_metrics_bar.jpg")):
    print(f"\n>> {os.path.basename(r)}")
    display(IPImage(r, width=600))

print("\n" + "="*60)
print("CLIP VERIFIER HEATMAP")
for r in sorted(glob.glob("outputs/evaluation_results/*_clip_heatmap.jpg")):
    print(f"\n>> {os.path.basename(r)}")
    display(IPImage(r, width=800))

print("\n" + "="*60)
print("SYSTEM SELF-AWARENESS (SCATTER PLOT)")
for r in sorted(glob.glob("outputs/evaluation_results/confidence_vs_miou_scatter.jpg")):
    print(f"\n>> {os.path.basename(r)}")
    display(IPImage(r, width=800))

print("\n" + "="*60)
print("SYSTEM CONFUSION MATRIX")
if os.path.exists("outputs/evaluation_results/batch_confusion_matrix.jpg"):
    display(IPImage("outputs/evaluation_results/batch_confusion_matrix.jpg", width=800))
```

---
## Phase 3: Web App (Optional — choose one)

### Option A — Advanced Gradio Dashboard (Recommended)
Launch the Gradio UI with live visual analytics, bounding boxes, SAM masks, pipeline view, and reasoning logs.
```python
import app
demo = app.build_app()
demo.launch(share=True)
```

### Option B — Streamlit App
```python
!pip install -q streamlit pyngrok
```
```python
# Cell 1: Start Streamlit server
!nohup streamlit run streamlit_app.py --server.port 8501 --server.headless true &

# Cell 2: Create public URL
from pyngrok import ngrok
url = ngrok.connect(8501)
print(f"Open Streamlit app: {url}")
```
> **Note:** Streamlit includes ground truth mask upload for computing metrics (mIoU, F1) directly in the UI.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `CUDA out of memory` | Ensure `bitsandbytes` is installed: `!pip install -q bitsandbytes>=0.41.0` |
| `No module named 'addict'` | Run Cell 2 again |
| `cannot import sam_model_registry` | Run `!pip install -q -e segment_anything/` |
| `get_extended_attention_mask` TypeError | Already fixed in `run_evaluate.py` — just `!git pull origin main` |
| Session disconnects | Re-run from Cell 2 (deps), Cell 3 will skip if weights exist |
| Agent JSON parsing failed | Non-critical — Agent 5 still synthesizes from raw output |
| `git pull` blocked by local changes | Run `!git checkout -- .` then `!git pull origin main` |

---

## Architecture Overview

```
Image -> Agent 1 (Scene Context)     -+
      -> Agent 2 (Spatial Anomaly)    -+
      -> Agent 3 (Semantic Analysis)  -+-> Agent 5 (Synthesis) -> prompt_v1, prompt_v2
      -> Agent 4 (Visual Appearance)  -+           |
                                         GroundingDINO (detection)
                                              |
                                         CLIP (verification)
                                              |
                                         SAM (segmentation)
                                              |
                                         Evaluation (mIoU, F1)
```
