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
> ⚠️ **Run this every new Colab session** — Colab resets packages on runtime restart.
```python
!pip install -q -r requirements.txt
!pip install -q gradio>=4.0.0 addict yapf bitsandbytes>=0.41.0
!pip install -q -e segment_anything/
!cd GroundingDINO && pip install -q -e . && cd ..

# Verify critical packages
import torch, bitsandbytes
print(f"✅ torch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"✅ bitsandbytes installed (4-bit quantization enabled)")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("✅ All dependencies installed!")
```

### Cell 3 — Download Weights (~3 min, first time only)
```python
import os
if not os.path.exists("weights/groundingdino_swint_ogc.pth"):
    !mkdir -p weights
    !wget -q -P weights/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    !wget -q -P weights/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    print("✅ Weights downloaded!")
else:
    print("✅ Weights already exist!")
```

---

## Phase 2A: Single Image Test (Quick validation ~15 min)

### Cell 4A — Prepare Single Image Test
```python
!mkdir -p ./data/test_single/original
!mkdir -p ./data/test_single/labels
!cp ./data/challenging_subset/original/animals03_Zebras_in_the_road.jpg ./data/test_single/original/
!cp ./data/challenging_subset/labels/animals03_Zebras_in_the_road.png ./data/test_single/labels/
print("✅ Test folder ready (1 image)")
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
    print(f"📊 mIoU: {m['mIoU']:.4f} | F1: {m['F1']:.4f} | Precision: {m['Precision']:.4f} | Recall: {m['Recall']:.4f}")
    print(f"📊 Detection Rate: {results['detection_rate']}%\n")

# Show agent reasoning
with open("outputs/test_single_prompts/agent5_final_synthesis_results.json") as f:
    agent5 = json.load(f)
    for r in agent5.get("results", []):
        img = r.get("image", "")
        syn = r.get("synthesis_result", {})
        prompts = syn.get("grounded_sam_prompts", {})
        print(f"\n🤖 {img}")
        print(f"   Prompt V1: {prompts.get('prompt_v1', 'N/A')}")
        print(f"   Prompt V2: {prompts.get('prompt_v2', 'N/A')}")
        print(f"   Confidence: {syn.get('overall_confidence', 'N/A')}")
        print(f"   Anomaly Type: {syn.get('anomaly_type', 'N/A')}")
        print(f"   Reasoning: {syn.get('reasoning', 'N/A')}")

# Show visualizations
print("\n" + "="*60)
print("🧠 PIPELINE REASONING & VISUALIZATION")
for r in sorted(glob.glob("outputs/test_single_results/*_pipeline_vis.jpg")):
    print(f"\n📷 {os.path.basename(r)}")
    display(IPImage(r, width=1200))

print("\n" + "="*60)
print("🎯 SYSTEM BALANCE (RADAR CHART)")
for r in sorted(glob.glob("outputs/test_single_results/*_spider_chart.jpg")):
    print(f"\n📊 {os.path.basename(r)}")
    display(IPImage(r, width=600))

print("\n" + "="*60)
print("📈 SYSTEM SELF-AWARENESS (SCATTER PLOT)")
for r in sorted(glob.glob("outputs/test_single_results/*_scatter.jpg")):
    print(f"\n📊 {os.path.basename(r)}")
    display(IPImage(r, width=600))
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
    print(f"📊 mIoU: {m['mIoU']:.4f} | F1: {m['F1']:.4f} | Precision: {m['Precision']:.4f} | Recall: {m['Recall']:.4f}")
    print(f"📊 Detection Rate: {results['detection_rate']}%")

# Agent reasoning per image
with open("outputs/challenging_subset_prompts/agent5_final_synthesis_results.json") as f:
    agent5 = json.load(f)
    print(f"\n{'='*60}")
    print("🤖 AGENT REASONING")
    print(f"{'='*60}")
    for r in agent5.get("results", []):
        img = r.get("image", "")
        syn = r.get("synthesis_result", {})
        prompts = syn.get("grounded_sam_prompts", {})
        print(f"\n📸 {img}")
        print(f"   V1: {prompts.get('prompt_v1', 'N/A')} | V2: {prompts.get('prompt_v2', 'N/A')}")
        print(f"   Confidence: {syn.get('overall_confidence', 'N/A')} | Type: {syn.get('anomaly_type', 'N/A')}")
        print(f"   Reasoning: {syn.get('reasoning', 'N/A')[:200]}")
# Visualizations
print(f"\n{'='*60}")
print("🧠 PIPELINE REASONING & VISUALIZATION")
for r in sorted(glob.glob("outputs/evaluation_results/*_pipeline_vis.jpg")):
    print(f"\n📷 {os.path.basename(r)}")
    display(IPImage(r, width=1200))

print("\n" + "="*60)
print("🎯 SYSTEM BALANCE (RADAR CHART)")
for r in sorted(glob.glob("outputs/evaluation_results/*_spider_chart.jpg")):
    print(f"\n📊 {os.path.basename(r)}")
    display(IPImage(r, width=600))

print("\n" + "="*60)
print("📈 SYSTEM SELF-AWARENESS (SCATTER PLOT)")
for r in sorted(glob.glob("outputs/evaluation_results/*_scatter.jpg")):
    print(f"\n📊 {os.path.basename(r)}")
    display(IPImage(r, width=800))

print("\n" + "="*60)
print("🧮 SYSTEM CONFUSION Matrix")
from IPython.display import display, Image as IPImage
if os.path.exists("outputs/evaluation_results/batch_confusion_matrix.jpg"):
    display(IPImage("outputs/evaluation_results/batch_confusion_matrix.jpg", width=800))
```

---
## Phase 3: Web App (Optional — choose one)

### Option A — Advanced Gradio Dashboard (Recommended)
Launch the beautiful new Gradio UI that includes live visual analytics and reasoning logs!
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
print(f"🌐 Open Streamlit app: {url}")
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
Image → Agent 1 (Scene Context)     ─┐
      → Agent 2 (Spatial Anomaly)    ─┤
      → Agent 3 (Semantic Analysis)  ─┼→ Agent 5 (Synthesis) → prompt_v1, prompt_v2
      → Agent 4 (Visual Appearance)  ─┘           ↓
                                         GroundingDINO (detection)
                                              ↓
                                         CLIP (verification)
                                              ↓
                                         SAM (segmentation)
                                              ↓
                                         Evaluation (mIoU, F1)
```
