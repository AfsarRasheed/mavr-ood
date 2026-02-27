# MAVR-OOD: Google Colab Implementation Guide
> Complete step-by-step cells to run the full pipeline on Google Colab (T4 GPU)

---

## Phase 1: Setup (Run once)

### Cell 1 — Clone Repository
```python
!git clone https://github.com/AfsarRasheed/mavr-ood.git
%cd mavr-ood
```

### Cell 2 — Install Dependencies + Fix Compatibility
```python
!pip install -r requirements.txt
!pip install gradio>=4.0.0
!python fix_colab_compat.py
```

### Cell 3 — Download Weights (~3 min)
```python
!mkdir -p weights
!wget -q -P weights/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
!wget -q -P weights/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
print("✅ Weights downloaded!")
```

### Cell 4 — Verify GPU
```python
import torch
print(f"✅ CUDA: {torch.cuda.is_available()}")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

---

## Phase 2A: Single Image Test (Quick validation ~5 min)

### Cell 5A — Prepare Single Image Test
```python
!mkdir -p ./data/test_single/original
!mkdir -p ./data/test_single/labels
!cp ./data/challenging_subset/original/animals03_Zebras_in_the_road.jpg ./data/test_single/original/
!cp ./data/challenging_subset/labels/animals03_Zebras_in_the_road.png ./data/test_single/labels/
print("✅ Test folder ready (1 image)")
```

### Cell 6A — Run All Agents on 1 Image (~5 min)
```python
!python src/agents/run_all_agents.py \
    --image_dir ./data/test_single/original \
    --output_dir ./outputs/test_single_prompts \
    --delay 2
```

### Cell 7A — Evaluate Single Image
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

### Cell 8A — View Single Image Result
```python
from IPython.display import display, Image as IPImage
import glob, os

results = sorted(glob.glob("outputs/test_single_results/*.jpg"))
print(f"📊 Found {len(results)} result images\n")
for r in results:
    print(f"📷 {os.path.basename(r)}")
    display(IPImage(r, width=600))
```

---

## Phase 2B: Full Dataset Run (All 13 images ~35-40 min)

### Cell 5B — Run All Agents on Full Dataset (~30 min)
```python
!python src/agents/run_all_agents.py \
    --image_dir ./data/challenging_subset/original \
    --output_dir ./outputs/challenging_subset_prompts \
    --delay 2
```

### Cell 6B — Evaluate Full Dataset
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

### Cell 7B — View All Results
```python
from IPython.display import display, Image as IPImage
import glob, os

results = sorted(glob.glob("outputs/evaluation_results/*.jpg"))
print(f"📊 Found {len(results)} result images\n")
for r in results:
    print(f"\n📷 {os.path.basename(r)}")
    display(IPImage(r, width=600))
```

---

## Phase 2C: Full Road Anomaly Dataset (All 21 images ~50-60 min)

> Use this if you want to evaluate on the complete dataset instead of just the 13-image challenging subset.

### Cell 5C — Run All Agents on Full Road Anomaly Dataset
```python
!python src/agents/run_all_agents.py \
    --image_dir ./datasets/RoadAnomaly/original \
    --output_dir ./outputs/road_anomaly_full_prompts \
    --delay 2
```

### Cell 6C — Evaluate Full Road Anomaly Dataset
```python
!python run_evaluate.py \
    --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint weights/groundingdino_swint_ogc.pth \
    --sam_checkpoint weights/sam_vit_h_4b8939.pth \
    --dataset_dir ./datasets/RoadAnomaly \
    --dataset_type road_anomaly \
    --multiagent_prompts ./outputs/road_anomaly_full_prompts/agent5_final_synthesis_results.json \
    --output_dir ./outputs/road_anomaly_full_results \
    --clip_threshold 0.20 \
    --device cuda
```

### Cell 7C — View Full Dataset Results
```python
from IPython.display import display, Image as IPImage
import glob, os

results = sorted(glob.glob("outputs/road_anomaly_full_results/*.jpg"))
print(f"📊 Found {len(results)} result images\n")
for r in results:
    print(f"\n📷 {os.path.basename(r)}")
    display(IPImage(r, width=600))
```

---

## Phase 3: Gradio App (Optional)

### Cell 9 — Launch Gradio App
```python
import app
demo = app.build_app()
demo.launch(share=True)
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| CUDA OOM | Already fixed — 4-bit quantization enabled automatically |
| `No module named 'addict'` | Run `!python fix_colab_compat.py` again |
| `cannot import sam_model_registry` | Run `!python fix_colab_compat.py` again |
| `get_head_mask` AttributeError | Run `!python fix_colab_compat.py` again |
| Session disconnects | Re-run from Cell 1, weights will be re-downloaded |
