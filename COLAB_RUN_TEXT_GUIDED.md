# Text-Guided Detection: Google Colab Guide
> Multi-agent text-guided object detection with step-by-step visualization on Google Colab (T4 GPU)
>
> **Models:** LLaVA-7B (2 agents) + GroundingDINO + CLIP + SAM | **Runtime:** T4 GPU

---

## Phase 1: Setup (Same as MAVR-OOD)

> If you already ran `COLAB_RUN.md` Phase 1 in this session, skip to Phase 2.

### Cell 1 -- Clone Repository
```python
# First time: clone the repo
!git clone https://github.com/AfsarRasheed/mavr-ood.git
%cd mavr-ood

# Returning session: just pull latest
# %cd mavr-ood
# !git checkout -- .
# !git pull origin main
```

### Cell 2 -- Install Dependencies
```python
!pip install -q -r requirements.txt
!pip install -q gradio>=4.0.0 addict yapf bitsandbytes>=0.41.0
!pip install -q -e segment_anything/
!cd GroundingDINO && pip install -q -e . && cd ..
!pip install git+https://github.com/openai/CLIP.git -q

import torch, bitsandbytes
print(f"[OK] torch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
print(f"[OK] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("[OK] All dependencies installed!")
```

### Cell 3 -- Download Weights
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

## Phase 2: Text-Guided Detection

### Cell 4 -- Setup Image and Prompt
```python
import os
# Use any image from the dataset
TEST_IMAGE = "./data/challenging_subset/original/animals03_Zebras_in_the_road.jpg"
USER_PROMPT = "the zebra"  # <<< CHANGE THIS to any prompt

# Or upload your own image:
# from google.colab import files
# uploaded = files.upload()
# TEST_IMAGE = list(uploaded.keys())[0]

print(f"[OK] Image: {TEST_IMAGE}")
print(f"[OK] Prompt: {USER_PROMPT}")
```

### Cell 5 -- Run Detection Pipeline
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

# ---- Phase 1: Run LLaVA agents FIRST (before detection models) ----
gc.collect()
torch.cuda.empty_cache()

from src.text_guided.scene_agent import scene_understanding
from src.text_guided.attribute_agent import attribute_matching_agent

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

# ---- Phase 2: Load detection models and run pipeline ----
print("\n>> Phase 2: Running detection pipeline...")
from src.model_loader import load_gdino_model, load_sam_predictor, load_clip_verifier
from src.text_guided import run_text_guided_pipeline

gdino = load_gdino_model()
sam = load_sam_predictor()
clip_v = load_clip_verifier()

results = run_text_guided_pipeline(
    image_np=image_np,
    user_prompt=USER_PROMPT,
    image_path=TEST_IMAGE,
    gdino_model=gdino,
    sam_predictor=sam,
    clip_verifier=clip_v,
    box_threshold=0.35,
    clip_threshold=0.25,
    precomputed_scene=scene_result,
    precomputed_attr=attr_result,
)

print("\n[OK] Text-guided detection complete!")
```

### Cell 6 -- View Step-by-Step Results
```python
from IPython.display import display, Image as IPImage
import matplotlib.pyplot as plt

step_images = results.get("step_images", {})
step_names = [
    ("step1_scene", "Step 1: Scene Understanding Agent (LLaVA)"),
    ("step2_query", "Step 2: Attribute Matching Agent (LLaVA)"),
    ("step3_candidates", "Step 3: Candidate Detection (GroundingDINO)"),
    ("step4_clip", "Step 4: CLIP Verification"),
    ("step5_spatial", "Step 5: Spatial Selection"),
    ("step6_final", "Step 6: Final Segmentation (SAM)"),
]

!mkdir -p outputs/text_guided

for key, title in step_names:
    img = step_images.get(key)
    if img is not None:
        plt.figure(figsize=(14, 8))
        plt.imshow(img)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"outputs/text_guided/{key}.jpg", dpi=150, bbox_inches='tight')
        plt.show()
        print(f"[OK] Saved: outputs/text_guided/{key}.jpg\n")

# Print summary log
print("\n" + results.get("summary", ""))
```

---

## Phase 3: Try Different Prompts

### Cell 7 -- More Examples
```python
# Change prompt and re-run (models already loaded, only LLaVA needs reload)
test_cases = [
    ("./data/challenging_subset/original/animals03_Zebras_in_the_road.jpg", "the largest zebra"),
    ("./data/challenging_subset/original/animals01_Horse_in_the_road.jpg", "the horse"),
]

for img_path, prompt in test_cases:
    img_np = np.array(Image.open(img_path).convert("RGB"))

    # Run LLaVA agents
    gc.collect()
    torch.cuda.empty_cache()
    scene = scene_understanding(img_path)
    attr = attribute_matching_agent(img_path, scene, prompt)

    # Free LLaVA
    if hasattr(vlm_mod, '_model') and vlm_mod._model is not None:
        del vlm_mod._model
        vlm_mod._model = None
    if hasattr(vlm_mod, '_processor') and vlm_mod._processor is not None:
        del vlm_mod._processor
        vlm_mod._processor = None
    gc.collect()
    torch.cuda.empty_cache()

    # Run detection
    res = run_text_guided_pipeline(
        image_np=img_np, user_prompt=prompt, image_path=img_path,
        gdino_model=gdino, sam_predictor=sam, clip_verifier=clip_v,
        precomputed_scene=scene, precomputed_attr=attr,
    )

    final = res["step_images"].get("step6_final")
    if final is not None:
        plt.figure(figsize=(10, 6))
        plt.imshow(final)
        plt.title(f'"{prompt}"', fontsize=14)
        plt.axis('off')
        plt.show()
    print(f'[OK] "{prompt}" -- done\n')
```

---

## Phase 4: Launch Streamlit UI

### Cell 8 -- Run Streamlit App
> Full interactive UI with both Text-Guided and OOD Detection tabs.

**Option A: Using ngrok (recommended, faster)**
```python
# Install
!pip install pyngrok -q
!pip install git+https://github.com/openai/CLIP.git -q

# Set your ngrok auth token (free signup at https://dashboard.ngrok.com/get-started/your-authtoken)
!ngrok authtoken YOUR_TOKEN_HERE  # << paste your token

# Kill any old processes
!pkill -f streamlit 2>/dev/null

# Start Streamlit
!nohup streamlit run streamlit_app.py --server.port 8501 --server.headless true &

# Create tunnel
import time; time.sleep(8)
from pyngrok import ngrok
ngrok.kill()
url = ngrok.connect(8501)
print(f"\n🔗 Open this URL in your browser:\n{url}\n")
```

**Option B: Using localtunnel (no signup needed)**
```python
# Install
!pip install git+https://github.com/openai/CLIP.git -q
!npm install -g localtunnel

# Kill any old processes
!pkill -f streamlit 2>/dev/null

# Start Streamlit
!nohup streamlit run streamlit_app.py --server.port 8501 --server.headless true &

# Create tunnel
import time; time.sleep(8)
!lt --port 8501
```
> When localtunnel asks for a password, run this to get it:
> ```python
> import urllib.request
> print(urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip())
> ```

Then open the URL and use the **Text-Guided Detection** or **OOD Detection** tab.

---

## Phase 5: Evaluate on Dataset

### Cell 9 -- Run Evaluation (13 Images)
> Runs the VLM pipeline on all 13 test images and computes IoU/F1 metrics against ground truth masks.

```python
!python run_evaluate_vlm.py
```

### Cell 10 -- View Results
```python
from IPython.display import display, Image as IPImage
import glob, json, os

# Show comparison images
for img_path in sorted(glob.glob("outputs/vlm_evaluation/*_comparison.jpg")):
    print(f"\n{os.path.basename(img_path)}")
    display(IPImage(img_path, width=800))

# Show summary chart
if os.path.exists("outputs/vlm_evaluation/evaluation_summary.jpg"):
    display(IPImage("outputs/vlm_evaluation/evaluation_summary.jpg", width=800))

# Print metrics
with open("outputs/vlm_evaluation/evaluation_results.json") as f:
    results = json.load(f)
print(f"\n📊 Average IoU: {results['summary']['avg_iou']:.4f}")
print(f"📊 Average F1:  {results['summary']['avg_f1']:.4f}")
print(f"📊 Success Rate: {results['summary']['n_success']}/{results['summary']['n_images']}")
```

---

## Phase 6: Evaluate on Custom Images (Optional)

> Use your own images with manually annotated masks.

### Step 1: Create Masks
- Use [makesense.ai](https://www.makesense.ai/) or MS Paint
- Paint the **target object white**, everything else **black**
- Save as `.png` with the same name as your image

### Cell 11 -- Upload Custom Images + Masks
```python
import os
os.makedirs("data/custom_eval/original", exist_ok=True)
os.makedirs("data/custom_eval/labels", exist_ok=True)

from google.colab import files

print("📷 Upload IMAGES (jpg/png):")
uploaded_imgs = files.upload()
for name in uploaded_imgs:
    with open(f"data/custom_eval/original/{name}", "wb") as f:
        f.write(uploaded_imgs[name])
    print(f"  ✅ {name}")

print("\n🏷️ Upload MASK PNGs (same names, .png extension):")
uploaded_masks = files.upload()
for name in uploaded_masks:
    with open(f"data/custom_eval/labels/{name}", "wb") as f:
        f.write(uploaded_masks[name])
    print(f"  ✅ {name}")
```

### Cell 12 -- Add Queries and Run
```python
# ---- Add your queries here ----
queries = {
    "highway_scene.jpg": "the white car on the right",
    # "city_road.jpg": "the red truck",
    # "parking.jpg": "the person next to the blue car",
}

# Patch query map
with open("run_evaluate_vlm.py", "r") as f:
    code = f.read()
new_entries = "\n".join([f'    "{k}": "{v}",' for k, v in queries.items()])
code = code.replace(
    '"animals26_Unnamed_Road_Kazakhstan.jpg": "the cow",\n}',
    f'"animals26_Unnamed_Road_Kazakhstan.jpg": "the cow",\n{new_entries}\n}}'
)
with open("run_evaluate_vlm.py", "w") as f:
    f.write(code)

print("✅ Queries added!")
for k, v in queries.items():
    print(f"  {k} → \"{v}\"")

# Run evaluation
!python run_evaluate_vlm.py --data-dir ./data/custom_eval
```

### Cell 13 -- View Custom Results
```python
from IPython.display import display, Image as IPImage
import glob, json, os

for img_path in sorted(glob.glob("outputs/vlm_evaluation/*_comparison.jpg")):
    print(f"\n{os.path.basename(img_path)}")
    display(IPImage(img_path, width=800))

if os.path.exists("outputs/vlm_evaluation/evaluation_summary.jpg"):
    display(IPImage("outputs/vlm_evaluation/evaluation_summary.jpg", width=800))

with open("outputs/vlm_evaluation/evaluation_results.json") as f:
    results = json.load(f)
print(f"\n📊 Average IoU: {results['summary']['avg_iou']:.4f}")
print(f"📊 Average F1:  {results['summary']['avg_f1']:.4f}")
```
