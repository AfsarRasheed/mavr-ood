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
    ("step1_scene", "Step 1: Scene Agent (LLaVA)"),
    ("step2_query", "Step 2: Attribute Agent (LLaVA)"),
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
!mkdir -p outputs
plt.savefig("outputs/text_guided_pipeline.jpg", dpi=150, bbox_inches='tight')
plt.show()

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

## Phase 4: Launch Gradio UI (Optional)

### Cell 8 -- Run Gradio App
```python
import app
demo = app.build_app()
demo.launch(share=True)
```
Then open the shared link and use the **Text-Guided Detection** tab (Tab 1).
