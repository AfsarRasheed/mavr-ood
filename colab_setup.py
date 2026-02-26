#!/usr/bin/env python3
"""
MAVR-OOD: Google Colab Setup & Run Script
==========================================
Copy each section below into separate Colab cells.
Make sure to enable GPU: Runtime → Change runtime type → T4 GPU
"""

# ============================================================
# CELL 1: Upload and extract your project zip
# ============================================================
"""
# 📦 Upload your project (Option A: zip upload)
from google.colab import files
uploaded = files.upload()  # Upload MAVR-OOD.zip

!unzip -q MAVR-OOD.zip
%cd MAVR-OOD
"""

# ============================================================
# CELL 1 ALT: Clone from GitHub (Option B)
# ============================================================
"""
# 📦 Clone from GitHub
!git clone https://github.com/YOUR_USERNAME/MAVR-OOD.git
%cd MAVR-OOD
"""

# ============================================================
# CELL 2: Install dependencies
# ============================================================
"""
# 🔧 Install dependencies
!pip install -r requirements.txt

# Build GroundingDINO (requires compilation)
%cd GroundingDINO
!pip install -e .
%cd ..

# Verify GPU
import torch
print(f"✅ CUDA: {torch.cuda.is_available()}")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
"""

# ============================================================
# CELL 3: Download model weights
# ============================================================
"""
# 📦 Download Weights
import os
os.makedirs("weights", exist_ok=True)

# GroundingDINO weights
!wget -q -P weights/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
print("✅ GroundingDINO weights downloaded")

# SAM weights
!wget -q -P weights/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
print("✅ SAM weights downloaded")
"""

# ============================================================
# CELL 4: Run Stage 1 — Multi-Agent Analysis (~20-30 min)
# ============================================================
"""
# 🤖 Stage 1: Run all 5 Agents
!python src/agents/run_all_agents.py \
    --image_dir ./data/challenging_subset/original \
    --output_dir ./outputs/challenging_subset_prompts \
    --delay 2
"""

# ============================================================
# CELL 5: Run Stage 2 — Detection & Evaluation
# ============================================================
"""
# 🎯 Stage 2: GroundingDINO + CLIP + SAM → Evaluation
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
"""

# ============================================================
# CELL 6: Launch Gradio App
# ============================================================
"""
# 🌐 Launch Gradio (gets a public URL!)
import app
demo = app.build_app()
demo.launch(share=True)
"""

# ============================================================
# CELL 7 (OPTIONAL): View results inline
# ============================================================
"""
# 📸 View output images
from IPython.display import display, Image as IPImage
import glob, os

results = sorted(glob.glob("outputs/evaluation_results/*.jpg"))
for r in results[:6]:
    print(f"\n📷 {os.path.basename(r)}")
    display(IPImage(r, width=600))
"""
