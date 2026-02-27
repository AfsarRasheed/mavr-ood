# MAVR-OOD: Multi-Agent Vision-Language Reasoning for Reliable Out-of-Distribution Object Localization in Road Environments


## Abstract

Out-of-Distribution (OOD) detection is critical for ensuring the reliability of semantic segmentation models in safety-critical autonomous driving scenarios. Despite recent advances, existing state-of-the-art OOD segmentation methods fundamentally rely on local features and suffer from a critical lack of contextual understanding in complex road environments. A representative example is distant scenes with small objects that require contextual reasoning to distinguish them from background elements. To evaluate such complex and challenging cases, we construct a dedicated subset for robustness assessment. The root cause stems from their inability to perform contextual semantic reasoning about object appropriateness in road environments.

To address these fundamental limitations, we propose a novel multi-agent visual reasoning framework that leverages the powerful contextual understanding and semantic reasoning capabilities of Vision-Language Models (VLMs). Our framework decomposes the OOD detection task into specialized subtasks handled by multiple expert agents. This approach fundamentally shifts from local pattern recognition to in-context understanding-based OOD detection, enabling the system to understand not just what is anomalous, but why it is inappropriate for the given road context. Extensive experiments demonstrate that our framework significantly outperforms existing methods, particularly in challenging scenarios, while providing interpretable reasoning for safety-critical applications.

---

## Architecture

<img width="758" height="437" alt="image" src="https://github.com/user-attachments/assets/99081251-5ae9-4672-bfea-5475f181b938" />

### Pipeline Overview

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

### Multi-Agent System

| Agent | Role | Output |
|-------|------|--------|
| Agent 1 | Scene Context Analyzer | Scene type, environmental conditions, normality criteria |
| Agent 2 | Spatial Anomaly Detector | Positioning violations, traffic flow disruptions |
| Agent 3 | Semantic Inconsistency Analyzer | Domain violations, safety implications |
| Agent 4 | Visual Appearance Evaluator | Color, texture, shape anomalies |
| Agent 5 | Reasoning Synthesizer | Final `prompt_v1` and `prompt_v2` for GroundedSAM |

---

## Results

| Dataset | mIoU | F1 | Precision | Recall | Detection Rate |
|---------|------|-----|-----------|--------|----------------|
| Test Image (Zebras) | 0.9741 | 0.9869 | 0.9904 | 0.9834 | 100% |

---

## 🛠️ Installation

### Local Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/AfsarRasheed/mavr-ood.git
    cd mavr-ood
    ```

2. **Install Dependencies** (Python 3.10+, CUDA 11.8+):
    ```bash
    pip install -r requirements.txt
    pip install -e segment_anything/
    cd GroundingDINO && pip install -e . && cd ..
    ```

3. **Download Model Weights:**
    ```bash
    mkdir -p weights
    wget -P weights/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    wget -P weights/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    ```

4. **LLaVA-7B Model** — Automatically downloaded from HuggingFace on first run. Uses 4-bit quantization (~4GB VRAM).

### Google Colab Setup

> See [COLAB_RUN.md](COLAB_RUN.md) for complete step-by-step Colab instructions.

```python
!git clone https://github.com/AfsarRasheed/mavr-ood.git
%cd mavr-ood
!pip install -q -r requirements.txt
!pip install -q gradio>=4.0.0 addict yapf bitsandbytes>=0.41.0
!pip install -q -e segment_anything/
!cd GroundingDINO && pip install -q -e . && cd ..
```

---

## 📊 Data Preparation

### 1. Download Datasets
Our framework is evaluated on standard OOD benchmarks:
* [RoadAnomaly Dataset](https://www.epfl.ch/labs/cvlab/data/road-anomaly/)
* [Fishyscapes Dataset](https://fishyscapes.com/dataset)
* [SMIYC Dataset](https://segmentmeifyoucan.com/datasets)

### 2. Reconstruct the Challenging Subset
```bash
python scripts/reconstruct_subset.py \
    --original_dir /path/to/your/RoadAnomaly \
    --id_file data/challenging_subset_ids.txt \
    --output_dir ./data/challenging_subset
```

---

## 🚀 How to Reproduce Results

Our pipeline is a two-stage process:

### Stage 1: Generate Prompts (Multi-Agent Reasoning with LLaVA-7B)

Run `run_all_agents.py` to process all images. This runs Agents 1–5 using local LLaVA-7B and creates `agent5_final_synthesis_results.json`.

```bash
python src/agents/run_all_agents.py \
    --image_dir ./data/challenging_subset/original \
    --output_dir ./outputs/challenging_subset_prompts \
    --delay 2
```

- `--delay`: Delay (seconds) between inference calls (default: 2s)

### Stage 2: Run Evaluation (Grounded Segmentation + CLIP Verification)

Once you have the prompt JSON file, run `run_evaluate.py`. CLIP verification is enabled by default to filter false positive detections.

```bash
python run_evaluate.py \
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

- `--clip_threshold`: CLIP semantic verification threshold (set to `0.0` to disable)

Results, logs, and visualizations will be saved in `./outputs/evaluation_results`.

---

## 🌐 Gradio Demo

Launch the interactive web app for real-time OOD detection:

```bash
python app.py
```

On Google Colab:
```python
import app
demo = app.build_app()
demo.launch(share=True)
```

Features:
- **Single Image**: Upload any road scene and detect anomalous objects
- **Batch Dataset**: Run full pipeline on a dataset folder
- Adjustable CLIP and Box thresholds
- Visualizations: Bounding boxes, SAM masks, Final OOD mask
- Full agent analysis breakdown

---

## Project Structure

```
mavr-ood/
├── src/
│   ├── agents/
│   │   ├── agent1.py          # Scene Context Analyzer
│   │   ├── agent2.py          # Spatial Anomaly Detector
│   │   ├── agent3.py          # Semantic Inconsistency Analyzer
│   │   ├── agent4.py          # Visual Appearance Evaluator
│   │   ├── agent5.py          # Reasoning Synthesizer
│   │   ├── run_all_agents.py  # Sequential agent runner
│   │   └── vlm_backend.py     # LLaVA-7B inference backend
│   └── clip_verifier.py       # CLIP semantic verification
├── GroundingDINO/              # GroundingDINO (object detection)
├── segment_anything/           # SAM (segmentation)
├── run_evaluate.py             # Evaluation pipeline
├── app.py                      # Gradio web interface
├── dataset.py                  # Dataset loaders
├── data/
│   └── challenging_subset/     # Test images and labels
├── weights/                    # Model weights (downloaded)
├── COLAB_RUN.md                # Google Colab guide
└── requirements.txt
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{mavr-ood2026,
  title={MAVR-OOD: Multi-Agent Vision-Language Reasoning for Reliable Out-of-Distribution Object Localization in Road Environments},
  author={Afsar Rasheed},
  year={2026}
}
```
