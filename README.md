# Multi-Agent Visual Reasoning for Out-of-Distribution Detection in Complex Road Environments
IEEE Access 2025

<!-- [Paper (IEEE Access)] | [ArXiv (TBD)] | [Project Page (TBD)]

---
-->

## Abstract

Out-of-Distribution (OOD) detection is critical for ensuring the reliability of semantic segmentation models in safety-critical autonomous driving scenarios. Despite recent advances, existing state-of-the-art OOD segmentation methods fundamentally rely on local features and suffer from a critical lack of contextual understanding in complex road environments. A representative example is distant scenes with small objects that require contextual reasoning to distinguish them from background elements. To evaluate such complex and challenging cases, we construct a dedicated subset for robustness assessment. The root cause stems from their inability to perform contextual semantic reasoning about object appropriateness in road environments.

To address these fundamental limitations, we propose a novel multi-agent visual reasoning framework that leverages the powerful contextual understanding and semantic reasoning capabilities of Vision-Language Models (VLMs). Our framework decomposes the OOD detection task into specialized subtasks handled by multiple expert agents. This approach fundamentally shifts from local pattern recognition to in-context understanding-based OOD detection, enabling the system to understand not just what is anomalous, but why it is inappropriate for the given road context. Extensive experiments demonstrate that our framework significantly outperforms existing methods, particularly in challenging scenarios, while providing interpretable reasoning for safety-critical applications.

---

## Architecture

<img width="758" height="437" alt="image" src="https://github.com/user-attachments/assets/99081251-5ae9-4672-bfea-5475f181b938" />


---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SongJeongHyo/MAVR-OOD.git
    cd MAVR-OOD

    ```

2.  **Create Environment & Install Dependencies:**
    (Python 3.10 recommended, CUDA 11.8+)
    ```bash
    pip install -r requirements.txt
    ```
    
    This will install:
    - **LLaVA-7B** dependencies (transformers, bitsandbytes, accelerate)
    - **CLIP** (OpenAI CLIP for semantic verification)
    - **GroundingDINO + SAM** dependencies

3.  **Download Model Weights:**
    Create a `weights/` directory and download the pre-trained model weights into it.
    ```bash
    mkdir weights
    
    # Download GroundingDINO weight
    wget -P weights/ [https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)
    
    # Download SAM weight
    wget -P weights/ [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
    ```

4.  **LLaVA-7B Model** 🧠
    The LLaVA-1.5-7B model will be automatically downloaded from HuggingFace on first run.
    It uses 4-bit quantization by default (~5-7GB VRAM). Configure in `.env`:
    ```env
    VLM_MODEL_NAME=llava-hf/llava-1.5-7b-hf
    VLM_USE_4BIT=true
    VLM_DEVICE=cuda
    ```
---

## 📊 Data Preparation

### 1. Download Datasets
Our framework is evaluated on standard OOD benchmarks. Please download the datasets from their official sites:
* [RoadAnomaly Dataset](https://www.epfl.ch/labs/cvlab/data/road-anomaly/)
* [Fishyscapes Dataset](https://fishyscapes.com/dataset)
* [SMIYC Dataset](https://segmentmeifyoucan.com/datasets)

Unzip them to a known location (e.g., `/home/user/my_datasets/`).

### 2. Reconstruct the Challenging Subset
To reproduce our results on the "Challenging Subset," run our reconstruction script. This script uses the IDs from `data/challenging_subset_ids.txt` to copy the corresponding files from your original RoadAnomaly dataset location.

**Important:** You must replace `/path/to/your/RoadAnomaly` with the actual path where you downloaded the dataset in Step 1.

```bash
python scripts/reconstruct_subset.py \
    --original_dir /path/to/your/RoadAnomaly \
    --id_file data/challenging_subset_ids.txt \
    --output_dir ./data/challenging_subset
```
This will create a new folder ./data/challenging_subset containing the images and labels for evaluation.

## 🚀 How to Reproduce Results
Our pipeline is a two-stage process:

Stage 1: Run the Multi-Agent framework to analyze images and generate OOD prompts.

Stage 2: Run the GroundedSAM evaluation script using these prompts to get segmentation masks and scores.

Stage 1: Generate Prompts (Multi-Agent Reasoning with LLaVA-7B)
Run run_all_agents.py to process all images. This runs Agents 1-5 using local LLaVA-7B and creates `agent5_final_synthesis_results.json`.

```Bash

python src/agents/run_all_agents.py \
    --image_dir ./data/challenging_subset/original \
    --output_dir ./outputs/challenging_subset_prompts \
    --delay 2
```
--delay: Sets the delay (in seconds) between inference calls (default 2s for local model).

Stage 2: Run Evaluation (Grounded Segmentation + CLIP Verification)
Once you have the prompt JSON file, run run_evaluation.py. CLIP verification is enabled by default to filter false positive detections.

```Bash

# Ensure your GroundingDINO/ and segment_anything/ folders are in the root
python run_evaluation.py \
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
--clip_threshold: CLIP semantic verification threshold (set to 0.0 to disable).
Results, logs, and visualizations will be saved in ./outputs/evaluation_results.

## Citation

```Bash

@article{song2025multi,
  title={Multi-Agent Visual Reasoning for Out-of-Distribution Detection in Complex Road Environments},
  author={Song, Jeonghyo and Yun, Kimin and Jo, Dae Ung and Kim, Jinyoung and Yoo, Youngjoon},
  journal={IEEE Access},
  volume={13},
  pages={188198--188216},
  year={2025},
  publisher={IEEE}
}

```
