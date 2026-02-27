# MAVR-OOD: Multi-Agent Vision-Language Reasoning for Out-of-Distribution Object Detection

## Complete Project Documentation

---

## 1. Project Overview

### 1.1 What is MAVR-OOD?

MAVR-OOD (Multi-Agent Visual Reasoning for Out-of-Distribution Object Detection) is a **confidence-aware multi-agent vision-language system** designed for **reliable object localization** in road environments. The system identifies **anomalous/out-of-distribution (OOD) objects** — objects that do not belong in road scenes — such as animals on highways, fallen debris, or misplaced obstacles.

### 1.2 Problem Statement

Autonomous driving systems and road safety applications need to detect objects that are **unexpected** or **out-of-place** in road environments. Traditional object detection models are trained on fixed categories (cars, pedestrians, traffic signs) and fail to recognize novel or anomalous objects. MAVR-OOD addresses this by using **multi-agent reasoning with vision-language models (VLMs)** to identify and describe anomalies, then ground them using detection and segmentation models.

### 1.3 Key Innovation

Unlike single-model approaches, MAVR-OOD employs a **5-agent collaborative framework** where each agent specializes in a different aspect of anomaly analysis:
- Contextual understanding
- Spatial rule violation detection
- Semantic appropriateness evaluation
- Visual appearance analysis
- Multi-source reasoning synthesis

This multi-perspective approach mirrors how human experts analyze road scenes — from multiple angles before reaching a conclusion.

---

## 2. System Architecture

### 2.1 High-Level Pipeline

```
Input Image
    │
    ▼
┌─────────────────────────────────────────────┐
│         PHASE 1: Multi-Agent VLM Analysis    │
│                                              │
│  Agent 1: Scene Context Analyzer             │
│  Agent 2: Spatial Anomaly Detector           │
│  Agent 3: Semantic Inconsistency Analyzer    │
│  Agent 4: Visual Appearance Evaluator        │
│  Agent 5: Reasoning Synthesizer              │
│                                              │
│  Output: Optimized text prompts (V1, V2)     │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│         PHASE 2: Grounded Detection          │
│                                              │
│  GroundingDINO: Open-vocabulary detection    │
│  CLIP Verifier: Semantic verification        │
│  SAM: Pixel-precise segmentation             │
│                                              │
│  Output: Segmentation masks + metrics        │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│         PHASE 3: Evaluation                  │
│                                              │
│  IoU, F1, AUROC, FPR@95                      │
│  Per-image threshold optimization            │
│  Visualization generation                    │
└─────────────────────────────────────────────┘
```

### 2.2 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| VLM Backend | LLaVA-1.5-7B (HuggingFace) | Vision-language reasoning |
| Quantization | bitsandbytes (4-bit NF4) | VRAM optimization (~4GB) |
| Object Detection | GroundingDINO (SwinT-OGC) | Open-vocabulary grounding |
| Semantic Verification | OpenAI CLIP (ViT-B/32) | Detection filtering |
| Segmentation | SAM ViT-H (Segment Anything) | Pixel-precise masks |
| Frontend | Gradio | Interactive web demo |
| Compute | Google Colab (T4 GPU) | Cloud inference |

---

## 3. Module-by-Module Description

### 3.1 VLM Backend (`src/agents/vlm_backend.py`)

**Purpose**: Provides a unified inference interface for all 5 agents using LLaVA-1.5-7B.

**Technical Details**:
- **Model**: `llava-hf/llava-1.5-7b-hf` — a 7-billion parameter multimodal model combining a CLIP vision encoder with a Vicuna-7B language model
- **Quantization**: Automatic 4-bit NF4 quantization using `bitsandbytes` on Linux/Colab, reducing VRAM from ~14GB to ~4GB
- **Fallback**: FP16 with `device_map="auto"` on Windows for GPU/CPU split
- **Singleton Pattern**: Model loaded once, reused across all agents within a process
- **Memory Management**: `gc.collect()` and `torch.cuda.empty_cache()` after each inference
- **Input Modes**: 
  - Vision + Language (image + text prompt) for Agents 1-4
  - Text-only (no image) for Agent 5
- **Prompt Format**: Uses LLaVA chat format: `USER: <image>\n{prompt}\nASSISTANT:`
- **Generation Parameters**: temperature=0.2, top_p=0.9, max_new_tokens=4096

**Key Function**: `run_vlm(messages, image_path=None) → str`

---

### 3.2 Agent 1: Scene Context Analyzer (`src/agents/agent1.py`)

**Purpose**: Establishes contextual baselines — what is "normal" for the given road environment.

**Responsibilities**:
1. Determine scene type (urban, rural, highway, intersection, residential)
2. Assess environmental conditions (weather, lighting, time of day)
3. Identify expected object inventory based on context
4. Establish context-dependent normality criteria

**Output Schema**:
```json
{
    "scene_analysis": {
        "scene_type": "",
        "road_infrastructure": "",
        "environmental_conditions": {
            "weather": "",
            "lighting": "",
            "time_period": ""
        }
    },
    "contextual_baseline": {
        "expected_objects": [],
        "expected_behaviors": [],
        "infrastructure_elements": [],
        "typical_layout": ""
    },
    "normality_criteria": {
        "object_appropriateness": "",
        "spatial_expectations": "",
        "behavioral_norms": ""
    },
    "context_confidence": 0.0
}
```

**Why This Agent Matters**: Provides the baseline understanding against which other agents compare their findings. A zebra on a Kenyan road near a wildlife reserve may be normal, but the same zebra on a German highway is anomalous — context determines this.

---

### 3.3 Agent 2: Spatial Anomaly Detector (`src/agents/agent2.py`)

**Purpose**: Identifies objects that violate spatial positioning rules and traffic flow conventions.

**Responsibilities**:
1. Evaluate object positioning relative to road infrastructure
2. Assess traffic flow disruptions
3. Analyze scale consistency based on apparent distance
4. Identify unusual clustering or density patterns

**Output Schema**:
```json
{
    "spatial_analysis": {
        "observed_objects": [],
        "object_positions": {
            "on_roadway": [],
            "roadside": [],
            "infrastructure": []
        }
    },
    "positioning_violations": [],
    "scale_inconsistencies": [],
    "clustering_anomalies": [],
    "traffic_flow_analysis": {
        "disruption_points": [],
        "accessibility_issues": [],
        "safety_hazards": []
    },
    "spatial_confidence": 0.0
}
```

**Why This Agent Matters**: An object might be semantically appropriate (e.g., a cow) but spatially anomalous (standing in the middle of a highway, blocking traffic). This agent captures spatial rule violations specifically.

---

### 3.4 Agent 3: Semantic Inconsistency Analyzer (`src/agents/agent3.py`)

**Purpose**: Evaluates domain appropriateness — whether objects semantically belong in road environments.

**Responsibilities**:
1. Evaluate whether objects belong in road environments
2. Assess safety considerations using common-sense reasoning
3. Apply traffic regulation domain knowledge
4. Identify objects normal elsewhere but inappropriate on roads

**Output Schema**:
```json
{
    "semantic_analysis": {
        "detected_objects": [],
        "object_categorization": {
            "road_appropriate": [],
            "questionable": [],
            "inappropriate": []
        }
    },
    "domain_violations": [],
    "safety_implications": {
        "immediate_hazards": [],
        "regulatory_violations": [],
        "functional_conflicts": []
    },
    "semantic_reasoning": {
        "overall_assessment": "",
        "primary_concerns": [],
        "context_considerations": ""
    },
    "semantic_confidence": 0.0
}
```

**Why This Agent Matters**: Focuses on "does this object make sense here?" — a semantic judgment that requires world knowledge. A wild boar is a normal animal, but it's semantically out-of-place on an urban road.

---

### 3.5 Agent 4: Visual Appearance Evaluator (`src/agents/agent4.py`)

**Purpose**: Detects condition-based visual anomalies — color, texture, shape, and material irregularities.

**Responsibilities**:
1. Analyze color inconsistencies
2. Examine texture and material irregularities
3. Identify shape deformations
4. Detect condition-based hazards (damaged infrastructure, unusual markings)

**Output Schema**:
```json
{
    "visual_analysis": {
        "lighting_conditions": "",
        "overall_visual_quality": "",
        "detected_objects": []
    },
    "color_anomalies": [],
    "texture_irregularities": [],
    "shape_deformations": [],
    "material_condition_issues": [],
    "visual_integrity_assessment": {
        "overall_condition": "",
        "primary_visual_concerns": [],
        "hazard_indicators": []
    },
    "visual_confidence": 0.0
}
```

**Why This Agent Matters**: Catches anomalies the other agents might miss — damaged road surfaces, unusual debris textures, or objects that look visually foreign to the road environment.

---

### 3.6 Agent 5: Reasoning Synthesizer (`src/agents/agent5.py`)

**Purpose**: Integrates all 4 agent findings into a final judgment and generates optimized detection prompts.

**Key Design**:
- **Text-only mode**: Does NOT receive the image; instead synthesizes from agent outputs
- Applies priority rules: Animals > Misplaced vehicles > Obstacles > Others
- Generates exactly 2 prompt variants for GroundingDINO:
  - **V1**: `adjective + noun` (e.g., "wild zebra")
  - **V2**: `single noun` (e.g., "zebra")
- Selects the TOP 1 most anomalous object

**Input**: JSON outputs from Agents 1-4
**Output**: Optimized GroundedSAM text prompts + confidence scores

**Why This Agent Matters**: Acts as the "decision maker" — the agent that weighs all perspectives and produces the final actionable output. Without synthesis, the system would have 4 separate opinions but no unified decision.

---

### 3.7 CLIP Semantic Verifier (`src/clip_verifier.py`)

**Purpose**: Post-detection verification layer that filters false positive detections from GroundingDINO.

**Technical Details**:
- **Model**: OpenAI CLIP ViT-B/32 (~1GB)
- **Method**: Crops each detected bounding box, encodes it with CLIP's vision encoder, and computes cosine similarity with the text prompt
- **Threshold**: Default 0.20 — detections below this similarity score are filtered out
- **Role**: Sits between GroundingDINO detection and SAM segmentation

**Key Function**: `verify_detections(image, boxes, phrases, text_prompt) → filtered results`

**Why This Module Matters**: GroundingDINO can sometimes detect non-matching regions (false positives). CLIP acts as a semantic "second opinion" — if the cropped region doesn't actually look like the described anomaly, it's filtered out.

---

### 3.8 GroundingDINO (Open-Vocabulary Detector)

**Purpose**: Detects objects in images based on free-form text descriptions — no fixed class vocabulary.

**Technical Details**:
- **Architecture**: DINO detector + BERT text encoder + cross-attention fusion
- **Backbone**: Swin Transformer (SwinT-OGC variant)
- **Input**: Image + text prompt (e.g., "wild zebra" or "fallen debris")
- **Output**: Bounding boxes with confidence scores
- **Key Advantage**: Can detect ANY object described in natural language, unlike traditional detectors limited to pre-trained categories

**Configuration**:
- Box threshold: 0.3 (minimum detection confidence)
- Text threshold: 0.25 (minimum text-logit match)
- Per-image threshold optimization available

---

### 3.9 SAM — Segment Anything Model

**Purpose**: Generates pixel-precise segmentation masks from bounding box prompts.

**Technical Details**:
- **Architecture**: ViT-H (Vision Transformer - Huge) image encoder + prompt encoder + mask decoder
- **Variant**: SAM ViT-H (most accurate, 2.5GB checkpoint)
- **Input**: Image + bounding boxes from GroundingDINO
- **Output**: Binary segmentation masks at full image resolution

**Why SAM**: Unlike semantic segmentation models that require training per class, SAM is a foundation model that can segment ANY object given a spatial prompt (point, box, or mask). This makes it ideal for OOD detection where the target classes are unknown at training time.

---

### 3.10 Dataset Handler (`dataset.py`)

**Purpose**: Unified dataset loading for multiple anomaly detection benchmarks.

**Supported Datasets**:

| Dataset | Mask Format | Description |
|---------|------------|-------------|
| Road Anomaly | Binary (0/1) | Animals, objects on roads |
| Fishyscapes | Multi-value (0=bg, 1=OOD, 255=known) | Lost & Found, Static variants |
| Segment Me | RGB color-coded (orange = anomaly) | AnomalyTrack, ObstacleTrack |

**Architecture**: Factory pattern with `DatasetFactory.create_dataset()` and abstract base class `BaseAnomalyDataset`.

---

### 3.11 Evaluation Pipeline (`run_evaluate.py`)

**Purpose**: End-to-end evaluation combining detection, verification, segmentation, and metrics computation.

**Pipeline Flow**:
1. Load GroundingDINO model + SAM predictor + CLIP verifier
2. Load dataset with ground truth masks
3. For each image:
   - Load multi-agent prompts (V1, V2) from Agent 5 output
   - Run GroundingDINO detection with V1 prompt
   - CLIP-verify detections (filter false positives)
   - Run SAM segmentation on verified boxes
   - If V1 fails, fallback to V2 prompt
   - Optimize thresholds per-image for best IoU
4. Compute aggregate metrics

**Evaluation Metrics**:
- **IoU** (Intersection over Union): Overlap between predicted and ground truth masks
- **F1 Score**: Harmonic mean of precision and recall
- **AUROC**: Area Under ROC Curve — overall discrimination ability
- **FPR@95**: False Positive Rate at 95% True Positive Rate
- **AUUPRC**: Area Under Precision-Recall Curve

**Threshold Optimization**: Tests combinations of box_threshold (0.15–0.50) and text_threshold (0.15–0.40) to find optimal per-image settings.

---

### 3.12 Gradio Web Application (`app.py`)

**Purpose**: Interactive web interface for single-image and batch OOD detection.

**Features**:
- **Tab 1: Single Image** — Upload one image, run full pipeline (agents → detection → segmentation), view results
- **Tab 2: Batch Dataset** — Run pipeline on entire dataset folder
- **Visualizations**: Detection boxes, segmentation masks, binary mask overlay
- **Deployment**: `share=True` creates public URL accessible from any browser

---

## 4. Multi-Agent Workflow (Detailed Data Flow)

### Step-by-Step Execution:

```
Step 1: run_all_agents.py orchestrates sequential execution
    │
    ├── Agent 1: Reads image → LLaVA analyzes scene → JSON output
    │   (scene_type, weather, expected_objects, normality_criteria)
    │
    ├── Agent 2: Reads image → LLaVA detects spatial violations → JSON output
    │   (positioning_violations, traffic_disruptions, safety_hazards)
    │
    ├── Agent 3: Reads image → LLaVA evaluates semantics → JSON output
    │   (domain_violations, object_categorization, safety_implications)
    │
    ├── Agent 4: Reads image → LLaVA checks visual appearance → JSON output
    │   (color_anomalies, texture_irregularities, shape_deformations)
    │
    └── Agent 5: Reads Agent 1-4 JSONs → LLaVA synthesizes → Final prompts
        (prompt_v1: "wild zebra", prompt_v2: "zebra", confidence: 0.85)

Step 2: run_evaluate.py uses Agent 5's prompts
    │
    ├── GroundingDINO("wild zebra") → Bounding boxes
    ├── CLIP verification → Filter false positives
    ├── SAM segmentation → Pixel masks
    └── Compare with ground truth → IoU, F1, AUROC
```

---

## 5. Directory Structure

```
MAVR-OOD/
├── src/
│   ├── agents/
│   │   ├── vlm_backend.py        # LLaVA-7B inference engine
│   │   ├── agent1.py             # Scene Context Analyzer
│   │   ├── agent2.py             # Spatial Anomaly Detector
│   │   ├── agent3.py             # Semantic Inconsistency Analyzer
│   │   ├── agent4.py             # Visual Appearance Evaluator
│   │   ├── agent5.py             # Reasoning Synthesizer
│   │   └── run_all_agents.py     # Master orchestrator script
│   └── clip_verifier.py          # CLIP semantic verification
├── GroundingDINO/                 # Open-vocabulary detector
│   └── groundingdino/
│       ├── models/GroundingDINO/  # Model architecture
│       ├── config/                # Model configurations
│       └── util/                  # Utilities (inference, slconfig)
├── segment_anything/              # SAM segmentation
│   └── segment_anything/
│       ├── build_sam.py           # Model builder
│       ├── predictor.py           # SamPredictor
│       └── modeling/              # ViT encoder, decoder
├── data/
│   └── challenging_subset/
│       ├── original/              # Input images (13 road scenes)
│       └── labels/                # Ground truth masks
├── weights/                       # Model checkpoints
│   ├── groundingdino_swint_ogc.pth
│   └── sam_vit_h_4b8939.pth
├── outputs/                       # Pipeline outputs
│   ├── challenging_subset_prompts/# Agent JSON results
│   └── evaluation_results/        # Evaluation visualizations
├── dataset.py                     # Dataset loaders (Road Anomaly, Fishyscapes, SegmentMe)
├── run_evaluate.py                # Full evaluation pipeline
├── app.py                         # Gradio web frontend
├── fix_colab_compat.py            # Colab compatibility patcher
├── COLAB_RUN.md                   # Colab execution guide
└── requirements.txt               # Python dependencies
```

---

## 6. Dependencies and Libraries

### Core Libraries:
| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥2.0 | Deep learning framework |
| torchvision | ≥0.15 | Image transforms |
| transformers | ≥4.36.0 | HuggingFace model loading (LLaVA, BERT) |
| accelerate | ≥0.25.0 | Model device placement |
| bitsandbytes | ≥0.41.0 | 4-bit quantization |
| OpenAI CLIP | latest | Semantic verification |
| NumPy | ≥1.24 | Array operations |
| Pillow | ≥9.0 | Image I/O |
| OpenCV | ≥4.0 | Image processing |
| scikit-learn | ≥1.0 | Metrics (ROC, AUC) |
| Gradio | ≥4.0.0 | Web UI |
| matplotlib | ≥3.5 | Visualization |

### Model Checkpoints:
| Model | Size | Source |
|-------|------|--------|
| LLaVA-1.5-7B | ~4GB (4-bit) / ~14GB (FP16) | HuggingFace Hub |
| GroundingDINO SwinT-OGC | ~700MB | IDEA Research GitHub |
| SAM ViT-H | ~2.5GB | Meta AI |
| CLIP ViT-B/32 | ~1GB | OpenAI |

---

## 7. Quantization and VRAM Optimization

### The VRAM Challenge:
- Google Colab T4 GPU: **14.5GB VRAM**
- LLaVA-7B in FP16: **~14GB** (leaves no room for inference)
- Solution: **4-bit NF4 quantization** via bitsandbytes

### How 4-bit Quantization Works:
```
FP16 (16-bit):  Each weight = 2 bytes   → 7B params × 2 bytes = 14GB
4-bit NF4:      Each weight = 0.5 bytes  → 7B params × 0.5 bytes = 3.5GB + overhead ≈ 4GB
```

### NF4 (Normal Float 4-bit):
- Uses information-theoretically optimal data type for normally distributed weights
- Double quantization: quantizes the quantization constants themselves
- Compute dtype: FP16 (weights decompressed to FP16 during computation)
- Near-zero quality loss compared to FP16 inference

### VRAM Budget on Colab T4:
```
LLaVA (4-bit):       ~4.0 GB
GroundingDINO:        ~1.5 GB
SAM ViT-H:           ~2.5 GB
CLIP ViT-B/32:       ~1.0 GB
Inference overhead:   ~2.0 GB
────────────────────────────
Total:               ~11.0 GB of 14.5 GB available ✅
```

---

## 8. Evaluation Methodology

### 8.1 Datasets Used

**Road Anomaly Dataset (Challenging Subset)**:
- 13 curated images featuring animals on roads (zebras, cows, sheep, donkeys, rhinos, wild boars)
- Ground truth: Binary pixel masks (0 = background, 1 = anomaly)
- Source: Real-world road photographs from various countries

### 8.2 Metrics Explained

| Metric | What It Measures | Range | Better |
|--------|-----------------|-------|--------|
| **IoU** | Overlap accuracy of mask vs ground truth | 0–1 | Higher |
| **F1** | Balance of precision and recall | 0–1 | Higher |
| **AUROC** | Overall discrimination ability | 0–1 | Higher |
| **FPR@95** | False alarm rate at 95% detection rate | 0–1 | Lower |
| **AUUPRC** | Precision-recall area in uncertain threshold range | 0–1 | Higher |

### 8.3 Dual-Prompt Strategy

The system generates two prompt variants for robustness:
- **V1** (descriptive): e.g., "wild zebra crossing road" — more specific, higher precision
- **V2** (generic): e.g., "zebra" — broader match, higher recall

The evaluation tries V1 first; if no detections survive CLIP verification, it falls back to V2. This dual-prompt strategy significantly improves recall without sacrificing precision.

---

## 9. Key Design Decisions

### 9.1 Why Multi-Agent Instead of Single-Model?
- **Specialization**: Each agent focuses on one analysis dimension, avoiding cognitive overload
- **Robustness**: If one agent misses an anomaly, others may catch it
- **Explainability**: Each agent provides interpretable reasoning for its conclusions
- **Extensibility**: New agents can be added without modifying existing ones

### 9.2 Why LLaVA-7B Instead of Larger Models?
- Runs on free Colab T4 GPU (14.5GB VRAM) with 4-bit quantization
- No API costs (fully local inference)
- Good balance of capability vs. resource requirements
- Open-source and reproducible

### 9.3 Why GroundingDINO + SAM Instead of End-to-End Segmentation?
- **Open vocabulary**: Can detect ANY object described in text (no retraining needed)
- **Two-stage design**: Separates "what to find" (GroundingDINO) from "how to segment" (SAM)
- **Foundation models**: Both are pretrained on massive datasets, generalize well
- **CLIP verification**: Adds a third model's opinion to reduce false positives

### 9.4 Why Rule-Based Synthesis (Agent 5) Instead of Another VLM Call?
- Agent 5 uses text-only LLaVA (no image) to focus purely on reasoning
- Applies deterministic priority rules (Animals > Vehicles > Obstacles)
- Generates structured prompts optimized for GroundingDINO's input format

---

## 10. Limitations and Future Work

### Current Limitations:
1. **Processing speed**: ~20-25 seconds per image per agent on T4 GPU
2. **Model reloading**: Each agent subprocess reloads LLaVA (~55 sec overhead per agent)
3. **JSON parsing**: LLaVA sometimes generates malformed JSON, requiring fallback parsing
4. **Single object focus**: System prioritizes the TOP 1 anomaly per image
5. **Dataset size**: Evaluated on 13 images (challenging subset)

### Future Extensions:
1. Multi-object detection (detect ALL anomalies, not just the top one)
2. Real-time video processing pipeline
3. Integration with autonomous driving decision systems
4. Larger VLM backends (LLaVA-13B, GPT-4V) for improved reasoning
5. Active learning to improve prompts based on evaluation feedback
