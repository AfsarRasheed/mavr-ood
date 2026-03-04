# MAVR: Multi-Agent Vision-Language Reasoning for Reliable Object Localization in Road Environments

## Complete Project Documentation

---

## 1. Project Overview

### 1.1 What is MAVR?

MAVR (Multi-Agent Vision-Language Reasoning) is a **confidence-aware multi-agent vision-language system** designed for **reliable object localization** in road environments. The system addresses the challenge of accurately detecting and segmenting specific objects in complex road scenes through a collaborative multi-agent pipeline that combines VLM reasoning with grounded detection.

### 1.2 Problem Statement

Object localization in road environments presents unique challenges:
- **Visual Complexity**: Road scenes contain numerous objects — vehicles, pedestrians, signs, animals, debris — in varying lighting, weather, and occlusion conditions.
- **Ambiguous Queries**: Users may describe objects using spatial relationships ("the car next to the truck"), visual attributes ("the red truck on the left"), or relative descriptions ("the largest zebra").
- **Open-Vocabulary Requirements**: Traditional detectors are limited to pre-trained categories. Road scenes frequently contain novel or unexpected objects that don't fit standard class labels.
- **Reliability**: A single model may hallucinate or misidentify objects. Road safety demands verified, reliable detections.

MAVR addresses these challenges by employing a **multi-agent reasoning framework** where specialized VLM agents analyze the scene from different perspectives, then feed their analysis into a grounded detection and verification pipeline.

### 1.3 Key Innovations

1. **Multi-Agent Scene Analysis**: Two specialized LLaVA-7B agents (Scene Understanding + Attribute Matching) collaboratively analyze the image to understand context and refine detection prompts.
2. **Confidence-Aware Verification**: CLIP semantic verification filters false positives after GroundingDINO detection — a "second opinion" that ensures reliability.
3. **Advanced Spatial Reasoning**: Supports both absolute spatial terms (left, right, center, largest) and relational spatial reasoning (next to, behind, near) with reference-object detection.
4. **Reference-Object Detection**: For relational queries like "the car next to the truck," the system detects both the target and the anchor object, then uses Euclidean distance to select the correct target.
5. **Memory-Managed Pipeline**: Phased model loading and unloading ensures the entire pipeline runs on a single T4 GPU (14.5GB VRAM) without out-of-memory errors.
6. **Robust JSON Parsing**: Handles inconsistent LLaVA outputs through multi-layered fallback parsing — markdown fence removal, trailing comma fixes, and text extraction.

---

## 2. System Architecture

### 2.1 High-Level Pipeline

```
Input: Image + User Query (e.g., "the grey car on the left")
    │
    ▼
┌─────────────────────────────────────────────────────┐
│         PHASE 1: Multi-Agent VLM Analysis           │
│                                                     │
│  Agent 1: Scene Understanding (LLaVA-7B)            │
│    → Scene type, lighting, object inventory         │
│                                                     │
│  Agent 2: Attribute Matching (LLaVA-7B)             │
│    → Object identification, prompt refinement       │
│    → Ambiguity assessment, match reasoning          │
│                                                     │
│  Output: Optimized detection prompt + scene context │
│  Memory: LLaVA freed from GPU after this phase      │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│         PHASE 2: Query Parsing                      │
│                                                     │
│  Extract: target object, visual attributes,         │
│           spatial term, anchor (reference) object    │
│                                                     │
│  Examples:                                          │
│    "the grey car on the left"                       │
│      → target: "grey car", spatial: "left"          │
│    "the car next to the truck"                      │
│      → target: "car", anchor: "truck",              │
│        spatial: "next_to"                           │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│         PHASE 3: Grounded Detection                 │
│                                                     │
│  Step 3: GroundingDINO — candidate detection        │
│    → Open-vocabulary detection from text prompt     │
│    → Retry logic: lower threshold, raw prompt       │
│                                                     │
│  Step 4: CLIP Verification — semantic filtering     │
│    → Cosine similarity between crop and prompt      │
│    → Threshold: 0.25 (rejects non-matching)         │
│                                                     │
│  Step 5: Spatial Reasoning — target selection       │
│    → Absolute: left/right/center/largest/smallest   │
│    → Relational: next_to/behind/in_front/above      │
│    → Anchor detection via separate DINO call        │
│    → Euclidean distance for relational selection    │
│                                                     │
│  Step 6: SAM Segmentation — pixel-precise masks     │
│    → Box-prompted segmentation via SAM ViT-H        │
│                                                     │
│  Output: Segmentation masks + step visualizations   │
└─────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| VLM Backend | LLaVA-1.5-7B (HuggingFace) | Scene understanding + attribute matching |
| Quantization | bitsandbytes (4-bit NF4) | VRAM optimization (~4GB) |
| Object Detection | GroundingDINO (SwinT-OGC) | Open-vocabulary grounding |
| Semantic Verification | OpenAI CLIP (ViT-B/32) | Detection filtering |
| Segmentation | SAM ViT-H (Segment Anything) | Pixel-precise masks |
| Frontend | Streamlit + ngrok | Interactive web demo |
| Compute | Google Colab (T4 GPU) | Cloud inference |

---

## 3. Module-by-Module Description

### 3.1 VLM Backend (`src/agents/vlm_backend.py`)

**Purpose**: Provides a unified inference interface for all VLM agents using LLaVA-1.5-7B.

**Technical Details**:
- **Model**: `llava-hf/llava-1.5-7b-hf` — a 7-billion parameter multimodal model combining a CLIP vision encoder with a Vicuna-7B language model
- **Quantization**: Automatic 4-bit NF4 quantization using `bitsandbytes` on Linux/Colab, reducing VRAM from ~14GB to ~4GB
- **Fallback**: FP16 with `device_map="auto"` on Windows for GPU/CPU split
- **Singleton Pattern**: Model loaded once, reused across all agents within a process
- **Memory Management**: `gc.collect()` and `torch.cuda.empty_cache()` after each inference
- **Prompt Format**: Uses LLaVA chat format: `USER: <image>\n{prompt}\nASSISTANT:`
- **Generation Parameters**: temperature=0.2, top_p=0.9, max_new_tokens=4096

**Key Function**: `run_vlm(messages, image_path=None) → str`

---

### 3.2 Agent 1: Scene Understanding Agent (`src/text_guided/scene_agent.py`)

**Purpose**: Establishes full scene context — what objects are present, where they are, and what the environment looks like.

**Responsibilities**:
1. Identify the scene type (city street, highway, intersection, rural road, parking lot)
2. Assess lighting and weather conditions
3. Catalog all visible objects with their positions, colors, and approximate sizes
4. Provide a structured JSON inventory of the scene

**Prompt Design**:
The agent is instructed to return **JSON only** — no conversational text. This is enforced through strict prompt engineering:
```
"Analyze this road scene image. Return ONLY valid JSON, no other text.
{
  "scene_type": "...",
  "lighting": "...",
  "objects": [
    {"name": "...", "position": "...", "color": "...", "size": "..."}
  ]
}"
```

**Robust JSON Parsing** (`parse_json_response`):
Since LLaVA-7B occasionally returns malformed JSON, the parser includes multi-layered fallbacks:
1. Direct `json.loads()` on the raw response
2. Strip markdown code fences (`` ```json ... ``` ``)
3. Remove trailing commas (`,}` → `}`, `,]` → `]`)
4. Fix escaped newlines and special characters
5. Extract JSON substring from mixed text
6. Fallback to `{"scene_type": "unknown", "objects": []}` if all parsing fails

**Output Schema**:
```json
{
    "scene_type": "city street",
    "lighting": "bright",
    "objects": [
        {"name": "red car", "position": "left", "color": "red", "size": "medium"},
        {"name": "pedestrian", "position": "center", "color": "dark clothing", "size": "small"},
        {"name": "white truck", "position": "right", "color": "white", "size": "large"}
    ]
}
```

**Why This Agent Matters**: Provides the foundational scene context that the Attribute Matching Agent uses to identify and locate the user's target object. Without scene understanding, the system would rely solely on GroundingDINO's raw detection, losing the VLM's semantic reasoning advantage.

---

### 3.3 Agent 2: Attribute Matching Agent (`src/text_guided/attribute_agent.py`)

**Purpose**: Takes the user's natural language query and the scene context, then produces an optimized detection prompt and reasoning about which object best matches.

**Responsibilities**:
1. Interpret the user's query in the context of the analyzed scene
2. Match the described object to scene inventory based on color, type, position, and size
3. Assess ambiguity — are there multiple possible matches?
4. Generate a refined GroundingDINO-optimized prompt
5. Provide reasoning for the selection

**Input**: User prompt + Agent 1's scene analysis JSON

**Prompt Design**:
```
"Given the scene analysis and user query, determine which object the user is referring to.
Return ONLY valid JSON:
{
  "reasoning": "explanation of match logic",
  "recommended_prompt": "optimized prompt for GroundingDINO",
  "ambiguity": "low/medium/high",
  "matched_objects": [{"name": "...", "position": "...", "confidence": "..."}]
}"
```

**Output Schema**:
```json
{
    "reasoning": "The user wants the grey car on the left. There is one grey vehicle on the left side.",
    "recommended_prompt": "grey car",
    "ambiguity": "low",
    "matched_objects": [
        {"name": "grey car", "position": "left", "confidence": "high"}
    ]
}
```

**Template Rejection Logic**:
LLaVA-7B sometimes copies template text literally (e.g., "optimized detection prompt for GroundingDINO"). The pipeline detects and rejects these using keyword filtering:
```python
bad_keywords = ['groundingdino', 'optimized', 'detection prompt', 'example', 'template']
```

**Why This Agent Matters**: Bridges the gap between human natural language ("find the big white truck on the right") and GroundingDINO's optimal input format ("white truck"). The agent's reasoning also helps explain the system's decisions.

---

### 3.4 Query Parser (`src/text_guided/query_parser.py`)

**Purpose**: Rule-based parser that extracts structured query components from the user's natural language prompt.

**Extracted Components**:

| Component | Description | Example |
|-----------|-------------|---------|
| `object_prompt` | The core object description for GroundingDINO | "grey car" |
| `attribute` | Visual attribute (color) | "grey" |
| `spatial` | Spatial positioning term | "left", "next_to" |
| `anchor` | Reference object for relational queries | "truck" (from "next to the truck") |
| `detect_all` | Whether to find ALL matches or filter to one | `True` if no spatial term |

**Supported Spatial Terms**:

| Category | Terms | Logic |
|----------|-------|-------|
| **Absolute Position** | left, right, center, top, bottom | Min/max of box center coordinates |
| **Size-Based** | largest, smallest | Max/min of box area |
| **Depth-Based** | nearest, farthest | Max/min of y-center (perspective) |
| **Relational** | next_to, beside, near, behind, in_front, above, below | Euclidean distance to anchor object |

**Relational Phrase Detection**:
```python
RELATIONAL_PHRASES = [
    "next to the", "beside the", "near the", "close to the",
    "behind the", "in front of the", "above the", "below the",
]
```

When a relational phrase is found, the parser extracts both the target and the anchor:
- Input: `"find the car next to the red truck"`
- Output: `target="car"`, `anchor="red truck"`, `spatial="next_to"`

**Spatial Filter Function** (`spatial_filter`):

For absolute terms, uses simple coordinate logic:
- `left` → `argmin(x_centers)`
- `right` → `argmax(x_centers)`
- `largest` → `argmax(areas)`

For relational terms with anchor boxes:
```python
# Detect anchor object via separate GroundingDINO call
# Calculate Euclidean distance from each target candidate to anchor center
distances = sqrt((x_centers - anchor_cx)² + (y_centers - anchor_cy)²)
# Select the target closest to the anchor
selected = argmin(distances)
```

---

### 3.5 Detection Pipeline (`src/text_guided/pipeline.py`)

**Purpose**: Main orchestrator that runs all 6 steps sequentially with memory management.

**Pipeline Steps**:

| Step | Component | Action |
|------|-----------|--------|
| 1 | Scene Understanding Agent | Analyze scene → JSON inventory |
| 2 | Attribute Matching Agent | Match query to scene → refined prompt |
| — | LLaVA Cleanup | Free LLaVA from GPU memory |
| 3 | GroundingDINO | Detect candidates using refined prompt |
| 4 | CLIP Verification | Filter false positives (threshold: 0.25) |
| 5 | Spatial Filter | Select target based on spatial/relational term |
| 6 | SAM Segmentation | Generate pixel-precise mask for selected object |

**Retry Logic (Step 3)**:
If GroundingDINO finds zero candidates:
1. Retry with lower box_threshold (0.20)
2. Retry with raw user prompt instead of agent-refined prompt

**Anchor Object Detection (Step 5)**:
For relational queries, a second GroundingDINO call detects the anchor object:
```
User: "the car next to the truck"
  → GroundingDINO("car") → 5 car candidates
  → GroundingDINO("truck") → 1 truck detected (anchor)
  → Euclidean distance: pick the car closest to the truck
```

**Memory Management**:
LLaVA agents (Phase 1) run first, then LLaVA is explicitly freed from GPU before loading GroundingDINO, CLIP, and SAM (Phase 2). This phased approach prevents OOM on T4 GPU:

```python
# After LLaVA agents complete:
del vlm_mod._model
vlm_mod._model = None
gc.collect()
torch.cuda.empty_cache()
# Now safe to load GroundingDINO + CLIP + SAM
```

---

### 3.6 CLIP Semantic Verifier (`src/clip_verifier.py`)

**Purpose**: Post-detection verification layer that filters false positive detections from GroundingDINO.

**Technical Details**:
- **Model**: OpenAI CLIP ViT-B/32 (~1GB)
- **Method**: Crops each detected bounding box, encodes it with CLIP's vision encoder, and computes cosine similarity with the text prompt
- **Threshold**: Default 0.25 — detections below this similarity score are filtered out
- **Role**: Sits between GroundingDINO detection and spatial filtering

**Key Functions**:
- `verify_detections(image, boxes, phrases, text_prompt) → filtered results`
- `generate_heatmap(image, text_prompt) → numpy heatmap` — extracts spatial similarity from CLIP ViT patch tokens

**Fallback Mechanism**: If CLIP rejects ALL detections, the system retains the highest-scoring detection to avoid losing valid targets.

**Why This Module Matters**: GroundingDINO can detect non-matching regions (false positives). CLIP acts as a semantic "second opinion" — if the cropped region doesn't look like the described object, it's filtered out. This multi-model verification is what makes the system "confidence-aware."

---

### 3.7 GroundingDINO (Open-Vocabulary Detector)

**Purpose**: Detects objects in images based on free-form text descriptions — no fixed class vocabulary.

**Technical Details**:
- **Architecture**: DINO detector + BERT text encoder + cross-attention fusion
- **Backbone**: Swin Transformer (SwinT-OGC variant)
- **Input**: Image + text prompt (e.g., "grey car" or "white truck")
- **Output**: Bounding boxes with confidence scores in cxcywh format
- **Key Advantage**: Can detect ANY object described in natural language

**Configuration**:
- Box threshold: 0.35 (minimum detection confidence)
- Text threshold: 0.25 (minimum text-logit match)
- Image preprocessing: RandomResize([800], max_size=1333) + Normalize

---

### 3.8 SAM — Segment Anything Model

**Purpose**: Generates pixel-precise segmentation masks from bounding box prompts.

**Technical Details**:
- **Architecture**: ViT-H image encoder + prompt encoder + mask decoder
- **Variant**: SAM ViT-H (most accurate, 2.5GB checkpoint)
- **Input**: Image + bounding boxes from GroundingDINO/spatial filter
- **Output**: Binary segmentation masks at full image resolution

**Device Handling**: `apply_boxes_torch` internally uses `deepcopy`, which can reset the CUDA device. The pipeline explicitly moves transformed boxes to the SAM model's device after transformation.

---

### 3.9 Step-by-Step Visualizer (`src/text_guided/visualizer.py`)

**Purpose**: Generates 6 annotated visualization images showing each pipeline step's contribution.

**Visualizations Generated**:

| Step | Image Content |
|------|---------------|
| Step 1 | Original image with scene type, lighting, and object list overlay |
| Step 2 | Original image with agent reasoning, recommended prompt, and ambiguity |
| Step 3 | Image with ALL GroundingDINO candidate bounding boxes (cyan) |
| Step 4 | Image with CLIP-verified boxes (green=pass, red=reject) with scores |
| Step 5 | Image with spatially selected box(es) highlighted (green) |
| Step 6 | Image with final SAM segmentation mask overlay and detection label |

**Confidence Display**: Shows confidence values for matched objects. Displays "N/A" instead of empty brackets when LLaVA doesn't return confidence values.

---

### 3.10 Streamlit Web Application (`streamlit_app.py`)

**Purpose**: Full interactive web interface with two detection modes.

**Features**:
- **Tab 1: Multi-Agent VLM Detection** — Upload image + text prompt, run full 6-step pipeline, view step-by-step visualizations and pipeline summary
- **Tab 2: OOD Detection** — Autonomous anomaly detection using 5 LLaVA agents + 3-panel pipeline visualization
- **Pipeline Summary**: Clean, human-readable summary of each step's results
- **Configurable Thresholds**: CLIP threshold (0.25) and Box threshold (0.35) adjustable via sidebar
- **Lazy Loading**: Models imported only when detection is triggered — UI renders instantly

**Deployment**:
- Local: `streamlit run streamlit_app.py`
- Colab: Streamlit + ngrok or localtunnel for public URL

---

## 4. Multi-Agent Workflow (Detailed Data Flow)

### Step-by-Step Execution:

```
User Input: "find the grey car next to the red truck"
    │
    ▼
Step 1: Scene Understanding Agent (LLaVA-7B)
    │   Reads image → Analyzes scene → JSON output
    │   Scene: "city street", Lighting: "bright"
    │   Objects: [{red car, left}, {grey car, center},
    │            {red truck, right}, {bus, far-right}]
    │
    ▼
Step 2: Attribute Matching Agent (LLaVA-7B)
    │   Reads image + scene context + user query
    │   Reasoning: "Grey car is near the red truck on the right"
    │   Recommended prompt: "grey car"
    │   Ambiguity: low
    │
    │── LLaVA freed from GPU ──
    │
    ▼
Query Parser
    │   target: "grey car"
    │   anchor: "red truck"
    │   spatial: "next_to"
    │
    ▼
Step 3: GroundingDINO Detection
    │   Prompt: "grey car" → 3 candidate boxes found
    │
    ▼
Step 4: CLIP Verification
    │   3/3 candidates verified (similarity > 0.25)
    │
    ▼
Step 5: Spatial Reasoning
    │   Anchor detection: GroundingDINO("red truck") → 1 box found
    │   Euclidean distance: car_2 is closest to truck → SELECTED
    │
    ▼
Step 6: SAM Segmentation
    │   Box → pixel-precise segmentation mask
    │
    ▼
Output: Segmentation mask of the grey car next to the red truck
        + 6 step-by-step visualization images
        + Pipeline summary
```

---

## 5. Directory Structure

```
mavr-ood/
├── src/
│   ├── agents/
│   │   └── vlm_backend.py         # LLaVA-7B inference backend (shared)
│   ├── text_guided/
│   │   ├── __init__.py            # Package init (exports run_text_guided_pipeline)
│   │   ├── scene_agent.py         # Agent 1: Scene Understanding
│   │   ├── attribute_agent.py     # Agent 2: Attribute Matching
│   │   ├── query_parser.py        # Query parsing + spatial filter + anchor detection
│   │   ├── pipeline.py            # Main 6-step orchestrator
│   │   └── visualizer.py          # Step-by-step visualization generator
│   ├── clip_verifier.py           # CLIP semantic verification + heatmap
│   └── model_loader.py            # Singleton model loader (GroundingDINO, SAM, CLIP)
├── GroundingDINO/                 # GroundingDINO (object detection)
├── segment_anything/              # SAM (segmentation)
├── data/
│   └── challenging_subset/        # 13 test images with ground truth
│       ├── original/              # Input images (road scenes)
│       └── labels/                # Ground truth binary masks (.png)
├── weights/                       # Model checkpoints
│   ├── groundingdino_swint_ogc.pth
│   └── sam_vit_h_4b8939.pth
├── outputs/
│   ├── text_guided/               # Pipeline output JSONs + visualizations
│   └── vlm_evaluation/            # Evaluation results + comparison images
├── streamlit_app.py               # Streamlit web frontend
├── run_evaluate_vlm.py            # Evaluation script (IoU, F1, Precision, Recall)
├── COLAB_RUN_TEXT_GUIDED.md       # Colab execution guide
├── MAVR_PROJECT_DOCUMENTATION.md  # This documentation
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
| OpenCV | ≥4.0 | Image processing, visualization |
| Streamlit | latest | Web UI |
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
- LLaVA-7B in FP16: **~14GB** (leaves no room for other models)
- Solution: **4-bit NF4 quantization** + **phased model loading**

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

### VRAM Budget (Phased Loading on T4):
```
Phase 1 (VLM Analysis):
  LLaVA (4-bit):       ~4.0 GB
  Inference overhead:   ~2.0 GB
  ─────────────────────────────
  Total:               ~6.0 GB ✅

  → LLaVA freed from GPU

Phase 2 (Detection):
  GroundingDINO:        ~1.5 GB
  SAM ViT-H:           ~2.5 GB
  CLIP ViT-B/32:       ~1.0 GB
  Inference overhead:   ~2.0 GB
  ─────────────────────────────
  Total:               ~7.0 GB ✅

Peak VRAM: ~7.0 GB of 14.5 GB available
```

---

## 8. Evaluation Methodology

### 8.1 Evaluation Script (`run_evaluate_vlm.py`)

**Purpose**: End-to-end evaluation that runs the full VLM pipeline on test images and computes segmentation metrics against ground truth masks.

**Pipeline Flow**:
1. Load all models (GroundingDINO, SAM, CLIP)
2. For each image in the dataset:
   - Map image to a predefined text query (e.g., `animals03_Zebras_in_the_road.jpg` → `"the zebra"`)
   - Run the full 6-step VLM pipeline with the query
   - Extract predicted SAM segmentation mask
   - Load ground truth binary mask from `labels/` directory
   - Compute pixel-level metrics: IoU, F1, Precision, Recall
   - Save side-by-side comparison visualization (Original → Ground Truth → Predicted)
3. Compute aggregate metrics and generate summary charts

**Usage**:
```bash
# Evaluate on dataset (13 images)
python run_evaluate_vlm.py

# Evaluate on custom images
python run_evaluate_vlm.py --data-dir ./data/custom_eval
```

### 8.2 Evaluation Metrics

| Metric | What It Measures | Range | Better |
|--------|-----------------|-------|--------|
| **IoU** | Overlap between predicted mask and ground truth | 0–1 | Higher |
| **F1** | Harmonic mean of precision and recall | 0–1 | Higher |
| **Precision** | Fraction of predicted pixels that are correct | 0–1 | Higher |
| **Recall** | Fraction of ground truth pixels that are detected | 0–1 | Higher |
| **Localization Success** | IoU > 0.1 (object was found) | % | Higher |

### 8.3 Query Mapping

Each test image is mapped to a natural language query:

| Image | Query |
|-------|-------|
| `animals03_Zebras_in_the_road.jpg` | "the zebra" |
| `animals05_cows_near_Phonsavan_Laos.jpg` | "the cow" |
| `animals06_sheep_roads_lambs.jpg` | "the sheep" |
| `animals15_Doebeln_Pferdebahn.jpg` | "the horse" |
| `animals23_rhino_crossing_road.jpg` | "the rhino" |
| `animals25_jablonna_dziki.jpg` | "the wild boar" |
| ... | ... |

### 8.4 Custom Image Evaluation

To evaluate on your own images:
1. **Annotate**: Use [makesense.ai](https://www.makesense.ai/) or MS Paint to create binary masks (white = target, black = background)
2. **Organize**: Place images in `data/custom_eval/original/` and masks in `data/custom_eval/labels/`
3. **Add queries**: Update `QUERY_MAP` in `run_evaluate_vlm.py` with your image-query pairs
4. **Run**: `python run_evaluate_vlm.py --data-dir ./data/custom_eval`

### 8.5 Outputs

All results saved to `outputs/vlm_evaluation/`:
- `evaluation_results.json` — per-image and aggregate metrics
- `*_comparison.jpg` — side-by-side visualization per image (Original | Ground Truth | Predicted)
- `evaluation_summary.jpg` — bar chart (IoU/F1 per image) + pie chart (success rate)

---

## 9. Spatial Reasoning System

### 8.1 Absolute Spatial Terms

| Term | Selection Logic | Example Query |
|------|----------------|---------------|
| `left` / `leftmost` | Minimum x-center | "the car on the left" |
| `right` / `rightmost` | Maximum x-center | "the truck on the right" |
| `center` / `middle` | Closest to image center-x | "the car in the middle" |
| `top` / `upper` | Minimum y-center | "the sign at the top" |
| `bottom` / `lower` | Maximum y-center | "the car at the bottom" |
| `largest` / `biggest` | Maximum bounding box area | "the largest zebra" |
| `smallest` | Minimum bounding box area | "the smallest car" |
| `nearest` / `closest` | Maximum y-center (perspective) | "the nearest car" |
| `farthest` | Minimum y-center (perspective) | "the farthest car" |

### 8.2 Relational Spatial Terms (Reference-Object Detection)

| Term | Selection Logic | Example Query |
|------|----------------|---------------|
| `next_to` / `beside` / `near` | Min Euclidean distance to anchor | "the car next to the truck" |
| `behind` | Min distance among candidates above anchor | "the car behind the truck" |
| `in_front` | Min distance among candidates below anchor | "the person in front of the car" |
| `above` | Candidates above anchor, closest x-alignment | "the sign above the car" |
| `below` | Candidates below anchor, closest x-alignment | "the dog below the sign" |

### 8.3 Relational Detection Flow

```
Query: "the grey car next to the red truck"
  │
  ├── GroundingDINO("grey car") → 3 candidate boxes
  │     Box A: center=(100, 300)
  │     Box B: center=(500, 290)
  │     Box C: center=(800, 310)
  │
  ├── GroundingDINO("red truck") → 1 anchor box
  │     Anchor: center=(520, 300)
  │
  └── Euclidean Distance:
        A → Anchor: 420 pixels
        B → Anchor: 22 pixels  ← CLOSEST → SELECTED ✅
        C → Anchor: 280 pixels
```

---

## 10. Key Design Decisions

### 9.1 Why Multi-Agent Instead of Single-Model?
- **Specialization**: Scene Understanding focuses on "what's in the scene," Attribute Matching focuses on "which object matches the query"
- **Robustness**: Two perspectives reduce the chance of misidentification
- **Explainability**: Each agent provides interpretable reasoning for its conclusions
- **Prompt Optimization**: Agent 2's refined prompt significantly improves GroundingDINO's detection accuracy

### 9.2 Why LLaVA-7B Instead of Larger Models?
- Runs on free Colab T4 GPU (14.5GB VRAM) with 4-bit quantization
- No API costs (fully local inference)
- Good balance of capability vs. resource requirements
- Open-source and reproducible

### 9.3 Why GroundingDINO + SAM Instead of End-to-End Segmentation?
- **Open vocabulary**: Can detect ANY object described in text (no retraining needed)
- **Two-stage design**: Separates "what to find" (GroundingDINO) from "how to segment" (SAM)
- **Foundation models**: Both pretrained on massive datasets, generalize well to road scenes
- **CLIP verification**: Adds a third model's opinion to reduce false positives

### 9.4 Why Phased Model Loading?
- LLaVA-7B (even quantized) + GroundingDINO + SAM + CLIP exceed T4's 14.5GB VRAM
- By running LLaVA first, then freeing it before loading detection models, the pipeline fits within memory constraints
- This is why the Streamlit app shows two phases: "Phase 1: LLaVA agents" → "Phase 2: Detection pipeline"

### 9.5 Why Rule-Based Spatial Reasoning Instead of VLM-Based?
- Deterministic and predictable — same input always produces same output
- No hallucination risk — Euclidean distance is mathematical, not probabilistic
- Faster — no additional VLM inference needed for spatial selection
- Extensible — new spatial terms can be added without retraining any model

---

## 11. Limitations and Future Work

### Current Limitations:
1. **Processing speed**: ~15-25 seconds per image (LLaVA agents + detection pipeline)
2. **LLaVA JSON consistency**: LLaVA-7B still occasionally generates malformed JSON despite strict prompts — mitigated by fallback parsing
3. **Spatial reasoning scope**: Relational terms limited to proximity-based (Euclidean distance). Does not support complex spatial logic like "between" or "surrounded by"
4. **Single-target focus**: For relational queries, selects exactly one target object
5. **Dataset domain**: Primarily tested on road environment images

### Future Extensions:
1. **Complex spatial relations**: Support "between X and Y," "inside," "surrounded by"
2. **Multi-target detection**: Return multiple matching objects with confidence ranking
3. **Real-time video processing**: Frame-by-frame detection with temporal consistency
4. **Larger VLM backends**: Integration with LLaVA-13B, GPT-4V, or Qwen2.5-VL for improved reasoning
5. **Active prompt refinement**: Use detection feedback to iteratively improve prompts
6. **Batch CLIP processing**: Stack all cropped images into one tensor for parallel verification
7. **Cross-modal retrieval**: Use CLIP embeddings for image-to-image similarity search across datasets
