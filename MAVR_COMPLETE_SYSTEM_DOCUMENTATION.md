# MAVR-OOD Complete System Documentation

## 1. Introduction

MAVR-OOD is a multi-agent vision-language project for road-scene understanding, anomaly localization, and query-based object detection. The project combines large vision-language models with grounded detection, semantic verification, and segmentation in order to make object localization more reliable and explainable in road environments.

The system currently supports two main operating modes:

1. OOD Detection
   Detects out-of-distribution or abnormal objects in road scenes without requiring a user query.

2. Text-Guided Detection
   Detects a user-specified object from a road image using natural-language prompts such as `the white car on the right` or `the zebra on the left`.

This project is designed for road-safety and traffic-scene scenarios where both localization accuracy and explainability matter.

---

## 2. Project Goals

The project is built around the following goals:

- improve reliability of object localization in complex road scenes
- combine reasoning and grounding instead of relying on a single model
- support open-vocabulary detection through natural language
- provide explainable outputs rather than only raw masks or boxes
- support both anomaly detection and user-driven object search

---

## 3. Core Idea

Instead of asking one model to do everything, MAVR-OOD separates the task into specialized stages:

- a vision-language model reasons about the scene
- GroundingDINO proposes candidate detections from text
- CLIP acts as a semantic verification layer
- SAM produces precise segmentation masks

This modular design is one of the main strengths of the project. It makes the system easier to explain, easier to debug, and more flexible than a single end-to-end detector.

---

## 4. Main System Modes

### 4.1 OOD Detection Mode

This mode is designed for abnormal-object detection in road scenes. The user uploads a road image, and the system tries to identify the most anomalous object in that scene.

Typical examples:

- animal on a road
- large rock blocking a road
- burning vehicle
- unusual object placed in traffic space

### 4.2 Text-Guided Detection Mode

This mode is designed for query-based object localization. The user provides:

- an image
- a text query

The system then identifies and segments the most relevant object described by the query.

Typical examples:

- `the white car on the right`
- `the zebra on the left`
- `the largest vehicle`
- `the car next to the truck`

---

## 5. High-Level Architecture

At a high level, the project has two branches of logic that share common grounding modules.

```text
                +----------------------+
                |      Input Image     |
                +----------+-----------+
                           |
          +----------------+----------------+
          |                                 |
          v                                 v
  OOD Detection Mode                Text-Guided Detection Mode
          |                                 |
          v                                 v
  Multi-Agent VLM Analysis         Scene + Query Understanding
          |                                 |
          v                                 v
   Prompt / anomaly target            Parsed object prompt
          |                                 |
          +---------------+-----------------+
                          |
                          v
              GroundingDINO Candidate Detection
                          |
                          v
                  CLIP Semantic Verification
                          |
                          v
                    Spatial Selection
                  (text-guided only when needed)
                          |
                          v
                     SAM Segmentation
                          |
                          v
               Visualizations + Explanation + Output
```

---

## 6. Main Technologies Used

| Component | Technology | Purpose |
|---|---|---|
| Vision-Language Reasoning | LLaVA-1.5-7B | scene analysis, attribute reasoning, anomaly reasoning |
| Quantization | bitsandbytes 4-bit NF4 | VRAM reduction for Colab/T4 use |
| Open-Vocabulary Detection | GroundingDINO | candidate box detection from text |
| Semantic Verification | OpenAI CLIP ViT-B/32 | filtering false positives |
| Segmentation | SAM ViT-H | precise mask generation |
| Backend | FastAPI | API routes for web app |
| Frontend | static HTML/CSS/JS | current web UI |
| Evaluation | NumPy, OpenCV, matplotlib | metrics and visualizations |

---

## 7. Why This Architecture Was Chosen

The architecture was chosen because each model is used for the task it does best.

### LLaVA

Used for:

- scene understanding
- attribute reasoning
- anomaly reasoning
- query interpretation
- explanation generation

Not used for:

- exact bounding-box localization
- final segmentation

### GroundingDINO

Used for:

- open-vocabulary text-conditioned object detection
- generating candidate boxes from short prompts

### CLIP

Used for:

- verifying whether a detected crop semantically matches the text prompt
- filtering weak or unrelated detections

### SAM

Used for:

- converting selected detections into pixel-level masks

This separation is important because general VLMs are good at semantic reasoning, but dedicated grounding and segmentation models are stronger for precise spatial localization.

---

## 8. OOD Detection Architecture

OOD detection is based on a 5-agent reasoning framework.

### Agent 1: Scene Context Analyzer

Purpose:

- understand the road environment
- identify what is normally expected in the scene

Typical outputs:

- scene type
- road context
- lighting and weather
- expected objects
- normality criteria

### Agent 2: Spatial Anomaly Detector

Purpose:

- identify objects that violate normal positioning or traffic flow

Typical outputs:

- positioning issues
- traffic flow disruption
- safety hazards

### Agent 3: Semantic Inconsistency Analyzer

Purpose:

- determine whether detected objects semantically belong in a road environment

Typical outputs:

- domain violations
- inappropriate objects
- safety implications

### Agent 4: Visual Appearance Evaluator

Purpose:

- identify visual or condition-based irregularities

Typical outputs:

- unusual colors
- texture issues
- shape or material anomalies

### Agent 5: Reasoning Synthesizer

Purpose:

- combine outputs from Agents 1–4
- determine the most anomalous object
- generate prompt variants for GroundingDINO

Typical outputs:

- prompt_v1
- prompt_v2
- final reasoning
- anomaly ranking

### OOD Detection Flow

```text
Image
  -> Agent 1: Scene Context
  -> Agent 2: Spatial Anomaly
  -> Agent 3: Semantic Analysis
  -> Agent 4: Visual Appearance
  -> Agent 5: Synthesis
  -> GroundingDINO detection
  -> CLIP verification
  -> SAM segmentation
  -> Optional metrics if ground truth mask is provided
```

---

## 9. Text-Guided Detection Architecture

Text-guided detection is based on a structured multi-stage pipeline.

### Step 1: Scene Understanding Agent

This agent analyzes the image and produces high-level scene context.

Typical outputs:

- scene type
- lighting
- visible object inventory

### Step 2: Attribute Matching Agent

This agent compares the user’s query with scene content and tries to identify the intended object more clearly.

Typical outputs:

- reasoning about the query
- matched objects
- ambiguity estimate
- recommended prompt for GroundingDINO

### Step 2.5: Query Parsing

The project currently supports two parsing modes:

- structured rule parsing
- LLaVA-assisted advanced parsing

The parser extracts:

- object prompt
- attribute
- spatial term
- anchor object
- second anchor for some cases
- ordinal information

### Step 3: GroundingDINO Candidate Detection

GroundingDINO uses the parsed prompt to detect possible target boxes.

### Step 4: CLIP Semantic Verification

Each candidate crop is checked against the prompt. CLIP removes weak semantic matches.

### Step 5: Spatial Selection

If the query contains spatial constraints, the pipeline uses rule-based spatial logic to choose the correct candidate.

Supported examples include:

- left / right / center
- largest / smallest
- nearest / farthest
- next to / behind / in front / above / below
- between
- ordinal directions such as second from the right

### Step 6: SAM Segmentation

The selected detection is converted into a segmentation mask.

### Step 7: Reasoning Agent

The reasoning agent summarizes the overall pipeline result in natural language.

### Text-Guided Detection Flow

```text
Image + User Query
  -> Scene Understanding
  -> Attribute Matching
  -> Query Parsing
  -> GroundingDINO candidates
  -> CLIP verification
  -> Spatial filtering
  -> SAM segmentation
  -> Explainable reasoning
```

---

## 10. Current Query Capability in Text-Guided Mode

The current `improvement/web-ui` branch supports advanced but still structured natural-language prompts.

It works best with prompts that include:

- object identity
- color or attribute
- spatial location
- simple relational references
- ordinal position

Examples:

- `the white car on the right`
- `the zebra on the left`
- `the largest vehicle`
- `the second car from the right`
- `the car next to the truck`
- `the object between the two vehicles`

It is weaker on very open-ended or highly abstract language such as:

- `the most dangerous vehicle`
- `the object that seems suspicious`
- `the car that looks more important`

Those kinds of prompts are part of the planned natural-language-grounding branch, not the current stable branch.

---

## 11. Backend Architecture

The current web application uses a FastAPI backend.

### Main API Endpoints

#### `/api/health`

Returns:

- status
- GPU information
- VRAM
- loaded models

#### `/api/detect`

Used for text-guided detection.

Input:

- image
- query

Returns:

- parsed query information
- step images
- final overlay
- attribute reasoning
- final reasoning
- pipeline summary
- total execution time

#### `/api/ood_detect`

Used for OOD detection.

Input:

- image
- optional ground-truth mask

Returns:

- number of detections
- prompt_v1 and prompt_v2
- reasoning
- visual outputs
- optional evaluation metrics
- full multi-agent outputs

---

## 12. Frontend / UI Structure

The current web UI is divided into two tabs:

### Text-Guided Detection Tab

Main UI sections:

- image upload
- search query input
- pipeline execution progress
- detection result
- pipeline visualization
- step-by-step pipeline results gallery
- reasoning agent output

### OOD Detection Tab

Main UI sections:

- image upload
- optional ground-truth upload
- result panels
- VLM reasoning output
- collapsible detailed agent analysis
- optional evaluation metrics

The UI has been refined to reduce clutter and surface only the most useful information first.

---

## 13. Visual Output Types

The project produces several types of visual outputs.

### Text-Guided Outputs

- original image
- final segmentation overlay
- step-by-step pipeline images
  - scene understanding
  - attribute matching
  - candidates
  - CLIP verification
  - spatial selection
  - final segmentation

### OOD Outputs

- original image
- detection view
- mask visualization
- binary anomaly mask view
- optional metrics view when GT is present

### Documentation / Evaluation Outputs

- comparison tables
- baseline vs MAVR figures
- metrics charts
- runtime charts

---

## 14. Memory and VRAM Strategy

The system is designed to run under limited GPU memory, especially in Google Colab.

### Key strategy

- LLaVA is loaded for reasoning stages
- after scene and attribute reasoning, LLaVA is freed from GPU memory
- detection models then run:
  - GroundingDINO
  - CLIP
  - SAM
- when final reasoning is needed, LLaVA is loaded again

This phased memory management is important for keeping the pipeline usable on T4-class GPUs.

---

## 15. Model Loading Strategy

Detection models are loaded on backend startup:

- GroundingDINO
- SAM
- CLIP verifier

LLaVA is loaded on demand through the shared backend interface and released when possible to control memory usage.

This helps:

- reduce startup delays for the web UI
- avoid GPU overload
- keep runtime stable

---

## 16. Evaluation and Metrics

The project supports evaluation when a ground-truth mask is available.

### Common metrics used

- IoU
- F1 score
- Precision
- Recall

### OOD mode behavior

If no ground-truth mask is provided:

- metrics are hidden
- only qualitative results are shown

If a ground-truth mask is provided:

- the metrics block appears
- results can be quantitatively evaluated

This design keeps the UI honest and avoids showing meaningless evaluation blocks.

---

## 17. Project Directory Overview

Below is the practical structure of the project.

```text
MAVR-OOD/
├── src/
│   ├── agents/
│   │   ├── agent1.py
│   │   ├── agent2.py
│   │   ├── agent3.py
│   │   ├── agent4.py
│   │   ├── agent5.py
│   │   ├── run_all_agents.py
│   │   └── vlm_backend.py
│   ├── text_guided/
│   │   ├── scene_agent.py
│   │   ├── attribute_agent.py
│   │   ├── query_parser.py
│   │   ├── pipeline.py
│   │   ├── reasoning_agent.py
│   │   └── visualizer.py
│   ├── clip_verifier.py
│   └── model_loader.py
├── GroundingDINO/
├── segment_anything/
├── static/
├── data/
│   └── challenging_subset/
├── outputs/
├── weights/
├── web_app.py
├── run_evaluate.py
├── run_evaluate_vlm.py
├── run_baseline_comparison.py
├── requirements.txt
└── documentation markdown files
```

---

## 18. Important Documentation Files Already in the Project

The project already contains several useful documentation files.

### Existing project and planning docs

- `MAVR_PROJECT_DOCUMENTATION.md`
- `PROJECT_DOCUMENTATION.md`
- `TEXT_GUIDED_ANALYSIS.md`
- `TEXT_GUIDED_REFINEMENT_ROADMAP.md`
- `NATURAL_LANGUAGE_GROUNDING_PLAN.md`
- `NATURAL_LANGUAGE_AGENT_ARCHITECTURE.md`
- `CURRENT_TEXT_GUIDED_PROMPT_CAPABILITIES.md`
- `PROJECT_HANDOFF_SUMMARY.md`

### Colab and evaluation docs

- `COLAB_SINGLE_IMAGE_BASELINE_GUIDE.md`
- `LLAVA_FINETUNING_GUIDE.md`
- `NEXT_STEPS.md`

This new file is intended to serve as a more complete, project-wide documentation reference.

---

## 19. Current Branching Strategy

The project is currently organized around two important branches.

### `improvement/web-ui`

Purpose:

- stable UI branch
- current deployable and testable branch
- contains present text-guided and OOD UI refinements
- serves as the safe branch for demos and iterative improvement

### `improvement/natural-language-grounding`

Purpose:

- future experimental branch
- intended for deeper natural-language grounding improvements
- designed to explore more advanced query understanding and candidate ranking without destabilizing the main UI branch

This branch separation is important for maintaining a stable working version while allowing ambitious experimentation.

---

## 20. Strengths of the Project

The project already has several strong technical and presentation strengths.

### Technical strengths

- multi-agent reasoning for road-scene interpretation
- open-vocabulary object grounding
- semantic verification with CLIP
- precise segmentation with SAM
- support for both anomaly detection and user-guided localization
- memory-aware GPU pipeline design
- explainable intermediate outputs

### Practical strengths

- usable web interface
- step-by-step visualizations
- qualitative and quantitative result support
- Colab-friendly workflow
- suitable for academic project documentation and demonstrations

---

## 21. Current Limitations

Although the system is strong, there are still limitations.

### In text-guided mode

- still relies on a mostly structured natural-language pipeline
- can fail when the full meaning of a query is not enforced strongly enough
- CLIP and spatial selection may occasionally overpower the true semantic intent
- reasoning output is still more technical than ideal for final UI presentation

### In OOD mode

- anomaly ranking is focused on the top result
- reasoning quality depends on agent consistency
- full agent outputs are more useful for analysis than for end-user display

### General limitations

- runtime is still relatively slow
- multi-stage systems are harder to tune than single-model demos
- some failure cases require stronger confidence handling

---

## 22. Planned Future Improvements

The main future direction is stronger natural-language grounding.

This includes:

- better full-query understanding
- stronger candidate scoring
- better attribute and condition enforcement
- improved ambiguity handling
- exact match / closest match / no reliable match logic
- more scene-aware final reporting

These changes are intended for the `improvement/natural-language-grounding` branch rather than the stable `improvement/web-ui` branch.

---

## 23. Example Workflows

### Example A: OOD Detection

Scenario:

- upload an image of a zebra standing on a road

What happens:

1. Agents analyze the road scene from five perspectives
2. Agent 5 selects the zebra as the most anomalous object
3. GroundingDINO finds zebra candidates
4. CLIP filters weak detections
5. SAM segments the final anomaly mask
6. If GT exists, metrics are computed

Expected result:

- anomaly mask on the zebra
- VLM reasoning text
- optional metrics if mask is provided

### Example B: Text-Guided Detection

Scenario:

- upload an image with two cars
- query: `the white car on the right`

What happens:

1. scene agent analyzes the image
2. attribute agent interprets the query
3. query parser extracts object, color, and spatial cue
4. GroundingDINO finds candidate cars
5. CLIP verifies semantic matches
6. spatial filter selects the right-side object
7. SAM produces the final mask
8. reasoning agent summarizes the result

Expected result:

- segmented mask for the right white car
- step-by-step gallery
- textual explanation

---

## 24. How This Project Is Useful Academically

This project is useful for academic documentation because it combines:

- computer vision
- vision-language reasoning
- explainable AI
- anomaly detection
- open-vocabulary grounding
- human-centered result presentation

It is not only a model demo. It is a modular system that demonstrates how multiple foundation models can be orchestrated for safer and more explainable road-scene analysis.

---

## 25. Conclusion

MAVR-OOD is a multi-component road-scene understanding system that combines reasoning, grounding, verification, segmentation, and explainability. Its architecture is built around the idea that reliable visual understanding should not come from a single prediction step, but from coordinated reasoning and verification.

The project currently provides:

- OOD anomaly localization
- text-guided object localization
- explainable multi-stage outputs
- web-based interaction
- evaluation support
- Colab-friendly execution

The `improvement/web-ui` branch represents the stable, current working version, while future natural-language improvements are planned in the `improvement/natural-language-grounding` branch.

For documentation purposes, this project can be presented as a robust, modular, and explainable system for road-scene anomaly understanding and natural-language-driven object localization.
