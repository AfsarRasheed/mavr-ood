# MAVR-OOD: Deep Technical Explanation & Final Review Guide

> **Multi-Agent Vision-Language Reasoning for Reliable Object Localization in Road Environments**

---

# PART 1: OOD (Out-of-Distribution) Detection — Deep Explanation

---

## 1.1 What Is an Anomaly in MAVR-OOD?

An **anomaly** (Out-of-Distribution object) in the context of MAVR-OOD is any object that **does not belong** in a typical road environment. Our system defines anomaly not by a fixed database or list, but through **contextual reasoning** — what is "normal" depends on the scene.

### Formal Definition

An object O is anomalous if:
- O is **semantically inappropriate** for the road context (e.g., a cow on a highway)
- O is **spatially misplaced** (e.g., a chair in the middle of a lane)
- O **violates behavioral norms** (e.g., a vehicle parked perpendicular across traffic lanes)
- O has **visual characteristics** inconsistent with the environment (e.g., unusual color, damaged surfaces)

### Key Insight: Context-Dependent Anomaly

A bicycle parked at a bike rack is **normal**. The same bicycle lying in the middle of a highway at night is an **anomaly**. MAVR-OOD understands this distinction because it first establishes what "normal" looks like for the given scene **before** looking for violations.

---

## 1.2 How Does MAVR-OOD Identify Anomalies?

The system uses a **5-agent collaborative reasoning pipeline**. Each agent analyzes the image from a different perspective, and their findings are synthesized to produce a grounding prompt that directs the detection and segmentation models.

### The 5-Agent OOD Pipeline

```
Image Input
    │
    ├──→ Agent 1: Scene Context Analyzer
    │        └─ "What kind of road scene is this?"
    │        └─ Establishes: scene type, weather, lighting, expected objects
    │
    ├──→ Agent 2: Spatial Anomaly Detector
    │        └─ "Are any objects in wrong positions?"
    │        └─ Checks: positioning violations, traffic disruptions, hazards
    │
    ├──→ Agent 3: Semantic Inconsistency Analyzer
    │        └─ "Do all objects belong on this road?"
    │        └─ Identifies: inappropriate objects, domain violations
    │
    ├──→ Agent 4: Visual Appearance Evaluator
    │        └─ "Does anything look visually unusual?"
    │        └─ Detects: color anomalies, texture issues, shape deformations
    │
    └──→ Agent 5: Reasoning Synthesizer
             └─ Combines all 4 analyses
             └─ Generates: prompt_v1 (detailed) and prompt_v2 (simple)
             └─ Example: prompt_v1="wild zebras", prompt_v2="zebras"
                  │
                  ▼
         GroundingDINO → CLIP Verify → SAM Segment
                  │
                  ▼
         Bounding Boxes + Segmentation Masks + Binary OOD Mask
```

### Step-by-Step Explanation

#### Agent 1: Scene Context Analyzer
**Purpose:** Establish what is "normal" for this specific scene.

The Scene Context Analyzer receives the image and uses LLaVA-7B (a Vision-Language Model) to understand:
- **Scene type**: urban road, rural highway, intersection, residential street, etc.
- **Road infrastructure**: number of lanes, sidewalks, barriers, traffic signals
- **Environmental conditions**: weather (clear, rainy, foggy), lighting (daylight, dusk, night), time of day
- **Expected objects**: based on the scene type, what should be here (cars, trucks, pedestrians, traffic signs)
- **Normality criteria**: what spatial arrangements and behaviors are typical

**Example output:**
```
Scene: urban_road
Weather: clear
Lighting: daylight
Expected objects: cars, trucks, pedestrians, traffic signals
Normality: vehicles in lanes, pedestrians on sidewalks
```

**Why this matters:** Without understanding the context, the system cannot determine what is anomalous. A tractor on a rural road is normal; on a highway, it may be anomalous.

#### Agent 2: Spatial Anomaly Detector
**Purpose:** Detect objects that violate spatial rules of the road.

This agent analyzes:
- **Objects on the road surface** that should not be there
- **Positioning violations**: objects in wrong lanes, wrong side of road, blocking intersections
- **Traffic disruptions**: anything interrupting normal traffic flow
- **Safety hazards**: immediate dangers to road users

**What it catches:**
- Animal on the carriageway
- Debris blocking a lane
- Vehicle facing the wrong direction
- Object on a pedestrian crossing when signal is green

#### Agent 3: Semantic Inconsistency Analyzer
**Purpose:** Determine if objects are **domain-appropriate** for road environments.

This is the most critical agent for OOD detection. It evaluates:
- **Detected objects**: enumerate everything visible
- **Road-appropriate objects**: cars, trucks, buses, motorcycles, traffic signs, road markings — normal
- **Inappropriate objects**: animals (cow, dog, zebra, deer), household items (chair, couch), construction materials in live lanes, people lying on the road
- **Domain violations**: why specific objects violate road-scene expectations
- **Safety assessment**: severity of the anomaly

**Core logic:** Animals on roads are **always** classified as inappropriate. This is hardcoded in the agent's reasoning prompt because no animal belongs on an active roadway in any driving context worldwide.

#### Agent 4: Visual Appearance Evaluator
**Purpose:** Detect anomalies based on how objects **look**, not what they **are**.

This agent examines:
- **Color anomalies**: unusual coloring that doesn't match the environment (e.g., bright object on a dark road)
- **Texture irregularities**: surfaces that look damaged, wet when road is dry, etc.
- **Shape deformations**: crashed or deformed vehicles, bent infrastructure
- **Overall condition**: general visual state of the scene
- **Most unusual object**: the single most visually out-of-place element

**What this catches that others miss:** A camouflaged object, a partially hidden obstacle, or damaged infrastructure that is semantically correct (it's a car) but visually anomalous (it's on fire).

#### Agent 5: Reasoning Synthesizer
**Purpose:** Integrate all four analyses and produce the final detection prompt.

The synthesizer:
1. Receives structured JSON from all four agents
2. Cross-references their findings to identify consensus anomalies
3. Prioritizes by severity: **Animals > Misplaced vehicles > Obstacles > Others**
4. Generates two grounding prompts:
   - **prompt_v1**: "adjective noun" (e.g., "wild zebras", "stray dog", "fallen debris")
   - **prompt_v2**: single noun (e.g., "zebras", "dog", "debris")
5. Reports overall confidence (0.0–1.0)

**Why two prompts?** GroundingDINO (the detection model) performs differently with different text inputs. If prompt_v1 finds nothing, the system automatically retries with prompt_v2 (broader search). This improves recall without sacrificing precision.

### After the Agents: Detection and Segmentation

Once Agent 5 produces the grounding prompts, the pipeline continues:

1. **GroundingDINO** receives the prompt and image → produces bounding boxes around matching objects
2. **CLIP Verification** checks that each detected region actually matches the prompt semantically (removes false positives)
3. **SAM (Segment Anything Model)** takes the verified boxes → produces pixel-precise segmentation masks
4. **Metrics computation** (if ground truth is available): IoU, F1, Precision, Recall

---

## 1.3 What Types of Anomalies Can MAVR-OOD Detect?

### Category 1: Animals on Roads
- **Examples:** cow, dog, horse, deer, zebra, goat, cat, wild boar
- **Detection strength:** HIGH — all agents flag animals as anomalous
- **How identified:** Agent 3 classifies any animal as "inappropriate_object"; Agent 2 flags spatial violation; Agent 5 prioritizes animals highest

### Category 2: Misplaced or Damaged Vehicles
- **Examples:** overturned truck, car facing wrong direction, vehicle parked in active lane, abandoned vehicle
- **Detection strength:** MEDIUM-HIGH — spatial and visual agents detect these
- **How identified:** Agent 2 detects positioning violation; Agent 4 detects shape deformation or unusual orientation

### Category 3: Road Debris and Obstacles
- **Examples:** fallen tree, construction materials, tire debris, furniture, rocks on road
- **Detection strength:** MEDIUM — depends on visual distinctiveness
- **How identified:** Agent 3 flags domain violation (furniture doesn't belong on road); Agent 2 flags lane obstruction

### Category 4: Pedestrian Anomalies
- **Examples:** person lying on road, jaywalker in highway, crowd blocking intersection
- **Detection strength:** MEDIUM — context-dependent
- **How identified:** Agent 2 flags spatial violation (person in vehicle lane); Agent 3 assesses appropriateness

### Category 5: Infrastructure Anomalies
- **Examples:** missing guardrail, collapsed bridge section, obscured traffic sign, road surface damage (potholes, cracks)
- **Detection strength:** LOWER — these are often subtle
- **How identified:** Agent 4 detects visual irregularities; Agent 1 notes infrastructure gaps

### Category 6: Environmental Anomalies
- **Examples:** flooding on road, oil spill, smoke/fire on road surface, unusual weather conditions causing visibility hazards
- **Detection strength:** MEDIUM — depends on visual prominence
- **How identified:** Agent 4 detects texture/color anomalies; Agent 1 notes environmental factors

### What MAVR-OOD Cannot Reliably Detect
- **Very small objects** (< 20 pixels) — below GroundingDINO's resolution
- **Objects identical to road elements** (e.g., a grey rock on grey asphalt) — low visual contrast
- **Abstract anomalies** (e.g., "this road is too narrow") — requires engineering knowledge
- **Temporal anomalies** (e.g., "this car has been parked here for 3 days") — single-frame analysis only

---

## 1.4 Technical Architecture for OOD

```
┌────────────────────────────────────────────────────────────────┐
│                    Image Input (Road Scene)                     │
└────────────────────┬───────────────────────────────────────────┘
                     │
     ┌───────────────┼───────────────┬───────────────┐
     ▼               ▼               ▼               ▼
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ Agent 1  │   │ Agent 2  │   │ Agent 3  │   │ Agent 4  │
│  Scene   │   │ Spatial  │   │ Semantic │   │  Visual  │
│ Context  │   │ Anomaly  │   │Inconsist.│   │Appearance│
│ LLaVA-7B │   │ LLaVA-7B │   │ LLaVA-7B │   │ LLaVA-7B │
└────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
     │               │               │               │
     └───────────────┼───────────────┼───────────────┘
                     │               │
                     ▼               ▼
              ┌─────────────────────────┐
              │      Agent 5            │
              │ Reasoning Synthesizer   │
              │       LLaVA-7B          │
              │                         │
              │ Output: prompt_v1,      │
              │         prompt_v2,      │
              │         confidence      │
              └──────────┬──────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   GroundingDINO     │
              │  (SwinT backbone)   │
              │  Box Detection      │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   CLIP ViT-B/32     │
              │  Semantic Verify    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   SAM ViT-H         │
              │  Pixel Segmentation │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Output:            │
              │  - Bounding Boxes   │
              │  - SAM Masks        │
              │  - Binary OOD Mask  │
              │  - Metrics (if GT)  │
              └─────────────────────┘
```

### Model Specifications

| Model | Architecture | Parameters | VRAM Usage | Purpose |
|-------|-------------|-----------|------------|---------|
| LLaVA-7B | LLaMA-2 + CLIP ViT-L | 7B (4-bit quantized) | ~5 GB | Multi-agent reasoning |
| GroundingDINO | Swin-T + BERT | ~172M | ~700 MB | Open-vocabulary detection |
| SAM | ViT-H | ~636M | ~2.4 GB | Segment Anything |
| CLIP | ViT-B/32 | ~151M | ~300 MB | Semantic verification |

### Memory Management Strategy

LLaVA-7B and the detection models (GDINO + SAM + CLIP) cannot all fit on a single 15GB T4 GPU simultaneously. MAVR-OOD handles this with a **load-free-load** strategy:

1. Load LLaVA-7B → Run all 5 agents → **Free LLaVA from GPU**
2. Load GDINO + SAM + CLIP (these stay loaded from startup)
3. Run detection → segmentation → verification
4. Total peak VRAM: ~8 GB (on A100, comfortable; on T4, tight)

---

# PART 2: Text-Guided Detection — Deep Explanation

---

## 2.1 What Is Text-Guided Detection?

Text-Guided Detection allows a user to describe an object in natural language and the system **locates, verifies, and segments** that specific object in a road scene image.

**Example queries:**
- "the red car on the left"
- "the largest vehicle"
- "the car between the truck and the bus"
- "the second car from the right"
- "the pedestrian near the crosswalk"

---

## 2.2 The 7-Step Text-Guided Pipeline

```
User: "the white car on the right" + Image
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Step 1: Scene Understanding Agent (LLaVA-7B)             │
│   Analyzes scene type, lighting, enumerates all objects   │
│   Output: scene JSON with objects list                    │
├──────────────────────────────────────────────────────────┤
│ Step 2: Attribute Matching Agent (LLaVA-7B)              │
│   Matches user query to scene objects                     │
│   Output: matched objects, recommended prompt, reasoning  │
├──────────────────────────────────────────────────────────┤
│ Step 2.5: Advanced Query Parsing (LLaVA-7B)             │
│   Extracts: object, color, spatial term, anchor object    │
│   Output: structured parse (object_prompt, spatial, etc.) │
├──────────────────────────────────────────────────────────┤
│         *** FREE LLaVA FROM GPU ***                      │
├──────────────────────────────────────────────────────────┤
│ Step 3: Candidate Detection (GroundingDINO)              │
│   Detects all objects matching the prompt                 │
│   Output: N bounding boxes with confidence scores         │
├──────────────────────────────────────────────────────────┤
│ Step 4: CLIP Verification                                │
│   Verifies each candidate matches the query semantically  │
│   Output: passed/rejected for each candidate              │
├──────────────────────────────────────────────────────────┤
│ Step 5: Spatial Filtering                                │
│   Applies spatial logic (left, right, largest, between)   │
│   Output: the single correct object                       │
├──────────────────────────────────────────────────────────┤
│ Step 6: SAM Segmentation                                 │
│   Generates pixel-precise mask for selected object        │
│   Output: segmentation mask overlay                       │
├──────────────────────────────────────────────────────────┤
│ Step 7: Reasoning Agent (LLaVA-7B)                       │
│   Explains why this object was selected                   │
│   Output: natural language explanation                    │
└──────────────────────────────────────────────────────────┘
```

---

## 2.3 Advanced Query Parsing (LLaVA-Based)

The system uses **LLaVA-7B as the query parser** (not rule-based). This means it can understand complex, natural language queries.

### What the Parser Extracts

| Field | Description | Example |
|-------|-------------|---------|
| `object_prompt` | The core object to detect | "white car" |
| `spatial` | Spatial relationship | "right", "left", "largest", "between", "next_to" |
| `anchor` | Reference object for relational queries | "truck" |
| `anchor2` | Second reference for "between" queries | "bus" |
| `ordinal` | Numeric position | 2 (for "second car") |
| `ordinal_direction` | Direction for counting | "from_right" |
| `detect_all` | Whether to detect all matches | false |
| `attribute` | Color/size attributes | "white", "large" |

### Supported Spatial Queries

| Query Type | Example | How It Works |
|-----------|---------|-------------|
| **Directional** | "car on the left" | Compares x-center of all boxes, selects leftmost |
| **Relative size** | "the largest vehicle" | Compares box area, selects maximum |
| **Relational** | "car next to the truck" | Detects anchor (truck), finds nearest car to it |
| **Between** | "car between the truck and bus" | Detects both anchors, finds car with center closest to midpoint |
| **Ordinal** | "second car from the right" | Sorts boxes by x-position, selects 2nd from right |
| **Ahead/Behind** | "car ahead of the truck" | Uses y-coordinate (lower y = further ahead in driving perspective) |

### Retry Logic

If GroundingDINO finds **0 candidates**:
1. **Retry 1**: Lower confidence threshold from 0.35 to 0.20
2. **Retry 2**: Use the raw user prompt instead of parsed prompt
3. **Fallback**: If CLIP rejects all candidates, keep the best-scoring one

This multi-retry approach ensures the system **rarely returns empty results**.

---

## 2.4 Why Multi-Agent Instead of Single Model?

| Approach | Limitation |
|----------|-----------|
| Single detector (YOLO, Faster R-CNN) | Fixed classes, cannot understand "the red car on the left" |
| Single VLM (LLaVA only) | Can describe but cannot produce bounding boxes or masks |
| GroundingDINO alone | No scene understanding, no spatial reasoning, no verification |
| **MAVR-OOD (Multi-Agent)** | Combines reasoning + detection + verification + segmentation |

The multi-agent approach gives us:
- **Interpretability**: each agent's output can be inspected
- **Modularity**: agents can be upgraded independently
- **Robustness**: multiple perspectives reduce single-point failures
- **Flexibility**: handles both OOD and text-guided tasks with the same models

---

# PART 3: Final Review Presentation Guide

---

## 3.1 Presentation Structure (Two Presenters)

### Suggested Division of Responsibilities

#### Presenter 1 (Person A): Problem, Architecture, OOD Detection
- Slides 1–8 (approximately 12–15 minutes)

#### Presenter 2 (Person B): Text-Guided Detection, Demo, Results, Conclusion
- Slides 9–16 (approximately 12–15 minutes)

---

## 3.2 Slide-by-Slide Content

### SLIDE 1: Title Slide (Presenter 1)

**Title:** Multi-Agent Vision-Language Reasoning for Reliable Object Localization in Road Environments

**Subtitle:** MAVR-OOD: A Confidence-Aware Multi-Agent System

**Names:** [Your Names]
**Guide:** [Mentor Name]
**Department / University**
**Date**

---

### SLIDE 2: Problem Statement (Presenter 1)

**Key Points to Say:**

"Autonomous driving systems need to detect not just known objects like cars and pedestrians, but also **unexpected objects** — objects that were never seen during training. A cow on a highway, debris on a road, or a shopping cart in a lane. These are called **Out-of-Distribution objects**, and they are the leading cause of edge-case failures in self-driving systems."

"Existing object detectors like YOLO or Faster R-CNN are **closed-vocabulary** — they can only detect the classes they were trained on. If a zebra appears on a road, they have no category for it."

"Similarly, when a human operator needs to identify a **specific object** in a scene — like 'the red car on the left' — current systems cannot handle this natural language instruction."

**Bullet Points:**
- Traditional detectors fail on novel/unexpected objects
- No mechanism for spatial reasoning ("left car", "car between two trucks")
- No interpretability — detectors give boxes, not explanations
- MAVR-OOD solves all three with a multi-agent VLM system

---

### SLIDE 3: Proposed Solution Overview (Presenter 1)

**Key Points to Say:**

"We propose MAVR-OOD, a system that combines **four specialized AI models** in a multi-agent architecture. Instead of training a single model for everything, we assign specific roles to each model and let them collaborate."

**Show Architecture Diagram (from the repo)**

**Bullet Points:**
- 5 LLaVA-7B agents for reasoning (OOD pipeline)
- 7-step pipeline for text-guided detection
- GroundingDINO for open-vocabulary detection
- CLIP for semantic verification
- SAM for pixel-level segmentation
- No training required — all models used in zero-shot / inference-only mode

---

### SLIDE 4: Model Stack (Presenter 1)

**Create a table slide:**

| Model | Role | Why This Model |
|-------|------|---------------|
| LLaVA-7B (4-bit) | Scene understanding, reasoning | Strong multimodal reasoning, runs on consumer GPUs |
| GroundingDINO (SwinT) | Open-vocabulary object detection | Can detect ANY object described in text |
| CLIP ViT-B/32 | Semantic verification | Cross-checks detection against query |
| SAM ViT-H | Segmentation | State-of-the-art zero-shot segmentation |

**Key Point to Say:**

"Critically, none of these models are trained by us. We use them in a **zero-shot, inference-only** pipeline. This means our system can detect objects it has never been explicitly trained on — including completely novel anomalies."

---

### SLIDE 5: OOD Detection — The 5-Agent Pipeline (Presenter 1)

**Show the agent flow diagram**

**Key Points to Say:**

"For OOD detection, five LLaVA-powered agents examine the road scene simultaneously, each from a different angle."

"Agent 1 establishes the baseline — what **should** be in this scene. Agent 2 checks spatial consistency — is anything in a wrong position? Agent 3 performs the core anomaly check — does every object **belong** on a road? Agent 4 looks for visual oddities — unusual colors, damaged structures. Finally, Agent 5 synthesizes all findings and generates a detection prompt."

"This prompt is then passed to GroundingDINO, which actually locates the anomaly in the image, followed by CLIP verification and SAM segmentation."

---

### SLIDE 6: OOD — Types of Anomalies Detected (Presenter 1)

**Table:**

| Category | Examples | Detection Confidence |
|----------|----------|---------------------|
| Animals on roads | Cow, zebra, dog, deer | HIGH |
| Misplaced vehicles | Overturned truck, wrong-way car | MEDIUM-HIGH |
| Road debris | Fallen tree, furniture, tire debris | MEDIUM |
| Pedestrian anomalies | Person lying on road | MEDIUM |
| Infrastructure damage | Missing guardrail, collapsed section | LOWER |

**Key Point to Say:**

"Our system detects anomalies through reasoning, not pattern matching. This means it can identify objects it has never seen before, as long as they violate the contextual norms of a road environment."

---

### SLIDE 7: OOD — Sample Results (Presenter 1)

**Show 2-3 example images:**
1. Zebras on road → detected and segmented
2. Cow on highway → binary OOD mask generated
3. Debris on road → localized with bounding box

**For each, show:**
- Original image
- Bounding box detection
- SAM segmentation mask
- Binary OOD mask (pink overlay)

---

### SLIDE 8: GPU Memory Management (Presenter 1)

**Key Point to Say:**

"A key engineering challenge was fitting 7B-parameter LLaVA alongside three other large models on a single GPU. We solved this with a load-free-load strategy: LLaVA runs first for reasoning, then is completely freed from GPU memory before the detection models run."

**Show memory timeline:**
```
Time →
[LLaVA 5GB]──────────[FREE]──[GDINO 700MB + SAM 2.4GB + CLIP 300MB]
     Agent 1-5                     Detection → Verification → Segmentation
```

**Transition:** "Now [Presenter 2] will walk through the text-guided detection system and demonstrate the live web interface."

---

### SLIDE 9: Text-Guided Detection — Overview (Presenter 2)

**Key Points to Say:**

"The second major capability of MAVR-OOD is **text-guided detection**. A user provides a natural language query — for example, 'the red car on the left' — and the system identifies, verifies, and segments that specific object."

"This is a 7-step pipeline that combines language understanding, visual grounding, semantic verification, spatial reasoning, and segmentation."

---

### SLIDE 10: The 7-Step Pipeline (Presenter 2)

**Show pipeline diagram with the 7 steps**

**Walk through each step:**

1. **Scene Agent**: LLaVA analyzes the scene, lists all visible objects with positions and colors
2. **Attribute Agent**: LLaVA matches the user's query to specific objects in the scene
3. **Query Parser**: LLaVA extracts structured information (object type, spatial term, anchor object)
4. **GroundingDINO**: Detects all candidate objects matching the parsed prompt
5. **CLIP Verification**: Verifies each candidate semantically matches the query
6. **Spatial Filter**: Applies spatial logic (left, right, nearest, largest, between, ordinal)
7. **SAM + Reasoning**: Segments the final object and explains the decision

---

### SLIDE 11: Advanced Query Understanding (Presenter 2)

**Key Points to Say:**

"Our system uses LLaVA-7B as the query parser — not simple keyword matching. This means it can understand complex spatial instructions."

**Show examples:**

| User Query | System Understanding |
|-----------|---------------------|
| "the red car on the left" | object=red car, spatial=left |
| "largest vehicle" | object=vehicle, spatial=largest |
| "car between the truck and bus" | object=car, spatial=between, anchor=truck, anchor2=bus |
| "second car from the right" | object=car, ordinal=2, direction=from_right |
| "car next to the pedestrian" | object=car, spatial=next_to, anchor=pedestrian |

"This level of natural language understanding is what separates MAVR from traditional detection systems."

---

### SLIDE 12: CLIP Verification — Why It Matters (Presenter 2)

**Key Points to Say:**

"GroundingDINO sometimes returns false positives — it might detect a truck when the user asked for a car. CLIP acts as a second opinion. It crops each detected region and computes a similarity score between the cropped image and the text query. Only detections with high semantic similarity pass through."

**Show example:**
```
Query: "white car"
Candidate 1: truck (CLIP=0.18) → REJECTED
Candidate 2: white car (CLIP=0.72) → PASSED
Candidate 3: grey car (CLIP=0.31) → REJECTED
```

---

### SLIDE 13: Live Demo (Presenter 2)

**Show the web interface running on Colab**

**Demo flow:**
1. Upload a road scene image
2. Type: "the white car on the right"
3. Show the pipeline progress animation
4. Show detection result with image comparison slider
5. Show step-by-step visualization
6. Show the reasoning agent's explanation

**Alternative: Show OOD Demo**
1. Switch to OOD tab
2. Upload a road scene with an anomaly
3. Show detection, SAM masks, binary OOD mask
4. Show agent analysis cards

---

### SLIDE 14: System Comparison (Presenter 2)

**Table comparing MAVR-OOD vs alternatives:**

| Feature | YOLO/RCNN | GroundingDINO Only | MAVR-OOD |
|---------|-----------|-------------------|----------|
| Open vocabulary | No | Yes | Yes |
| Natural language queries | No | Limited | Full |
| Spatial reasoning | No | No | Yes |
| OOD detection | No | No | Yes |
| Semantic verification | No | No | Yes (CLIP) |
| Pixel segmentation | No | No | Yes (SAM) |
| Explainability | None | None | Full (reasoning agent) |
| Multi-agent reasoning | No | No | Yes (5 agents) |

---

### SLIDE 15: Technical Challenges Solved (Presenter 2)

**Bullet Points:**

1. **GPU Memory**: 7B LLaVA + 3 detection models on single GPU → load-free-load strategy
2. **LLaVA JSON Parsing**: LLaVA often returns malformed JSON → 5-strategy robust parser (direct, code block, brace matching, cleanup, truncation recovery)
3. **False Positive Reduction**: GroundingDINO over-detects → CLIP verification + spatial filtering
4. **Zero Candidate Recovery**: When nothing is detected → retry with lower threshold → retry with raw prompt → keep best CLIP match
5. **Python 3.12 Compatibility**: GroundingDINO CUDA build fails → manual `setup.py build_ext --inplace` workaround
6. **Web Interface**: Production-grade FastAPI web UI with glassmorphism design, real-time progress tracking

---

### SLIDE 16: Conclusion & Future Work (Presenter 2)

**Conclusion Points:**

"MAVR-OOD demonstrates that multi-agent vision-language reasoning can successfully:
- Detect **novel, unseen anomalies** on roads without any training
- Understand **complex natural language queries** with spatial relationships
- Provide **interpretable explanations** for every detection decision
- Deliver results through a **professional web interface** suitable for real-world use"

**Future Work:**
- LoRA fine-tuning of LLaVA on driving-domain data (50 BDD images prepared)
- Video-level OOD detection (temporal reasoning across frames)
- Edge deployment optimization (model distillation for on-device use)
- Integration with autonomous driving simulation environments

---

## 3.3 Handling Mentor Questions

### "How does the system know what is an anomaly?"

"The system uses contextual reasoning, not a fixed list. Agent 1 establishes what is 'normal' for the scene type and environmental conditions. Agents 2, 3, and 4 then independently check for spatial, semantic, and visual violations. If multiple agents agree that an object is out of place, Agent 5 generates a high-confidence detection prompt. This approach means the system can detect anomalies it has never been trained on."

### "What types of anomalies can it detect vs. cannot detect?"

"It detects animals on roads (high confidence), misplaced vehicles, road debris, and infrastructure damage. It cannot reliably detect very small objects (< 20 pixels), objects that perfectly blend with the road surface, or temporal anomalies that require video context."

### "Why not just train a detector?"

"Training requires labeled data for every possible anomaly class. By definition, OOD objects are objects you didn't anticipate during training. Our zero-shot approach using vision-language models can reason about ANY object, including ones never seen before."

### "What is the role of CLIP in the pipeline?"

"CLIP acts as a semantic safety net. GroundingDINO may return false positives — it might detect a truck when we asked for a car. CLIP crops each detection, computes text-image similarity, and rejects detections that don't match. This significantly reduces false positives."

### "How do you handle GPU memory with so many models?"

"LLaVA-7B (the largest model) is loaded first, runs all reasoning tasks, then is completely deleted from GPU memory. Only then do the detection models (GroundingDINO, SAM, CLIP) run. This sequential load-free-load approach keeps peak memory under 8 GB."

### "Is there any training involved?"

"No. All models are used in zero-shot inference mode. LLaVA-7B is pre-trained by the LLaVA team. GroundingDINO, CLIP, and SAM are pre-trained by their respective research groups. Our contribution is the multi-agent architecture and orchestration pipeline."

### "What is the advantage of the web interface?"

"It makes the system accessible to non-technical users. A reviewer, traffic engineer, or safety inspector can upload a road image and get instant OOD analysis without writing any code. The glassmorphism UI with real-time pipeline progress provides a professional demonstration platform."

---

## 3.4 Key Numbers to Remember

| Metric | Value |
|--------|-------|
| Total models used | 4 (LLaVA, GDINO, CLIP, SAM) |
| OOD agents | 5 |
| Text-guided pipeline steps | 7 |
| LLaVA parameters | 7B (4-bit quantized) |
| Total model weights | ~8.5 GB |
| Typical detection time | 100–230 seconds |
| GPU requirement | A100 40GB (recommended) or T4 16GB (tight) |
| Spatial query types supported | 8+ (left, right, largest, nearest, between, ordinal, ahead, behind) |
| JSON parsing strategies | 5 (direct, code block, brace match, cleanup, truncation recovery) |
| Zero training required | Yes — fully zero-shot |

---

## 3.5 Presentation Tips

1. **Start strong**: Open with a dramatic example — zebra on a highway → "Can your YOLO detect this?"
2. **Show the web interface**: A live demo is worth 100 slides
3. **Emphasize zero-shot**: Reviewers love this — no training, generalizes to anything
4. **Prepare backup screenshots**: In case Colab/network fails during demo
5. **Know the limitations**: Being honest about what it cannot do shows maturity
6. **Practice transitions**: Smooth handoff between presenters at Slide 8→9
7. **Time management**: Keep each slide under 2 minutes; save 5 minutes for questions

---

*This document was prepared for the MAVR-OOD project final review.*
