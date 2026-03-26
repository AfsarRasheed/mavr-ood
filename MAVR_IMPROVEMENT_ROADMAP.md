# MAVR Project — Improvement Roadmap

> Based on review feedback and project analysis (March 2026)

---

## Review Feedback Summary

| Feedback | Concern |
|----------|---------|
| Project looks simple | Only tested with 2 cars, easy images |
| Limited query types | Only left/right/center keywords |
| No training | "Just combined models, what's your contribution?" |
| Will it scale? | Does it work with multiple objects in complex scenes? |
| Scope unclear | Where can this be deployed? |

---

## Honest Project Assessment

### Strengths
- Working end-to-end demo (upload → detect → segment → explain)
- 7-step pipeline with step-by-step visualizations
- Reasoning Agent provides explainability (rare in detection systems)
- 6 fallback mechanisms — pipeline never crashes
- Open vocabulary — works with ANY object description

### Weaknesses
- LLaVA-7B is unreliable (needs fallbacks to compensate)
- GroundingDINO does ~80% of the work — pipeline is heavily dependent on it
- Rule-based query parser limited to ~10 spatial keywords
- Tested on only 13 easy/curated images
- CLIP threshold (0.25) chosen manually, no data-driven justification
- Slow (~30-60s per image due to double LLaVA loading)
- No training = weaker academic contribution

---

## Your Contribution (How to Defend)

> "Our contribution is the multi-agent reasoning architecture, not model training."

| Contribution | Details |
|-------------|---------|
| Pipeline architecture | 7-step orchestration deciding which model runs when |
| Prompt engineering | Crafted prompts for 3 LLaVA agents |
| Query parser + spatial logic | Custom NLP parser + spatial filtering (doesn't exist in any pre-trained model) |
| CLIP verification layer | Semantic gate that reduces false positives |
| Fallback design | 6 mechanisms ensuring robustness |
| Explainable AI | Reasoning Agent generates human-readable justifications |
| Memory management | GPU VRAM optimization for single-GPU deployment |

---

## Use Cases

| Domain | Application |
|--------|------------|
| Traffic Control Centers | Operators search CCTV feeds via text descriptions |
| Autonomous Driving | Test perception with spatial + attribute queries |
| Accident Investigation | Auto-segment specific vehicles from scene photos |
| Road Anomaly Detection | Detect animals, debris, stalled vehicles |
| Smart City Surveillance | Find illegally parked vehicles via description |
| Assistive Navigation | Help visually impaired users locate objects |

---

## Improvement Roadmap

### Phase 1: Prove What You Have *(Experiments)*

#### 1A. Ablation Study ← **MOST IMPORTANT**
Remove each component, measure impact:

| Variant | What's removed | Purpose |
|---------|---------------|---------|
| Full MAVR | Nothing | Reference |
| No CLIP | CLIP verification | Proves CLIP reduces false positives |
| No Spatial | Spatial filter | Proves spatial reasoning matters |
| No LLaVA | Scene + Attribute agents | Proves VLM improves prompt quality |
| No Reasoning | Reasoning agent | Shows explainability contribution |
| GDino-only | Everything except GDino+SAM | Baseline |

#### 1B. Threshold Analysis
- Sweep CLIP (0.05–0.50) and Box (0.10–0.50) thresholds
- Generate heatmap of IoU vs threshold pairs
- Find optimal values with data, not intuition

#### 1C. Query Complexity Study

| Level | Query Example | Tests |
|-------|--------------|-------|
| Easy | "the car" | Basic detection |
| Medium | "the red car" | Attribute matching |
| Hard | "the red car on the left" | Spatial reasoning |
| Relational | "the car next to the truck" | Multi-object reasoning |
| Extreme | "the second car from the right" | Ordinal understanding |

---

### Phase 2: Scale Evaluation

#### 2A. Expand Dataset
- Current: 13 images (only animals on roads)
- Target: **50-100 images** (urban, highway, night, rain, crowded)
- Sources: BDD100K, Cityscapes, nuScenes, or manually collected
- Create GT masks using [makesense.ai](https://www.makesense.ai/)

#### 2B. Cross-Dataset Testing
- Tune thresholds on one dataset, test on another
- Proves generalization

---

### Phase 3: Compare Against State-of-the-Art

| Method | Type |
|--------|------|
| GroundingDINO + SAM | Your direct baseline |
| Grounded-SAM | Official GDino+SAM pipeline |
| GLIP | Microsoft's grounded pre-training |
| OWL-ViT | Google's open-vocabulary detector |
| LISA | LLM-based reasoning segmentation |

Metrics: IoU, F1, mAP, FPR, Latency

---

### Phase 4: Make the System Smarter

| Improvement | What it does |
|------------|-------------|
| LLaVA-based query parser | Replace rule-based parser → any natural language works |
| Multi-object return | "find all red cars" returns multiple objects |
| Confidence scoring | Combined GDino + CLIP score per detection |
| Adaptive thresholds | Auto-adjust based on scene complexity |
| Better VLM | Upgrade to LLaVA-13B or InternVL |

---

### Phase 5: Paper Structure

```
1. Introduction          — Problem + motivation
2. Related Work          — GDino, SAM, CLIP, VLMs, referring segmentation
3. Methodology           — 7-step pipeline architecture
4. Experiments
   4.1 Dataset           — Description + statistics
   4.2 Baselines         — GDino-only, Grounded-SAM, GLIP, etc.
   4.3 Ablation Study    — Remove each component
   4.4 Threshold Analysis — Sensitivity curves
   4.5 Query Complexity   — Easy → extreme
5. Results               — Tables + charts
6. Discussion            — Strengths, limitations, failure cases
7. Conclusion + Future   — Video, real-time, fine-tuning
```

---

## Implementation Priority

| # | Task | Effort | Impact | Script |
|---|------|--------|--------|--------|
| 1 | Baseline comparison | ✅ Done | High | `run_baseline_comparison.py` |
| 2 | Ablation study | 1 day | **Highest** | `run_ablation_study.py` |
| 3 | Add 30+ test images | 2-3 days | High | Manual + makesense.ai |
| 4 | Threshold sweep | 1 day | Medium | `run_threshold_sweep.py` |
| 5 | Query complexity analysis | 1 day | Medium | `run_query_complexity.py` |
| 6 | Compare with GLIP/OWL-ViT | 2 days | Medium | Custom eval scripts |
| 7 | LLaVA-based query parser | 1 day | Medium | Modify `query_parser.py` |
| 8 | Multi-object return | 1 day | Medium | Modify `pipeline.py` |
| 9 | Confidence scoring | 0.5 day | Low | Modify `pipeline.py` |

---

## Natural Language Improvement Levels

| Level | Current | Improved |
|-------|---------|----------|
| 1 | "left/right/center" | "between the truck and bus", "second from left" |
| 2 | Rule-based parser | LLaVA parses ANY query into structured JSON |
| 3 | Single turn only | Conversational: "now find the one next to it" |
| 4 | Appearance only | Behavior: "the car that appears to be speeding" |
