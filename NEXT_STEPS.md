# MAVR-OOD Next Steps

## Goal

Turn MAVR-OOD from a strong prototype into a stronger research-quality system with clearer evidence, tighter evaluation, and better reproducibility.

---

## Phase 1: Stabilize the Current System

- [ ] Keep one stable end-to-end pipeline working consistently
- [ ] Avoid adding new features until the current evaluation path is stable
- [ ] Confirm `improvement/web-ui` remains the active working branch
- [ ] Make sure the text-guided path works end to end
- [ ] Make sure the OOD path works end to end
- [ ] Verify the current UI changes are stable in local and Colab runs

### Output of this phase

- One branch that runs reliably
- No confusion about which app entrypoint is the recommended one

---

## Phase 2: Define the Exact Research Claim

- [ ] Write a single-sentence project claim
- [ ] Decide whether the main contribution is:
  - better anomaly reasoning
  - better prompt generation
  - better localization reliability
  - better explainability
- [ ] Make sure README, docs, and presentation all use the same framing

### Candidate claim

`MAVR improves reliable road-scene anomaly localization by using multi-agent vision-language reasoning to generate better grounding prompts, better candidate filtering, and more interpretable outputs.`

---

## Phase 3: Build Baselines

- [ ] Implement or finalize a GroundingDINO-only baseline
- [ ] Implement or finalize a GroundingDINO + SAM baseline
- [ ] Implement or finalize a GroundingDINO + CLIP + SAM baseline
- [ ] Compare all of them against the full MAVR pipeline
- [ ] Save results in a clean, reproducible format

### Metrics to compare

- [ ] IoU
- [ ] F1
- [ ] Precision
- [ ] Recall
- [ ] success rate
- [ ] runtime per image

### Output of this phase

- A baseline comparison table
- A clear answer to whether MAVR beats simpler pipelines

---

## Phase 4: Run Ablation Studies

- [ ] Evaluate the full pipeline
- [ ] Remove CLIP verification and test again
- [ ] Remove spatial filtering and test again
- [ ] Remove attribute matching and test again
- [ ] Replace multi-agent prompt synthesis with a simpler prompt and test again
- [ ] Measure how much each component changes downstream localization

### Key question

Does each component provide measurable value, or is the system more complex than necessary?

### Output of this phase

- An ablation table
- A stronger defense against reviewer criticism

---

## Phase 5: Build a Clean Evaluation Set

- [ ] Separate train, validation, and test data clearly
- [ ] Keep a held-out set for final reporting
- [ ] Include both easy and hard cases
- [ ] Include ambiguous and failure-prone road scenes
- [ ] Do not rely only on cherry-picked examples

### Evaluation categories to include

- [ ] simple scenes
- [ ] crowded scenes
- [ ] hard lighting
- [ ] small objects
- [ ] ambiguous user queries
- [ ] unusual road anomalies

---

## Phase 6: Document Failure Analysis

- [ ] Create a small failure log
- [ ] For each failure, record:
  - input image
  - user query or anomaly type
  - scene conditions
  - which component failed
  - likely reason
- [ ] Separate failures into:
  - VLM reasoning failures
  - prompt generation failures
  - GroundingDINO failures
  - CLIP filtering failures
  - SAM segmentation failures
  - spatial filter failures

### Output of this phase

- A short but honest failure analysis section for reports or papers

---

## Phase 7: Reduce Code Duplication

- [ ] Review overlap between `app.py`, `streamlit_app.py`, and `web_app.py`
- [ ] Move reusable backend logic into shared modules where practical
- [ ] Reduce duplicated OOD helper code
- [ ] Reduce duplicated visualization code where possible
- [ ] Keep UI-specific code separate from core pipeline logic

### Goal

Make the project look more mature and easier to maintain.

---

## Phase 8: Fine-Tuning Decision

- [ ] Do not start fine-tuning before baseline and ablation results are stable
- [ ] Decide which LLaVA task to tune first:
  - scene understanding
  - attribute matching
  - anomaly reasoning
  - JSON output consistency
- [ ] Use LoRA or QLoRA, not full fine-tuning
- [ ] Prepare high-quality instruction-format data
- [ ] Use a validation split
- [ ] Compare base vs fine-tuned LLaVA fairly

### Fine-tuning objective

Improve the parts of MAVR that truly depend on LLaVA, not the parts controlled by GroundingDINO or SAM.

---

## Phase 9: Presentation and Reporting

- [ ] Update README to match the actual current branch behavior
- [ ] Clean up documentation drift
- [ ] Create one architecture figure that matches the implemented system
- [ ] Prepare:
  - baseline table
  - ablation table
  - qualitative results
  - failure cases
  - limitations
- [ ] Keep claims precise and evidence-based

### Important rule

Do not claim improvement just because the outputs look more intelligent. Claim improvement only when downstream localization or reliability metrics support it.

---

## Suggested 2-Week Priority Order

### Week 1

- [ ] stabilize pipeline
- [ ] define exact claim
- [ ] build baselines
- [ ] prepare evaluation set
- [ ] start ablation experiments

### Week 2

- [ ] analyze results
- [ ] document failures
- [ ] reduce code duplication where needed
- [ ] decide whether fine-tuning is justified
- [ ] prepare evidence package for report or presentation

---

## Success Criteria

This project becomes much stronger if the following are true:

- [ ] one stable branch runs reliably
- [ ] the contribution is clearly stated
- [ ] MAVR is compared against simpler baselines
- [ ] each major module has ablation evidence
- [ ] failure cases are documented honestly
- [ ] fine-tuning is treated as a measured experiment, not assumed improvement

---

## Notes

- Complexity alone is not a contribution
- Interpretability alone is not enough
- The strongest version of MAVR is one that proves its extra reasoning stages improve reliability in measurable ways
