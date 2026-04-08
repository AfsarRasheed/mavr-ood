# Florence-2 Text-Guided Branch Plan

## Purpose

This document proposes a **new experimental branch** for integrating **Florence-2** into the project **only for the text-guided pipeline**.

This is **not** a replacement plan for the entire project.

The scope is:

- upgrade the **text-guided grounding pipeline**
- keep the **OOD pipeline unchanged**
- preserve the existing web UI structure so that:
  - text-guided can evolve
  - OOD still works as it currently does

This branch should be treated as a **carefully isolated experiment**.

Recommended branch name:

- `experiment/florence2-text-guided`

or

- `improvement/florence2-grounding`

---

# 1. Why Consider Florence-2

## Current issue in the text-guided branch

The present `improvement/natural-language-grounding` branch already improves:

- query parsing
- candidate ranking
- confidence handling
- semantic reasoning

But it still relies on a key candidate proposal bottleneck:

- `GroundingDINO`

GroundingDINO is effective for:

- short object prompts
- simple object + attribute prompts

But it is weaker for richer expressions such as:

- `the burning black car on the right`
- `the damaged white vehicle near the pole`
- `the second car from the right behind the truck`

This causes the pipeline to simplify prompts aggressively so that detection still works.

That simplification harms the project’s goal:

> natural language in, correct grounded object out

## Why Florence-2 is worth exploring

Florence-2 is attractive because it is designed as a more **general prompt-driven vision model**.

Instead of behaving mainly like a detector that prefers short object phrases, it is more naturally aligned with:

- instruction-like prompts
- prompt-based visual tasks
- richer vision-language interactions

That makes it a strong experimental candidate for:

- more natural text-guided grounding
- better prompt-following behavior
- reduced need for aggressive prompt simplification

---

# 2. What Florence-2 Is

## Conceptually

Florence-2 is a **general-purpose vision foundation model** from Microsoft that supports multiple prompt-based tasks.

It is not just:

- an object detector

It is more like:

- a unified vision-language model that can perform different visual tasks based on the text instruction

## Why this matters for this project

The current text-guided pipeline needs stronger support for:

- understanding richer user prompts
- generating candidate regions that match more than a simple noun phrase
- reducing the gap between text semantics and object localization

Florence-2 is relevant because it can potentially help bridge that gap better than GroundingDINO.

---

# 3. What Florence-2 Should Do In This Project

## It should NOT replace everything

Florence-2 should **not** be forced to replace all existing components.

That would create unnecessary risk.

The better approach is:

- replace or augment the **text-guided candidate proposal / grounding stage**
- keep the rest of the strong pipeline pieces where useful

## Best role for Florence-2 in this project

Florence-2 should be used for:

1. richer text-conditioned candidate proposal
2. better phrase grounding support for text-guided prompts
3. potentially stronger early localization for natural-language object requests

Florence-2 should **not** automatically replace:

- SAM segmentation
- the OOD pipeline
- the full UI
- reliability logic

---

# 4. Scope Of This New Branch

## In scope

- text-guided detection only
- candidate proposal / grounding for text-guided queries
- API changes only where needed for text-guided behavior
- UI updates only if necessary for the text-guided path

## Out of scope

- OOD pipeline redesign
- OOD model replacement
- OOD logic changes
- full UI redesign for both tabs

OOD must continue to work exactly as it does now.

---

# 5. High-Level Architecture Change

## Current text-guided architecture

Current flow:

1. Scene Understanding Agent (`LLaVA`)
2. Attribute Matching Agent (`LLaVA`)
3. Query Parsing / Semantic Control
4. `GroundingDINO` candidate proposal
5. `CLIP` semantic verification
6. Spatial / relation scoring
7. Reliability decision
8. `SAM` segmentation
9. Reasoning output

## Proposed Florence-2 text-guided architecture

New flow:

1. Scene Understanding Agent (`LLaVA`)
2. Attribute Matching Agent (`LLaVA`)
3. Query Parsing / Semantic Control
4. `Florence-2` text-guided grounding / candidate proposal
5. Optional candidate normalization / conversion layer
6. `CLIP` semantic verification
7. Semantic candidate judging
8. Spatial / relation support
9. Reliability decision
10. `SAM` segmentation
11. Reasoning output

## Key idea

The major architectural change is:

- `GroundingDINO` is replaced or bypassed only in the **text-guided grounding stage**

Everything else remains conceptually similar.

---

# 6. Recommended Integration Strategy

## Recommended strategy: add Florence-2 as an alternative backend

This is the safest approach.

Instead of immediately deleting GroundingDINO from the text-guided path, the new branch should support:

- `TEXT_GUIDED_BACKEND=gdino`
- `TEXT_GUIDED_BACKEND=florence2`

This gives:

1. safer experimentation
2. easier comparison
3. easier rollback
4. better research value

## Why not a hard replacement first

If Florence-2 is wired in as the only text-guided backend immediately:

- debugging becomes harder
- comparison becomes harder
- risk increases

So the right initial branch strategy is:

- support Florence-2 as a configurable text-guided backend

---

# 7. Proposed New Text-Guided Architecture

## 7.1 Query Understanding Layer

Keep:

- `scene_agent.py`
- `attribute_agent.py`
- `query_parser.py`
- `semantic_controller.py`

These still matter because Florence-2 does not remove the need for:

- semantic constraint extraction
- mandatory vs preferred constraints
- reliability-aware control

## 7.2 Candidate Proposal Layer

Replace in text-guided only:

- `GroundingDINO` candidate proposal

With:

- `Florence-2 grounding backend`

New module to add:

- `src/text_guided/florence2_backend.py`

Responsibilities:

- load Florence-2 model and processor
- run prompt-based grounding or detection for a user query
- produce candidate boxes in a unified format

## 7.3 Candidate Normalization Layer

New helper module:

- `src/text_guided/candidate_adapter.py`

Responsibilities:

- convert Florence-2 outputs into the same candidate format used by the pipeline
- keep compatibility with:
  - CLIP verifier
  - semantic judge
  - SAM
  - UI summaries

## 7.4 Semantic Verification Layer

Keep:

- `clip_verifier.py`
- `candidate_judge.py`
- `semantic_controller.py`

This remains useful because even if Florence-2 gives better grounding proposals, we still need:

- semantic verification
- condition enforcement
- reliability logic

## 7.5 Segmentation Layer

Keep:

- `SAM`

No need to replace this for the Florence-2 text-guided branch.

## 7.6 Reasoning / Explanation Layer

Keep:

- `reasoning_agent.py`

But update explanation wording so it reflects:

- Florence-2 as the proposal backend

instead of:

- GroundingDINO

---

# 8. What Files Will Change

## New files likely needed

1. `src/text_guided/florence2_backend.py`
- Florence-2 model loading and inference

2. `src/text_guided/candidate_adapter.py`
- unify Florence-2 outputs with the current pipeline format

3. optionally `src/model_loader_florence2.py`
- if model loading should be separated from the text-guided package

## Existing files that will likely change

1. `src/text_guided/pipeline.py`
- switch or route text-guided candidate proposal to Florence-2

2. `src/model_loader.py`
- optionally add Florence-2 loader

3. `web_app.py`
- load Florence-2 backend for text-guided path
- leave OOD loading untouched

4. `src/text_guided/reasoning_agent.py`
- update backend references in explanation

5. `src/text_guided/__init__.py`
- export Florence-2 text-guided backend if needed

## Files that should stay unchanged or mostly unchanged

- OOD agent files
- OOD pipeline files
- OOD evaluation logic
- OOD UI logic

---

# 9. How The Text-Guided Pipeline Will Change

## Current text-guided behavior

Current issue:

- prompt often gets simplified for GroundingDINO
- richer meaning is enforced only later

## Florence-2 text-guided behavior

Target behavior:

1. user enters natural-language prompt
2. semantic controller extracts constraints
3. Florence-2 receives a richer grounding prompt
4. Florence-2 produces candidate regions more aligned with the natural phrase
5. candidate judge and CLIP still verify / rerank
6. reliability module decides final trust state

This should reduce:

- over-simplification of prompt meaning
- mismatch between prompt and candidate proposal

---

# 10. What Should Stay The Same In The UI

The UI should still:

- keep the two-tab structure
  - Text-Guided Detection
  - OOD Detection
- preserve the OOD tab exactly as it currently works
- preserve the current text-guided result flow where possible

This is important because:

- users should not lose the current OOD functionality
- the Florence-2 branch should be an upgrade, not a disruption

---

# 11. What Should Change In The Text-Guided UI

Only minimal, necessary changes should be made first.

## Recommended changes

### 1. Backend label

In the text-guided pipeline display, replace or adjust:

- `GroundingDINO`

with:

- `Florence-2`

or

- `Candidate Grounding`

depending on whether you want the UI to be model-specific.

### 2. Summary / decision panel

Add a field like:

- `Grounding Backend: Florence-2`

This makes testing and comparison much easier.

### 3. Keep the rest of the UI stable

Do **not** redesign the entire UI at the same time.

At first, the UI should mainly:

- preserve familiarity
- allow easy comparison with the GroundingDINO branch

---

# 12. How OOD Should Behave

## OOD must remain unchanged

This is extremely important.

The Florence-2 branch is being introduced **only for text-guided detection**.

That means:

- OOD should still use its current pipeline
- OOD models should still load as before
- OOD UI should still behave as before

No OOD logic should be touched unless strictly necessary for shared infrastructure.

---

# 13. Runtime / Dependency Changes

## Current Colab flow

Your current Colab flow includes:

- cloning repo
- checking out `improvement/natural-language-grounding`
- installing requirements
- building GroundingDINO
- installing SAM and CLIP
- downloading GroundingDINO and SAM weights
- starting `web_app.py`

## What changes in Florence-2 branch

### Likely no longer required for text-guided path

If Florence-2 fully replaces GroundingDINO only for text-guided:

- GroundingDINO may no longer be required for the text-guided pipeline

But if OOD still uses GroundingDINO:

- GroundingDINO installation may still be required overall

So because OOD must remain unchanged, the safe answer is:

- GroundingDINO setup likely still remains in Colab unless OOD is decoupled from it

### New requirements likely needed

Depending on Florence-2 implementation:

- `transformers`
- `accelerate`
- `sentencepiece` if required
- possibly additional Hugging Face dependencies

### New model download/loading requirements

The Florence-2 model weights may need:

- Hugging Face model download
- possibly login/token depending on access requirements

---

# 14. What Changes Are Needed In The Colab Run Code

## Current branch checkout

Current:

```python
!git clone https://github.com/AfsarRasheed/mavr-ood.git
%cd /content/mavr-ood
!git checkout improvement/natural-language-grounding
```

### New Florence-2 branch checkout

Would become:

```python
!git clone https://github.com/AfsarRasheed/mavr-ood.git
%cd /content/mavr-ood
!git checkout experiment/florence2-text-guided
```

or whatever final Florence-2 branch name is chosen.

## Dependency install changes

Current install includes:

```python
!pip install -q -r requirements.txt
!pip install -q addict yapf fastapi uvicorn python-multipart pyngrok
```

Possible Florence-2 additions:

```python
!pip install -q transformers accelerate sentencepiece
```

If bitsandbytes or quantized loading is needed, that may also be added.

## GroundingDINO build step

Because OOD still depends on GroundingDINO, this likely stays:

```python
%cd /content/mavr-ood/GroundingDINO
!python setup.py build_ext --inplace
%cd /content/mavr-ood
```

## Weights section

Current text-guided weights:

- GroundingDINO
- SAM

In Florence-2 branch:

- SAM still needed
- GroundingDINO still may be needed because OOD is unchanged
- Florence-2 model weights may be pulled automatically through Hugging Face instead of `wget`

## Web server start

Likely unchanged:

```python
!nohup python web_app.py > server.log 2>&1 &
import time; time.sleep(30)
from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(8501)"))
```

This should remain stable if:

- `web_app.py` continues serving both text-guided and OOD routes

---

# 15. What Exact Code Areas Need Changes For Running

To run Florence-2 text-guided while keeping OOD unchanged, these implementation-level changes are likely required:

## In backend loading

- add Florence-2 model loading without removing OOD model loading
- either:
  - extend `load_all_models()` in `web_app.py`
  - or extend `src/model_loader.py`

## In text-guided execution

- modify `run_text_guided_pipeline(...)` to call Florence-2 backend instead of GroundingDINO when the Florence backend is selected

## In API response

- keep output schema as stable as possible
- only add:
  - `grounding_backend`
  - optional Florence-2 intermediate info

## In UI

- optionally show:
  - `Grounding Backend: Florence-2`

That is enough for the first version.

---

# 16. Recommended Branch Implementation Strategy

## Phase A: safe integration

1. create Florence-2 branch
2. add Florence-2 backend loader
3. keep GroundingDINO branch untouched
4. route text-guided candidate proposal through Florence-2
5. keep CLIP, SAM, reliability, UI mostly the same

## Phase B: comparison mode

6. allow backend toggle:
   - `gdino`
   - `florence2`

This is optional but highly useful.

## Phase C: evaluation

7. compare on:
- condition-heavy prompts
- crowded scenes
- relation-heavy scenes
- object + color + location prompts

---

# 17. Expected Benefits

If Florence-2 works well in this role, the project may improve in:

1. richer prompt following
2. less need for aggressive prompt simplification
3. better natural-language grounding behavior
4. better support for condition-heavy prompts
5. stronger research/demo story

---

# 18. Risks

This branch also has real risks:

1. runtime or memory cost may increase
2. Florence-2 integration may be less straightforward than expected
3. output format may not match current pipeline assumptions
4. gains may not be as large as hoped unless the semantic control layer remains strong

So this must be treated as an experimental upgrade, not an automatic guaranteed improvement.

---

# 19. Final Recommendation

Creating a Florence-2 experimental branch is a **good and justified next step** if the goal is to improve natural-language text-guided grounding beyond what GroundingDINO supports well.

However, it should be done in a controlled way:

- text-guided only
- OOD unchanged
- UI mostly stable
- backend isolated
- easy to compare against the current GroundingDINO-based branch

This is the cleanest way to evaluate whether Florence-2 gives real project value.

---

# 20. Best Summary

The Florence-2 branch should be treated as:

- a text-guided grounding experiment
- not an OOD redesign
- not a full system rewrite

The best implementation path is:

1. keep the semantic control layer
2. replace or augment only the text-guided candidate proposal backend
3. preserve OOD and the existing web interface structure
4. evaluate whether Florence-2 actually improves richer natural-language grounding

That is the most responsible and highest-value path forward.

---

# 21. What Florence-2 Should Replace Immediately

This section clarifies the exact role Florence-2 should play in the first implementation phase.

## Florence-2 should replace immediately

In the Florence-2 experimental branch, the model should replace only the **text-guided candidate proposal / grounding backend**.

That means Florence-2 should become the new module responsible for:

1. reading the text-guided prompt
2. grounding that prompt against the image
3. producing one or more candidate regions / boxes for the text-guided path

So the immediate replacement target is:

- `GroundingDINO` in the **text-guided pipeline only**

## Florence-2 should NOT replace immediately

In Phase 1, Florence-2 should **not** replace:

1. `CLIP` semantic verification
2. semantic candidate judging
3. spatial / relation support
4. reliability decision logic
5. `SAM` segmentation
6. the OOD pipeline
7. the overall web application structure

This is important because Florence-2 may be capable of several tasks conceptually, but that does **not** mean it should be trusted as the sole module for all those responsibilities without evaluation.

---

# 22. What Should Remain In Phase 1

The safest and most valuable first Florence-2 branch should keep these layers around the new grounding backend:

## Keep in Phase 1

1. **Semantic Query Controller**
- still needed to preserve:
  - mandatory constraints
  - preferred constraints
  - supportive constraints

2. **CLIP verification**
- still useful as a secondary semantic check
- helps validate whether a Florence-2 candidate really matches the intended text

3. **Semantic candidate judge**
- still needed for:
  - contradiction penalties
  - condition enforcement
  - candidate comparison in crowded scenes

4. **Spatial / relation support**
- still useful as supportive evidence
- especially if Florence-2 grounding is good but not perfect for positional relations

5. **Reliability decision**
- still required to determine:
  - exact match
  - closest match
  - ambiguous match
  - no reliable match

6. **SAM**
- should remain as the segmentation module
- even if Florence-2 can support grounding, SAM is still the safer dedicated segmentation component

## Why this is the best Phase 1 design

Because it lets us test:

- whether Florence-2 improves the weakest stage

without simultaneously removing:

- the safety checks
- the reliability layer
- the segmentation quality

This gives the cleanest comparison against the current GroundingDINO-based branch.

---

# 23. What May Be Removed Later If Florence-2 Performs Well

Only after testing and evaluation should we decide whether some current layers can be simplified or removed.

## Possible later simplifications

If Florence-2 proves strong enough, we may later reduce or simplify:

1. **CLIP verification**
- only if Florence-2 candidates are already semantically reliable enough

2. **extra semantic re-ranking**
- only if Florence-2 already follows richer prompts robustly

3. **some spatial heuristics**
- only if Florence-2 already grounds spatial language more naturally

## What should still likely remain later

Even in a stronger Florence-2 version, these are still likely valuable:

1. **Reliability decision**
- still needed for trustworthy outputs

2. **SAM segmentation**
- still useful as a dedicated final segmentation stage

3. **explanation / reasoning layer**
- still important for UI/demo/review quality

So even in the best case, Florence-2 should be thought of as a strong grounding backbone, not automatically the only system component.

---

# 24. Final Clarification

Florence-2 is a more capable multimodal model than GroundingDINO for prompt-driven vision tasks.

However, for this project, the correct Phase 1 architecture is:

1. replace the text-guided candidate proposal backend with Florence-2
2. keep the rest of the verification / reliability / segmentation stack around it
3. compare results carefully
4. simplify only later if Florence-2 proves strong enough

So the answer to the practical design question is:

- Florence-2 should replace the grounding backend first
- the rest of the pipeline should stay in place initially
- we should not assume “one model can do everything” without evaluation
