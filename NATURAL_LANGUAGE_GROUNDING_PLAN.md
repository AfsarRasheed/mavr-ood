# Natural Language Grounding Plan

## Goal

The goal is to move the text-guided pipeline from:

- mostly keyword-driven object selection

toward:

- fuller natural-language understanding
- better matching of what the user actually means
- more reliable selection when the query is complex

This does **not** mean replacing the current architecture.

It means improving how the project uses:

- LLaVA
- GroundingDINO
- CLIP
- spatial reasoning

so the final result better reflects the full sentence, not just a few extracted words.

## Current Limitation

Right now the pipeline works like this:

1. parse the query
2. reduce it to a shorter detection prompt
3. detect candidates
4. apply CLIP filtering
5. apply spatial filtering
6. segment the selected object

This is good, but it still behaves mostly like:

- natural language in
- reduced prompt + rules underneath

That means the system can miss the deeper meaning of the sentence.

Example:

`find the damaged white car near the pole on the right`

Current behavior is likely to reduce this into something like:

- `white car`

and then use:

- `right`
- maybe `near pole`

as separate simple constraints.

This is useful, but not full natural-language grounding.

## What We Actually Want

We want the system to behave more like:

- understand the whole sentence
- identify all important constraints
- detect multiple candidate objects
- compare candidates against the full sentence meaning
- choose the best one only if confidence is strong

This is much closer to how ChatGPT-style vision interaction feels.

## Key Design Idea

Do **not** rely only on better parsing.

Better parsing helps, but the real improvement comes from:

- evaluating each candidate against the **full query meaning**

That is the important shift.

## Proposed Architecture Upgrade

### Stage A: Full Query Understanding

Instead of reducing the user prompt only to:

- object
- color
- spatial

build a richer internal representation.

Example structure:

```json
{
  "object": "car",
  "attributes": {
    "color": "white",
    "condition": "damaged"
  },
  "spatial": {
    "relation": "near",
    "anchor": "pole",
    "region_bias": "right"
  },
  "count": "single",
  "priority_order": ["object", "color", "spatial", "condition"]
}
```

This is stronger than just producing:

- `white car`

### Stage B: Multi-Candidate Detection

Continue using GroundingDINO to generate candidates.

This part is already good.

Do not reduce the pipeline to one early hard selection.

### Stage C: Full-Query Candidate Scoring

For each candidate, compute how well it matches the **entire query**, not just the detection prompt.

Each candidate should get scores like:

- object match score
- attribute match score
- spatial match score
- anchor relation score
- scene consistency score
- CLIP semantic score

Then combine these into a total candidate score.

This is the core improvement.

### Stage D: VLM-Based Candidate Reranking

This is the most ChatGPT-like upgrade.

After you have a few strong candidates:

- crop each candidate
- send the candidate crop plus query context to the VLM
- ask which candidate best satisfies the query and why

This should be used as a reranking signal, not the only signal.

That means the final decision is based on:

- detector evidence
- CLIP evidence
- spatial evidence
- VLM semantic judgment

### Stage E: Confidence-Aware Final Decision

Do not always return a final object as if it is definitely correct.

Return one of:

- exact match found
- closest match found
- ambiguous match
- no reliable match

This is critical for trustworthiness.

## What Should Change in This Repo

## 1. Query Representation

### Current issue

Current parsing produces a basic structure but not enough detailed meaning.

### Needed improvement

Expand query representation in:

- [src/text_guided/query_parser.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/query_parser.py)

Add support for:

- multiple attributes
- condition/state descriptors
- explicit constraint priority
- richer anchor relationships
- count and uniqueness expectations

## 2. Candidate Scoring Layer

### Current issue

The current system filters and selects candidates in separate rigid steps.

### Needed improvement

Add a scoring function in:

- [src/text_guided/pipeline.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/pipeline.py)

Possible candidate score structure:

```text
final_score =
  object_score * w1 +
  attribute_score * w2 +
  clip_score * w3 +
  spatial_score * w4 +
  anchor_score * w5 +
  scene_consistency_score * w6
```

This lets the system compare candidates more intelligently.

## 3. Attribute Verification

### Current issue

Attributes are not strongly verified after detection.

### Needed improvement

Extend:

- [src/clip_verifier.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/clip_verifier.py)

Add discriminative comparisons such as:

- `red car` vs `white car`
- `car` vs `truck`
- `damaged car` vs `normal car`

This will improve matching of the actual user intent.

## 4. VLM Candidate Reranking

### Current issue

LLaVA is currently used mainly before and after detection, not as a strong judge of candidate correctness.

### Needed improvement

In:

- [src/text_guided/pipeline.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/pipeline.py)
- [src/agents/vlm_backend.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/agents/vlm_backend.py)

Add a lightweight reranking stage that asks:

- which candidate best matches the user query?
- which constraints are satisfied or violated?

This should not replace the detector. It should improve the final decision.

## 5. Output State and Confidence

### Current issue

The pipeline still tends to return something even when the match is weak.

### Needed improvement

In:

- [src/text_guided/pipeline.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/pipeline.py)
- [web_app.py](c:/Users/OMEN/Desktop/MAVR-OOD/web_app.py)
- [static/js/app.js](c:/Users/OMEN/Desktop/MAVR-OOD/static/js/app.js)

Add final result states:

- exact match
- closest match
- ambiguous match
- no reliable match

This is essential if the project is meant to feel intelligent and trustworthy.

## What Would Bring the Most Value First

If only one upgrade is done first, it should be:

- full-query candidate scoring

Why:

- parsing alone is not enough
- better reasoning alone is not enough
- scoring candidates against the whole query is the main missing capability

## Best Upgrade Order

### Phase 1

1. richer query representation
2. full-query candidate scoring

### Phase 2

3. attribute-specific CLIP verification
4. confidence-based output states

### Phase 3

5. VLM candidate reranking
6. stronger contradiction checking between agents

## Example of the Desired Behavior

User query:

`find the white car near the sign on the right`

Desired internal behavior:

1. parse:
   - object = car
   - color = white
   - anchor = sign
   - relation = near
   - region = right

2. detect multiple cars and possible anchors

3. evaluate each car:
   - is it a car?
   - is it white?
   - is it on the right?
   - is it close to the sign?
   - does CLIP support the full meaning?

4. rank candidates

5. return:
   - exact match if one candidate clearly wins
   - ambiguous if two candidates are close
   - closest match if partial constraints match
   - not found if no reliable match exists

This is much closer to how a stronger natural-language system should behave.

## What Not To Do

Avoid these as the first step:

- adding more agents immediately
- fully replacing GroundingDINO
- trying to localize directly from LLaVA alone
- heavy fine-tuning before the ranking logic is improved

Those may help later, but they are not the most important current gap.

## Final View

The current project already has a strong modular structure.

To make it feel more like ChatGPT-style image querying, the key improvement is:

- not just better parsing
- but better **candidate understanding and ranking against full natural-language meaning**

That is the direction that will add the most real value.
