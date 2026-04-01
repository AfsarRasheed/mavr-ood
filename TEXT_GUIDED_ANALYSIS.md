# Text-Guided Pipeline Analysis

## Purpose

This note summarizes how the current text-guided pipeline works, what is strong in the design, what is weak or risky, and which improvements would add the most real value to the project.

It is based on the current code in:

- [src/text_guided/pipeline.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/pipeline.py)
- [src/text_guided/query_parser.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/query_parser.py)
- [src/text_guided/scene_agent.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/scene_agent.py)
- [src/text_guided/attribute_agent.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/attribute_agent.py)
- [src/text_guided/reasoning_agent.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/reasoning_agent.py)
- [src/text_guided/visualizer.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/visualizer.py)
- [src/clip_verifier.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/clip_verifier.py)

## Current Pipeline

The current text-guided path works like this:

1. Scene understanding with LLaVA
2. Attribute matching with LLaVA
3. Advanced query parsing with LLaVA, with structured-rules fallback
4. GroundingDINO candidate detection
5. CLIP verification of detected crops
6. Spatial filtering
7. SAM segmentation
8. Final reasoning explanation with LLaVA

Main orchestrator:

- [src/text_guided/pipeline.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/pipeline.py)

## What Is Strong

### 1. Good model role separation

The architecture uses each model for what it is best at:

- LLaVA: scene context, query understanding, explanation
- GroundingDINO: open-vocabulary localization
- CLIP: semantic verification
- SAM: precise segmentation

This is a strong design decision. It is better than forcing LLaVA to directly produce precise bounding boxes.

### 2. Useful fallback behavior already exists

The pipeline is not fragile in only one way. It already has several recovery steps:

- advanced parser falls back to structured rules
- GroundingDINO retries with lower threshold if no candidates are found
- GroundingDINO retries with the raw prompt if refined prompt fails
- CLIP keeps the best candidate if all candidates are rejected

This makes the system more usable than a single brittle path.

### 3. Good debugging visibility

The pipeline already produces:

- intermediate JSON results
- step-by-step images
- reasoning text
- summary output

That is very useful for:

- project documentation
- debugging
- ablation studies
- explaining failures

## What Is Not Good Enough Yet

### 1. The pipeline still tends to return something even when confidence is weak

This is the biggest current weakness.

Examples:

- if CLIP rejects everything, the best candidate is still kept
- if query parsing is weak, the fallback still continues
- if user text is wrong or too specific, the system can still output a plausible-looking but wrong object

This means the system can be confidently wrong instead of safely uncertain.

### 2. Agent outputs are not checked for contradictions

The pipeline currently does not strongly compare:

- scene agent output
- attribute agent output
- parsed query
- final selected detection

So the system can continue even when its own internal reasoning disagrees.

Example:

- scene agent says two zebras are visible
- attribute agent says only one zebra is visible

This should affect confidence, but currently it does not.

### 3. Advanced parsing is better now, but still not fully reliable

The new advanced parser is a real improvement, but it still depends on:

- LLaVA returning valid JSON
- spatial terms being normalized correctly
- heuristic interpretations like:
  - `ahead` = larger `y`
  - `behind` = smaller `y`
  - `between` = closest to midpoint between two anchors

These are reasonable approximations, but not guaranteed to be correct in all scenes.

### 4. Attribute handling is still shallow

The system understands attributes better than before, but it does not strongly enforce them after detection.

Example:

- query says `red car`
- detector finds a car
- CLIP checks overall similarity with the full text prompt

But there is no explicit attribute confirmation step that says:

- is this actually red?
- does this crop better match `red car` than `white car`?

### 5. CLIP verification is useful, but simple

Current CLIP use is mostly:

- one crop
- one text prompt
- one similarity threshold

This is helpful, but not strong enough for fine distinctions like:

- white car vs silver car
- bus vs truck
- damaged car vs normal car

### 6. Reasoning is post-hoc, not decision-controlling

The reasoning agent currently explains what happened after the decision.

That is good for explainability, but it does not improve reliability directly because it is not used to:

- lower confidence
- detect contradiction
- reject weak results

## Best Improvements To Add Real Value

These are the most valuable improvements, in priority order.

### 1. Add a reliability gate

This is the most important improvement.

Before returning the final result, classify it as:

- `high_confidence_match`
- `ambiguous_match`
- `low_confidence_match`
- `no_reliable_match`

Signals you can use from the current pipeline:

- best GroundingDINO score
- best CLIP score
- number of verified candidates
- whether retries were needed
- parser mode
- ambiguity from attribute agent
- whether anchor detection succeeded
- whether scene and attribute reasoning agree

This would directly improve trustworthiness.

### 2. Add contradiction checking

Compare these four things:

- scene agent output
- attribute agent output
- parsed query
- selected detection

If they disagree strongly, reduce confidence or return:

- closest match found
- ambiguous result
- no reliable match

This would make the multi-agent design much stronger.

### 3. Add attribute-specific verification

This would be a high-value upgrade.

Instead of only checking one prompt with CLIP, compare multiple prompts against the same crop.

Examples:

- `red car` vs `white car` vs `blue car`
- `car` vs `truck` vs `bus`
- `damaged car` vs `normal car`

This makes the system much better at confirming what the user actually asked for.

### 4. Replace hard selection with candidate ranking

Right now, the pipeline mostly:

- detects
- filters
- spatially selects

A stronger approach is to assign each candidate a combined score based on:

- GroundingDINO confidence
- CLIP similarity
- spatial match
- attribute match
- anchor consistency

Then rank all candidates and choose the best one with a confidence label.

### 5. Use scene agent output more directly

Right now, scene analysis is informative but underused.

It should help constrain final decision-making.

Examples:

- if scene agent says there is only one white car, confidence should increase when that is selected
- if scene agent does not mention the requested object type, confidence should decrease
- if scene agent says multiple similar objects exist, ambiguity should increase

### 6. Add proper result states in the UI/API

Instead of always behaving like a clean final answer exists, return one of:

- exact match found
- closest match found
- multiple possible matches
- no reliable match found

This is important not just for UI, but for the trustworthiness claim of the project.

## What Not To Prioritize First

These are not the best next steps right now:

- adding more agents
- replacing LLaVA immediately
- heavy prompt tuning before reliability improvements
- fine-tuning before confidence handling is improved

Those may help later, but they are not the highest-value improvements at this stage.

## Honest Overall Verdict

The current text-guided pipeline already has a strong architecture.

Its biggest weakness is not the number of models or the parser quality alone.
Its biggest weakness is that it still lacks a strong notion of when it should distrust itself.

If that is fixed well, the project becomes much stronger:

- academically
- practically
- for documentation
- for comparison against baselines

## Recommended Next Step

The best next development direction is:

- add a reliability and consistency layer on top of the existing 7-step pipeline

That will likely give more value than adding more complexity.

## Suggested Implementation Priority

1. Reliability gate
2. Contradiction checking
3. Attribute-specific CLIP verification
4. Candidate ranking
5. Stronger scene-agent usage
6. UI/API result states
