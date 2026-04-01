# Text-Guided Refinement Roadmap

## Goal

The goal of refinement is not to make the project more complicated.

The goal is to make it:

- more reliable
- less likely to produce confident wrong detections
- better at handling difficult or imperfect user queries
- stronger as a research system

This roadmap focuses on improvements that strengthen the current pipeline without degrading it.

## Core Principle

Do not add complexity unless it clearly improves one of these:

- accuracy
- reliability
- confidence estimation
- failure handling
- interpretability

If a feature makes the system harder to control without adding clear value, it should not be prioritized.

## Current Direction

The current text-guided pipeline is already strong in architecture:

- LLaVA for semantic understanding
- GroundingDINO for object grounding
- CLIP for semantic filtering
- SAM for segmentation

So the next step is not to redesign everything.

The next step is to reduce weak behavior in the current system.

## High-Value Refinement Priorities

### Priority 1: Reliability Gate

#### Why this matters

The biggest current weakness is that the pipeline can still return a result when confidence is weak.

That is dangerous for a project that aims to be trustworthy.

#### What to add

Add a decision layer after CLIP and spatial filtering that classifies the result as:

- `high_confidence_match`
- `ambiguous_match`
- `low_confidence_match`
- `no_reliable_match`

#### Signals to use

- GroundingDINO detection score
- CLIP similarity score
- number of verified candidates
- whether retries were needed
- ambiguity from attribute agent
- whether parser fell back from LLaVA to rules
- whether anchor detection succeeded

#### Best files to update

- [src/text_guided/pipeline.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/pipeline.py)
- [web_app.py](c:/Users/OMEN/Desktop/MAVR-OOD/web_app.py)
- [static/js/app.js](c:/Users/OMEN/Desktop/MAVR-OOD/static/js/app.js)

#### Value added

- avoids confident wrong answers
- improves trustworthiness
- gives better project evidence

### Priority 2: Contradiction Checking

#### Why this matters

Different agents can currently disagree without consequence.

Example:

- scene agent says two objects exist
- attribute agent implies only one exists

That inconsistency should affect confidence.

#### What to add

Compare:

- scene agent output
- attribute agent output
- parsed query
- final selected candidate

If they disagree strongly:

- lower confidence
- mark result as ambiguous
- or return closest match / no reliable match

#### Best files to update

- [src/text_guided/pipeline.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/pipeline.py)
- [src/text_guided/scene_agent.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/scene_agent.py)
- [src/text_guided/attribute_agent.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/attribute_agent.py)

#### Value added

- strengthens multi-agent reasoning
- reduces silent failure
- makes the architecture more meaningful

### Priority 3: Attribute-Specific Verification

#### Why this matters

Right now, the system can verify object-level similarity, but not always requested attributes strongly enough.

Example:

- user asks for `red car`
- system may still pick a non-red car if the overall prompt matches weakly

#### What to add

For the selected crop, compare multiple text prompts using CLIP.

Examples:

- `red car` vs `white car` vs `blue car`
- `car` vs `truck` vs `bus`
- `damaged car` vs `normal car`

This is a discriminative check, not just a threshold check.

#### Best files to update

- [src/clip_verifier.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/clip_verifier.py)
- [src/text_guided/pipeline.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/pipeline.py)

#### Value added

- better handling of color/type confusion
- better match to the actual user query
- stronger final selection quality

### Priority 4: Candidate Ranking Instead of Hard Selection

#### Why this matters

The current pipeline mostly does:

- detect
- filter
- spatially select

This works, but it is rigid.

#### What to add

Assign each candidate a combined score using:

- GroundingDINO score
- CLIP score
- spatial consistency
- attribute consistency
- anchor consistency

Then rank candidates and pick the best one.

#### Best files to update

- [src/text_guided/pipeline.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/pipeline.py)
- [src/text_guided/query_parser.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/query_parser.py)

#### Value added

- more stable selection
- easier to explain why one candidate was chosen
- better handling of crowded scenes

### Priority 5: Stronger Use of Scene Agent Output

#### Why this matters

The scene agent currently adds context, but it is not deeply used for decision control.

#### What to add

Use scene output to influence confidence.

Examples:

- if scene agent sees no bicycle and user asks for bicycle, confidence should drop
- if scene agent sees only one white car, confidence should increase when that is selected
- if scene agent sees many similar vehicles, ambiguity should increase

#### Best files to update

- [src/text_guided/pipeline.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/pipeline.py)
- [src/text_guided/scene_agent.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/scene_agent.py)

#### Value added

- makes the scene agent genuinely useful
- strengthens the multi-agent claim
- improves explainability with actual control value

### Priority 6: Better Result States in UI/API

#### Why this matters

The UI should not always look like the system is fully sure.

#### What to add

Return one of:

- exact match found
- closest match found
- multiple possible matches
- no reliable match found

#### Best files to update

- [web_app.py](c:/Users/OMEN/Desktop/MAVR-OOD/web_app.py)
- [static/js/app.js](c:/Users/OMEN/Desktop/MAVR-OOD/static/js/app.js)
- [static/index.html](c:/Users/OMEN/Desktop/MAVR-OOD/static/index.html)

#### Value added

- better user trust
- more honest output behavior
- stronger demo quality

## Medium-Priority Refinements

These are useful, but should come after the high-value reliability improvements.

### 1. Better parser robustness

- improve JSON extraction safety in advanced parsing
- better support for natural language variants
- stronger handling of malformed LLaVA parse responses

Best file:

- [src/text_guided/query_parser.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/query_parser.py)

### 2. Better reasoning quality control

- make final reasoning shorter and more precise
- align reasoning with actual final decision state
- ensure reasoning reflects confidence level

Best file:

- [src/text_guided/reasoning_agent.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/reasoning_agent.py)

### 3. Better visualization for debugging

- highlight confidence state
- show candidate ranking
- show contradictions if any

Best files:

- [src/text_guided/visualizer.py](c:/Users/OMEN/Desktop/MAVR-OOD/src/text_guided/visualizer.py)
- [static/js/app.js](c:/Users/OMEN/Desktop/MAVR-OOD/static/js/app.js)

## What To Avoid For Now

These are not the best next moves right now.

### Avoid 1: Adding more agents immediately

More agents can increase complexity without increasing reliability.

### Avoid 2: Fine-tuning before reliability is improved

Fine-tuning may help later, but first the system should learn when not to trust weak detections.

### Avoid 3: Large architecture changes too early

The current structure is already good.
Refinement should focus on stronger behavior, not rebuilding everything.

### Avoid 4: More UI features before core confidence handling

The project’s value depends more on decision quality than on extra interface features.

## Recommended Execution Order

This is the safest refinement order.

### Phase 1

1. Reliability gate
2. Contradiction checking

### Phase 2

3. Attribute-specific verification
4. Candidate ranking

### Phase 3

5. Better scene-agent usage
6. Better UI/API result states

### Phase 4

7. Parser robustness
8. Reasoning refinement
9. Visualization improvements

## Best Immediate Next Step

If only one thing is improved next, it should be:

- reliability and uncertainty handling in the text-guided pipeline

That is the highest-value refinement for this project right now.

## Final Note

The project already has a strong design.

The main refinement opportunity is not adding more complexity.
It is making the current system:

- more self-aware
- more selective
- more honest when uncertain

That is what will strengthen the project most.
