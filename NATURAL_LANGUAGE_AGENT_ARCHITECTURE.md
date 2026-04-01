# Natural Language Agent Architecture

## Purpose

This document explains a proposed agent architecture for the natural-language grounding version of the project.

It answers:

- which agents should exist
- whether the Attribute Matching Agent should remain
- how the project would work after the architecture upgrade
- how the full pipeline would behave on a real example

This is a design document for the new branch:

- `improvement/natural-language-grounding`

It is intended to guide implementation carefully without damaging the current stable branch.

## Short Answer First

Yes, the project should still keep an **Attribute Matching Agent**.

It should **not** be removed immediately.

However, its role should become more meaningful.

Right now it mainly helps with:

- prompt rewriting
- ambiguity estimation
- rough object matching

In the natural-language grounding version, it should help with:

- attribute verification
- ambiguity reasoning
- candidate comparison
- support for full-query matching

So the architecture becomes richer, not weaker.

## Main Idea

The current text-guided pipeline behaves mostly like:

- parse query
- reduce query to a shorter detector prompt
- detect objects
- apply CLIP and spatial filtering
- segment final object

The new natural-language grounding version should behave more like:

- understand the whole sentence
- extract all important constraints
- detect multiple candidate objects
- evaluate each candidate against the full sentence meaning
- rank candidates
- choose only if confidence is strong
- explain why

That means the project is no longer only a filtered detection pipeline.

It becomes a **candidate reasoning and ranking system**.

## Proposed Agent and Module Layout

The best balance is:

- keep important reasoning agents
- add stronger candidate evaluation logic
- keep detection and segmentation modules as specialized tools

## Proposed Components

### 1. Scene Understanding Agent

#### Type

LLaVA-based agent

#### Role

This agent looks at the image globally and provides scene-level context.

It should describe:

- scene type
- road context
- lighting
- rough object inventory
- whether the scene is urban, highway, rural, night, etc.

#### Why it matters

This information helps later stages reason about:

- what objects are likely to exist
- how ambiguous the scene is
- whether the user query makes sense in context

#### Example output

```json
{
  "scene_type": "city street at night",
  "lighting": "dark/night",
  "objects": [
    {"name": "white sedan", "position": "right"},
    {"name": "black SUV", "position": "center"},
    {"name": "traffic sign", "position": "far right"}
  ]
}
```

## 2. Query Understanding Agent

#### Type

LLaVA-based agent or hybrid parser agent

#### Role

This is one of the most important additions.

Instead of reducing the query to just a short detection prompt, this agent should understand:

- object type
- attributes
- relations
- anchor objects
- uniqueness
- priority of constraints
- intent

#### Why it matters

This is what moves the project closer to ChatGPT-like image querying.

It should understand:

- what the user literally asked
- which parts of the sentence matter most
- how strict the match needs to be

#### Example query

`find the damaged white car near the traffic sign on the right`

#### Desired internal representation

```json
{
  "target_object": "car",
  "attributes": {
    "color": "white",
    "condition": "damaged"
  },
  "spatial_constraints": {
    "region": "right",
    "relation": "near",
    "anchor": "traffic sign"
  },
  "match_type": "single_best_match",
  "priority_order": ["object", "color", "relation", "region", "condition"]
}
```

## 3. Attribute Matching Agent

#### Type

LLaVA-based agent

#### Role

This agent stays in the system.

Its role becomes stronger than prompt suggestion.

It should:

- compare scene objects against the requested attributes
- identify which scene objects are plausible matches
- estimate ambiguity
- explain what attributes are satisfied and what are missing

#### Why it matters

If the query says:

- white car
- damaged truck
- pedestrian with red clothing

this agent helps narrow the meaning before final ranking.

#### Example contribution

Given:

- two cars in the scene
- one white, one black

and query:

- `the white car on the right`

this agent should say:

- the white sedan on the right is the strongest attribute match
- ambiguity is low

#### New role summary

The Attribute Matching Agent should become:

- an **attribute and ambiguity reasoning agent**

not only a prompt recommender.

## 4. Candidate Grounding Module

#### Type

GroundingDINO module

#### Role

This module proposes multiple possible candidate objects.

It should not try to decide the final answer alone.

It is responsible for:

- open-vocabulary object localization
- candidate generation

#### Why it matters

GroundingDINO is still the best specialized grounding module in this pipeline.

It should remain part of the architecture.

## 5. Semantic Verification Module

#### Type

CLIP-based module

#### Role

CLIP should validate how well a crop matches the intended semantics.

In the improved version, it should do more than a single threshold check.

It should support:

- object match verification
- attribute comparison
- type comparison

#### Example

For a candidate crop, compare:

- `white car`
- `black car`
- `truck`

This helps distinguish similar objects better than one prompt alone.

## 6. Spatial / Relation Reasoning Module

#### Type

Hybrid reasoning module

#### Role

This module handles:

- left/right
- top/bottom
- between
- near
- behind
- in front
- ordinal phrases
- anchor-based relations

#### Why it matters

This is what allows the system to use full sentence relations instead of only object category.

#### Important note

This module is still partly heuristic because it relies on 2D image geometry.

That is acceptable, but it must be treated as:

- useful
- not perfect

## 7. Candidate Ranking Module

#### Type

New scoring/ranking module

#### Role

This is one of the most important additions in the new architecture.

It should combine evidence from all earlier stages.

For every candidate, it should compute a total score from:

- detector confidence
- CLIP similarity
- attribute match
- spatial match
- anchor consistency
- scene consistency
- ambiguity penalties

#### Why it matters

This is the component that turns the project from:

- rigid pipeline filtering

into:

- candidate reasoning and ranking

## 8. Optional VLM Candidate Reranking Agent

#### Type

LLaVA-based judge agent

#### Role

This agent takes the top few candidates and helps answer:

- which candidate best satisfies the whole natural-language query?
- why?

It should **not** replace detector evidence.

It should be used as an additional decision signal.

#### Why it matters

This is what makes the system feel more like:

- upload image
- ask in natural language
- system reasons about which object best matches

That is closer to the interaction style the user wants.

## 9. Confidence / Reliability Module

#### Type

Decision control module

#### Role

This module decides whether the system should:

- return a final exact match
- return a closest match
- say the result is ambiguous
- say no reliable match was found

#### Why it matters

This is critical for trustworthiness.

Without it, the project may still return a plausible but wrong object too confidently.

## 10. Segmentation Module

#### Type

SAM module

#### Role

After final candidate selection, SAM creates the segmentation mask.

This stage should remain late in the pipeline.

The architecture should continue to use SAM only after candidate selection is strong enough.

## 11. Final Reasoning Agent

#### Type

LLaVA-based explanation agent

#### Role

This agent explains:

- why the system selected the final object
- which constraints matched
- what was rejected
- how confident the system is

#### Why it matters

This is useful both for:

- user trust
- research presentation

## Final Agent Structure

### LLaVA-Based Agents

1. Scene Understanding Agent
2. Query Understanding Agent
3. Attribute Matching Agent
4. Optional Candidate Reranking Agent
5. Final Reasoning Agent

### Specialized Modules

1. GroundingDINO Candidate Proposal
2. CLIP Semantic Verification
3. Spatial / Relation Reasoning
4. Candidate Ranking
5. Confidence / Reliability Decision
6. SAM Segmentation

## Architecture Flow

### Current stable flow

```text
User Query
   ↓
Query Parsing
   ↓
GroundingDINO
   ↓
CLIP
   ↓
Spatial Filter
   ↓
SAM
   ↓
Reasoning
```

### Proposed natural-language flow

```text
User Query + Image
      ↓
Scene Understanding Agent
      ↓
Query Understanding Agent
      ↓
Attribute Matching Agent
      ↓
GroundingDINO Candidate Proposal
      ↓
CLIP Semantic Verification
      ↓
Spatial / Relation Reasoning
      ↓
Candidate Ranking Module
      ↓
Optional VLM Candidate Reranking Agent
      ↓
Confidence / Reliability Decision
      ↓
SAM Segmentation
      ↓
Final Reasoning Agent
```

## Full Example

This section explains how the proposed architecture should work end to end.

## Example Query

User uploads an image and asks:

`find the white car near the traffic sign on the right`

## Step 1: Scene Understanding Agent

The scene agent analyzes the full image.

Possible output:

```json
{
  "scene_type": "multi-lane road in daylight",
  "lighting": "bright",
  "objects": [
    {"name": "black SUV", "position": "center"},
    {"name": "white sedan", "position": "right"},
    {"name": "traffic sign", "position": "far right"},
    {"name": "silver car", "position": "left"}
  ]
}
```

What this contributes:

- there is at least one white car
- there is a traffic sign
- the scene is not highly ambiguous

## Step 2: Query Understanding Agent

The query is interpreted fully.

Expected structured meaning:

```json
{
  "target_object": "car",
  "attributes": {
    "color": "white"
  },
  "spatial_constraints": {
    "relation": "near",
    "anchor": "traffic sign",
    "region": "right"
  },
  "priority_order": ["object", "color", "relation", "region"]
}
```

This is much stronger than reducing the query to:

- `white car`

## Step 3: Attribute Matching Agent

The attribute agent compares the scene objects against the query meaning.

Possible conclusion:

- white sedan on the right is the strongest match
- black SUV does not match color
- silver car does not match region
- ambiguity is low

## Step 4: GroundingDINO Candidate Proposal

GroundingDINO detects several possible vehicle boxes.

Candidates might include:

- candidate 1: black SUV
- candidate 2: white sedan
- candidate 3: silver car

It may also detect the traffic sign as an anchor object.

## Step 5: CLIP Semantic Verification

For each candidate crop, CLIP estimates semantic match.

This can include:

- candidate 1 vs `white car`
- candidate 2 vs `white car`
- candidate 3 vs `white car`

Expected result:

- candidate 2 scores highest

## Step 6: Spatial / Relation Reasoning

Now evaluate:

- is the candidate on the right?
- is it near the traffic sign?

Expected result:

- candidate 2 matches both better than the others

## Step 7: Candidate Ranking

All candidate evidence is combined.

Example:

- candidate 1: good object match, poor color match, poor region match
- candidate 2: strong object, color, and relation match
- candidate 3: moderate object match, weaker region and relation match

Candidate 2 becomes top-ranked.

## Step 8: Optional VLM Candidate Reranking

The VLM can compare top candidates and confirm:

- candidate 2 best fits the sentence
- candidate 1 and 3 are partial or weaker matches

## Step 9: Confidence Decision

The reliability module decides:

- exact match found

If the evidence were weak, it could instead say:

- closest match found
- ambiguous match
- no reliable match

## Step 10: SAM Segmentation

SAM segments the selected final candidate.

This gives the final output mask.

## Step 11: Final Reasoning Agent

The final reasoning agent explains:

- the system found a white sedan on the right
- it was near the traffic sign
- it matched both object type and color
- other vehicles were rejected because they were not white or not properly positioned

## Example Output Behavior

### Best case

- exact match found
- white car correctly segmented
- reasoning explains why

### Medium case

- closest match found
- maybe a silver-white vehicle near the sign
- explanation says the color match is partial

### Weak case

- ambiguous match
- two white cars near the right side
- reasoning says the system is unsure

### Failure case

- no reliable match found
- no white car near a traffic sign on the right

This is the kind of behavior that makes the project more trustworthy.

## Why This Architecture Is Better

This new architecture improves the project because it:

- preserves the current strong modules
- adds better natural-language understanding
- reasons across multiple candidates
- allows uncertainty instead of forced answers
- is closer to ChatGPT-style object finding behavior

## Final View

The project should **keep** the Attribute Matching Agent.

The natural-language version should not remove important existing pieces.

Instead, it should:

- strengthen their roles
- add candidate ranking
- add confidence handling
- improve full-query reasoning

That is the safest and strongest architectural direction.
