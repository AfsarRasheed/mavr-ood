# Current Text-Guided Prompt Capabilities

This document describes the kinds of prompts the current `improvement/web-ui` version of the text-guided pipeline can handle reasonably well.

It is meant to reflect the **current branch behavior**, not the future natural-language grounding branch.

## Important Note

The current system is **not full free-form natural-language grounding** yet.

It works best when the query includes one or more of these clearly expressed cues:

- target object name
- color or simple attribute
- spatial term
- size clue
- simple relation to another object
- ordinal cue such as `second from right`

So this version is best described as:

- **advanced structured natural-language parsing**
- not fully open-ended conversational grounding

## Prompt Categories That Work Best

### 1. Basic Object Prompts

These are the most reliable prompts.

Examples:

- `the zebra`
- `the horse`
- `the cow`
- `the donkey`
- `the rhino`
- `the wild boar`
- `the car`
- `the pedestrian`
- `the truck`

Use when:

- there is one clear target object
- or when you want a simple baseline query

---

### 2. Object + Color

These usually work well if the color is visually clear.

Examples:

- `the red car`
- `the white car`
- `the black car`
- `the brown horse`
- `the white vehicle`

Use when:

- there are multiple similar objects
- color helps disambiguate

Limit:

- if the color is subtle, shadowed, or ambiguous, performance may drop

---

### 3. Object + Basic Spatial Position

These are strong prompt types for the current version.

Examples:

- `the zebra on the left`
- `the car on the right`
- `the vehicle in the center`
- `the object at the bottom`
- `the sign at the top`

Supported simple spatial ideas:

- `left`
- `right`
- `center`
- `top`
- `bottom`

---

### 4. Object + Relative Size

These are supported when multiple candidates exist.

Examples:

- `the larger zebra`
- `the largest vehicle`
- `the smallest car`
- `the bigger animal`

Supported size-style wording:

- `largest`
- `smallest`
- `larger`
- `bigger`

---

### 5. Object + Distance Style Prompt

These can work when object layout is visually distinct.

Examples:

- `the nearest car`
- `the farthest vehicle`
- `the animal closest to the camera`

Supported distance-style wording:

- `nearest`
- `farthest`
- `closest`

Limit:

- this is still heuristic and based on image-space reasoning, not full depth understanding

---

### 6. Ordinal Prompts

These are part of the advanced parser and are useful when several similar objects are visible.

Examples:

- `the second car from the right`
- `the first vehicle from the left`
- `the third object from the left`

Useful ordinal words:

- `first`
- `second`
- `third`

Supported directional phrases:

- `from the left`
- `from the right`

Limit:

- best used when the objects are clearly separated horizontally

---

### 7. Simple Relation Prompts

These are supported more than before, but still depend on good detection of anchor objects.

Examples:

- `the car next to the truck`
- `the object near the pole`
- `the car behind the bus`
- `the sign above the vehicle`
- `the pedestrian below the sign`

Supported relation-style wording:

- `next to`
- `near`
- `behind`
- `in front`
- `above`
- `below`

Limit:

- these depend on both the target object and anchor object being detected properly

---

### 8. Between Queries

These are part of the advanced parser and can work for suitable images.

Examples:

- `the car between the truck and the bus`
- `the object between the two vehicles`

Limit:

- this needs correct detection of both anchors
- still heuristic, not full semantic scene graph reasoning

---

### 9. Object + Context Description

These can work when the descriptive phrase still points clearly to a real visible object.

Examples:

- `the horse on the road`
- `the rhino crossing the road`
- `the animal near the road edge`
- `the vehicle in the lane`

These work best when:

- the description is concrete
- the target object is visually obvious

---

### 10. Condition-Like Prompts

These may work in some cases, but are not yet fully reliable.

Examples:

- `the burning car`
- `the damaged vehicle`
- `the overturned truck`
- `the broken object on the road`

Important:

- these are the kinds of prompts that expose current weaknesses
- the system may detect the correct object, but it may also over-trust object class or spatial cues instead of the condition itself

So these are **possible prompts**, but not yet the most robust category.

## Strong Prompt Templates for This Version

These are good templates to follow.

### Template A: Simple Object

`the <object>`

Examples:

- `the zebra`
- `the horse`

### Template B: Object + Color

`the <color> <object>`

Examples:

- `the red car`
- `the white vehicle`

### Template C: Object + Position

`the <object> on the <left/right>`

Examples:

- `the zebra on the left`
- `the car on the right`

### Template D: Object + Size

`the <largest/smallest> <object>`

Examples:

- `the largest vehicle`
- `the larger zebra`

### Template E: Ordinal Object

`the <first/second/third> <object> from the <left/right>`

Examples:

- `the second car from the right`

### Template F: Relation

`the <object> <next to/behind/above/below> the <anchor>`

Examples:

- `the car next to the truck`
- `the sign above the vehicle`

### Template G: Between Relation

`the <object> between the <anchor1> and the <anchor2>`

Examples:

- `the car between the truck and the bus`

## Prompt Types That Are Risky in This Version

These are not impossible, but they are less reliable right now.

### 1. Very conversational prompts

Examples:

- `can you find the vehicle that looks most suspicious`
- `show me the object that seems dangerous`

### 2. Abstract intent prompts

Examples:

- `the most important object`
- `the object causing the biggest problem`

### 3. Subjective description prompts

Examples:

- `the car that looks older`
- `the object that seems unusual`

### 4. Complex multi-clause prompts

Examples:

- `find the car on the right that is slightly behind the other one and looks more damaged`

### 5. Hidden reasoning prompts

Examples:

- `the animal that should not be there`
- `the object most likely to create an accident`

These are closer to the **future natural-language grounding branch**, not the current stable branch.

## Practical Recommendation

For the current version, the best prompt design is:

1. name the object clearly
2. add one or two concrete constraints
3. avoid long conversational phrasing
4. avoid subjective or abstract reasoning-heavy wording

## Best Examples to Use Right Now

These are strong prompt examples for demos and testing.

- `the zebra on the left`
- `the larger zebra`
- `the horse on the road`
- `the rhino crossing the road`
- `the red car on the left`
- `the second car from the right`
- `the vehicle near the pole`
- `the car next to the truck`
- `the sign above the vehicle`
- `the car between the truck and the bus`

## Summary

Current `improvement/web-ui` text-guided detection can handle:

- basic object prompts
- color-based prompts
- simple spatial prompts
- size-based prompts
- ordinal prompts
- simple relation prompts
- some between/anchor-based prompts
- some descriptive context prompts

It is still weaker on:

- highly conversational prompts
- abstract intent prompts
- subjective reasoning-heavy prompts
- complex full natural-language grounding

That broader behavior is planned for the experimental natural-language branch, not this current stable branch.
