"""
Candidate-Level Match Reasoning

Adds lightweight, candidate-aware reasoning on top of structured scores so the
pipeline can explain which constraints each candidate satisfies or violates.
"""

from typing import Dict, List


def _confidence_to_score(label):
    mapping = {"high": 0.9, "medium": 0.7, "low": 0.5}
    return mapping.get(str(label or "").lower(), 0.6)


def summarize_candidate_match(parsed: dict, candidate_scores: dict, attr_result: dict | None = None) -> Dict[str, object]:
    """
    Build a candidate-level reasoning summary from the structured query and score
    components already computed by the pipeline.
    """
    attrs = parsed.get("attributes") or {}
    color = attrs.get("color")
    condition = attrs.get("condition")
    spatial = parsed.get("spatial")
    target_object = parsed.get("target_object")

    satisfied: List[str] = []
    violated: List[str] = []
    partial: List[str] = []

    object_score = float(candidate_scores.get("object_score", 0.0))
    attribute_score = float(candidate_scores.get("attribute_score", 0.0))
    color_score = float(candidate_scores.get("color_score", attribute_score))
    condition_score = float(candidate_scores.get("condition_score", attribute_score))
    spatial_score = float(candidate_scores.get("spatial_score", 0.0))
    clip_score = float(candidate_scores.get("clip_score", 0.0))
    anchor_confidence_score = float(candidate_scores.get("anchor_confidence_score", 1.0))
    is_relational = spatial in {"next_to", "behind", "in_front", "above", "below", "between"}

    if object_score >= 0.7:
        satisfied.append(f"object match ({target_object})")
    elif object_score >= 0.5:
        partial.append(f"object match is only moderate ({target_object})")
    else:
        violated.append(f"object evidence is weak for {target_object}")

    if color:
        color_contrast = float(candidate_scores.get("color_contrast", 0.0))
        if color_score >= 0.7 and color_contrast >= -0.02:
            satisfied.append(f"color matches ({color})")
        elif color_score >= 0.5:
            partial.append(f"color evidence is partial ({color})")
        else:
            violated.append(f"color match is weak ({color})")

    if condition:
        condition_contrast = float(candidate_scores.get("condition_contrast", 0.0))
        if condition_score >= 0.68 and condition_contrast >= -0.02:
            satisfied.append(f"condition matches ({condition})")
        elif condition_score >= 0.5:
            partial.append(f"condition evidence is partial ({condition})")
        else:
            violated.append(f"condition match is weak ({condition})")

    if spatial:
        if spatial_score >= 0.7:
            satisfied.append(f"spatial constraint fits ({spatial})")
        elif spatial_score >= 0.5:
            partial.append(f"spatial evidence is partial ({spatial})")
        else:
            violated.append(f"spatial constraint is weak ({spatial})")

    if is_relational:
        if anchor_confidence_score >= 0.7:
            satisfied.append("anchor evidence is strong")
        elif anchor_confidence_score >= 0.45:
            partial.append("anchor evidence is only moderate")
        else:
            violated.append("anchor evidence is weak")

    if clip_score >= 0.7:
        satisfied.append("full-query CLIP evidence is strong")
    elif clip_score >= 0.5:
        partial.append("full-query CLIP evidence is moderate")
    else:
        violated.append("full-query CLIP evidence is weak")

    ambiguity_penalty = 0.0
    relation_uncertainty_penalty = 0.0
    if isinstance(attr_result, dict):
        ambiguity = str(attr_result.get("ambiguity", "")).lower()
        matched = attr_result.get("matched_objects", []) or []
        if ambiguity == "high":
            ambiguity_penalty += 0.08

        if matched:
            top_conf = max(_confidence_to_score(item.get("confidence")) for item in matched[:3])
            ambiguity_penalty += max(0.0, 0.75 - top_conf) * 0.1

    if is_relational:
        relation_uncertainty_penalty += max(0.0, 0.65 - anchor_confidence_score) * 0.18

    return {
        "satisfied_constraints": satisfied,
        "partial_constraints": partial,
        "violated_constraints": violated,
        "ambiguity_penalty": round(float(ambiguity_penalty), 4),
        "relation_uncertainty_penalty": round(float(relation_uncertainty_penalty), 4),
        "reason": build_candidate_reason_text(satisfied, partial, violated),
    }


def build_candidate_reason_text(satisfied: List[str], partial: List[str], violated: List[str]) -> str:
    parts = []
    if satisfied:
        parts.append("satisfies " + ", ".join(satisfied[:3]))
    if partial:
        parts.append("has partial support for " + ", ".join(partial[:2]))
    if violated:
        parts.append("misses " + ", ".join(violated[:2]))
    return "; ".join(parts) if parts else "Candidate evidence is limited."
