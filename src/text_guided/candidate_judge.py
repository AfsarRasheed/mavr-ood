"""
Semantic Candidate Judge

Applies query-aware semantic checks to a scored candidate so the pipeline can
enforce mandatory constraints more strongly than supportive ones.
"""

from typing import Dict, List


def _append(bucket: List[str], text: str):
    if text and text not in bucket:
        bucket.append(text)


def judge_candidate_against_plan(semantic_plan: dict, parsed: dict, candidate_scores: dict,
                                 clip_scores: dict | None = None) -> Dict[str, object]:
    clip_scores = clip_scores or {}
    mandatory = semantic_plan.get("mandatory_constraints", [])
    preferred = semantic_plan.get("preferred_constraints", [])
    supportive = semantic_plan.get("supportive_constraints", [])
    query_type = semantic_plan.get("query_type", "object-centric")

    object_score = float(candidate_scores.get("object_score", 0.0))
    color_score = float(candidate_scores.get("color_score", candidate_scores.get("attribute_score", 0.0)))
    condition_score = float(candidate_scores.get("condition_score", candidate_scores.get("attribute_score", 0.0)))
    spatial_score = float(candidate_scores.get("spatial_score", 0.0))
    clip_score = float(candidate_scores.get("clip_score", 0.0))
    anchor_score = float(candidate_scores.get("anchor_confidence_score", 1.0))
    color_contrast = float(candidate_scores.get("color_contrast", 0.0))
    condition_contrast = float(candidate_scores.get("condition_contrast", 0.0))

    mandatory_satisfied: List[str] = []
    mandatory_violations: List[str] = []
    preferred_hits: List[str] = []
    preferred_misses: List[str] = []
    supportive_hits: List[str] = []

    for constraint in mandatory:
        kind = constraint.get("kind")
        value = constraint.get("value")
        if kind == "object":
            if object_score >= 0.58:
                _append(mandatory_satisfied, f"object={value}")
            else:
                _append(mandatory_violations, f"object={value}")
        elif kind == "color":
            if color_score >= 0.58 and color_contrast >= -0.03:
                _append(mandatory_satisfied, f"color={value}")
            else:
                _append(mandatory_violations, f"color={value}")
        elif kind == "condition":
            if condition_score >= 0.60 and condition_contrast >= -0.05:
                _append(mandatory_satisfied, f"condition={value}")
            else:
                _append(mandatory_violations, f"condition={value}")

    for constraint in preferred:
        kind = constraint.get("kind")
        value = constraint.get("value")
        if kind in {"relation", "region", "size_or_depth", "ordinal"}:
            if spatial_score >= 0.58:
                _append(preferred_hits, f"{kind}={value}")
            else:
                _append(preferred_misses, f"{kind}={value}")
        elif kind in {"anchor", "anchor2"}:
            if anchor_score >= 0.45:
                _append(preferred_hits, f"{kind}={value}")
            else:
                _append(preferred_misses, f"{kind}={value}")
        elif kind == "color":
            if color_score >= 0.55 and color_contrast >= -0.03:
                _append(preferred_hits, f"color={value}")
            else:
                _append(preferred_misses, f"color={value}")

    for constraint in supportive:
        kind = constraint.get("kind")
        value = constraint.get("value")
        if kind in {"region", "size_or_depth"} and spatial_score >= 0.52:
            _append(supportive_hits, f"{kind}={value}")

    contradiction_penalty = 0.0
    semantic_bonus = 0.0

    if mandatory_violations:
        contradiction_penalty += 0.16 * len(mandatory_violations)
    if preferred_misses:
        contradiction_penalty += 0.04 * len(preferred_misses)

    if mandatory_satisfied:
        semantic_bonus += 0.04 * len(mandatory_satisfied)
    if preferred_hits:
        semantic_bonus += 0.02 * len(preferred_hits)
    if supportive_hits:
        semantic_bonus += 0.01 * len(supportive_hits)

    if query_type == "condition-centric":
        if any(item.startswith("condition=") for item in mandatory_satisfied):
            semantic_bonus += 0.06
        if any(item.startswith("condition=") for item in mandatory_violations):
            contradiction_penalty += 0.10
        if clip_score < 0.42:
            contradiction_penalty += 0.04

    if query_type == "relation-centric" and anchor_score < 0.45:
        contradiction_penalty += 0.05

    return {
        "query_type": query_type,
        "mandatory_satisfied": mandatory_satisfied,
        "mandatory_violations": mandatory_violations,
        "preferred_hits": preferred_hits,
        "preferred_misses": preferred_misses,
        "supportive_hits": supportive_hits,
        "semantic_bonus": round(float(semantic_bonus), 4),
        "contradiction_penalty": round(float(contradiction_penalty), 4),
    }
