"""
Reliability / Confidence Decision Module

Centralizes the final decision policy for text-guided grounding so the project
can tune trustworthiness without rewriting the pipeline orchestration.
"""


def determine_match_decision(ranked_candidates):
    if not ranked_candidates:
        return {
            "state": "no_reliable_match",
            "confidence": 0.0,
            "selected_indices": [],
            "reason": "No candidates were available after grounding.",
        }

    top = ranked_candidates[0]
    top_score = float(top["scores"]["final_score"])
    second_score = float(ranked_candidates[1]["scores"]["final_score"]) if len(ranked_candidates) > 1 else 0.0
    margin = top_score - second_score
    clip_score = float(top["scores"].get("clip_score", 0.0))
    object_score = float(top["scores"].get("object_score", 0.0))
    spatial_score = float(top["scores"].get("spatial_score", 0.0))
    clip_passed = bool(top.get("clip_passed", False))

    strongest_violations = len(top.get("match_analysis", {}).get("violated_constraints", []))
    ambiguity_penalty = float(top.get("match_analysis", {}).get("ambiguity_penalty", 0.0))
    relation_uncertainty_penalty = float(top.get("match_analysis", {}).get("relation_uncertainty_penalty", 0.0))
    semantic_judgment = top.get("semantic_judgment", {}) or {}
    mandatory_violations = len(semantic_judgment.get("mandatory_violations", []))
    # `final_score` already includes ambiguity / relation penalties from the
    # ranking stage, so use it directly here to keep ranking and confidence in
    # sync.
    effective_confidence = max(0.0, top_score)

    if effective_confidence < 0.35:
        # Rescue simple cases where the pipeline has one clearly actionable
        # candidate but the aggregate score remains conservative.
        if clip_passed and clip_score >= 0.25 and object_score >= 0.40 and (
            len(ranked_candidates) == 1 or margin >= 0.08 or spatial_score >= 0.55
        ):
            return {
                "state": "closest_match",
                "confidence": round(max(effective_confidence, clip_score), 4),
                "selected_indices": [top["index"]],
                "reason": "A single CLIP-supported candidate was available even though the final trust score stayed conservative.",
            }
        return {
            "state": "no_reliable_match",
            "confidence": round(effective_confidence, 4),
            "selected_indices": [],
            "reason": "The best candidate remained too weak to trust.",
        }

    if effective_confidence >= 0.72 and margin >= 0.12 and strongest_violations == 0 and mandatory_violations == 0:
        return {
            "state": "exact_match",
            "confidence": round(effective_confidence, 4),
            "selected_indices": [top["index"]],
            "reason": "One candidate clearly satisfied the full-query constraints.",
        }

    if mandatory_violations > 0 and effective_confidence < 0.55:
        return {
            "state": "no_reliable_match",
            "confidence": round(effective_confidence, 4),
            "selected_indices": [],
            "reason": "Top candidates still violated mandatory query constraints.",
        }

    if len(ranked_candidates) > 1 and (
        margin <= 0.06 or ambiguity_penalty >= 0.08 or relation_uncertainty_penalty >= 0.06 or mandatory_violations > 0
    ):
        selected = [ranked_candidates[0]["index"], ranked_candidates[1]["index"]]
        return {
            "state": "ambiguous_match",
            "confidence": round(effective_confidence, 4),
            "selected_indices": selected,
            "reason": "The top candidates are too close or the query remains ambiguous.",
        }

    return {
        "state": "closest_match",
        "confidence": round(effective_confidence, 4),
        "selected_indices": [top["index"]],
        "reason": "The best candidate matched partially, but some constraints remained weaker.",
    }
