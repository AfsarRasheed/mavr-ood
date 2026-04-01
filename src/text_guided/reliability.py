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

    strongest_violations = len(top.get("match_analysis", {}).get("violated_constraints", []))
    ambiguity_penalty = float(top.get("match_analysis", {}).get("ambiguity_penalty", 0.0))
    effective_confidence = max(0.0, top_score - ambiguity_penalty)

    if effective_confidence < 0.35:
        return {
            "state": "no_reliable_match",
            "confidence": round(effective_confidence, 4),
            "selected_indices": [],
            "reason": "The best candidate remained too weak to trust.",
        }

    if effective_confidence >= 0.72 and margin >= 0.12 and strongest_violations == 0:
        return {
            "state": "exact_match",
            "confidence": round(effective_confidence, 4),
            "selected_indices": [top["index"]],
            "reason": "One candidate clearly satisfied the full-query constraints.",
        }

    if len(ranked_candidates) > 1 and (margin <= 0.06 or ambiguity_penalty >= 0.08):
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
