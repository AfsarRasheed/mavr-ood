"""
Semantic Query Controller

Builds a lightweight semantic control plan from the parsed query so the
grounding pipeline can distinguish mandatory constraints from supportive ones.
"""

from typing import Dict, List


RELATIONAL_SPATIAL = {"next_to", "behind", "in_front", "above", "below", "between"}
REGION_SPATIAL = {"left", "right", "center", "top", "bottom"}
SIZE_SPATIAL = {"largest", "smallest", "nearest", "farthest", "ahead"}


def _make_constraint(kind: str, value: str, importance: str) -> Dict[str, str]:
    return {"kind": kind, "value": value, "importance": importance}


def build_semantic_plan(user_prompt: str, parsed: dict, attr_result: dict | None = None,
                        scene_result: dict | None = None) -> Dict[str, object]:
    attrs = parsed.get("attributes") or {}
    target_object = (parsed.get("target_object") or parsed.get("object_prompt") or "object").strip()
    color = (attrs.get("color") or "").strip()
    condition = (attrs.get("condition") or "").strip()
    spatial = parsed.get("spatial")
    anchor = parsed.get("anchor")
    anchor2 = parsed.get("anchor2")
    ordinal = parsed.get("ordinal")

    query_type = "object-centric"
    if condition:
        query_type = "condition-centric"
    elif spatial in RELATIONAL_SPATIAL or anchor:
        query_type = "relation-centric"
    elif spatial or ordinal:
        query_type = "spatial-centric"

    mandatory: List[Dict[str, str]] = []
    preferred: List[Dict[str, str]] = []
    supportive: List[Dict[str, str]] = []

    if target_object:
        mandatory.append(_make_constraint("object", target_object, "mandatory"))

    if color:
        importance = "mandatory" if query_type in {"object-centric", "spatial-centric"} else "preferred"
        bucket = mandatory if importance == "mandatory" else preferred
        bucket.append(_make_constraint("color", color, importance))

    if condition:
        mandatory.append(_make_constraint("condition", condition, "mandatory"))

    if spatial:
        if spatial in RELATIONAL_SPATIAL:
            preferred.append(_make_constraint("relation", spatial, "preferred"))
        elif spatial in REGION_SPATIAL:
            if query_type == "condition-centric":
                supportive.append(_make_constraint("region", spatial, "supportive"))
            else:
                preferred.append(_make_constraint("region", spatial, "preferred"))
        elif spatial in SIZE_SPATIAL:
            supportive.append(_make_constraint("size_or_depth", spatial, "supportive"))

    if anchor:
        preferred.append(_make_constraint("anchor", str(anchor), "preferred"))
    if anchor2:
        preferred.append(_make_constraint("anchor2", str(anchor2), "preferred"))
    if ordinal:
        preferred.append(_make_constraint("ordinal", str(ordinal), "preferred"))

    # Keep detector prompting conservative, but preserve the strongest semantic
    # cues for condition-heavy queries so the detector phrase does not collapse
    # to a generic object label.
    detector_prompt_parts = []
    if condition:
        detector_prompt_parts.append(condition)
    if color:
        detector_prompt_parts.append(color)
    if target_object:
        detector_prompt_parts.append(target_object)
    detector_prompt = " ".join(part for part in detector_prompt_parts if part).strip() or parsed.get("object_prompt") or user_prompt

    plan = {
        "query": user_prompt,
        "query_type": query_type,
        "mandatory_constraints": mandatory,
        "preferred_constraints": preferred,
        "supportive_constraints": supportive,
        "detector_prompt": detector_prompt,
        "full_query_prompt": parsed.get("full_query_prompt") or user_prompt,
    }

    if isinstance(attr_result, dict):
        plan["attribute_agent_ambiguity"] = attr_result.get("ambiguity", "unknown")
    if isinstance(scene_result, dict):
        plan["scene_type"] = scene_result.get("scene_type", "unknown")

    return plan
