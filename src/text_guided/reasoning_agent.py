"""
Reasoning Agent (Agent 3): Explainable Decision Reasoning
Generates a natural language explanation of the pipeline's detection decision
using LLaVA-7B in text-only mode.

Called AFTER all detection steps complete, with pipeline results as input.
"""

from src.agents.vlm_backend import run_vlm


def reasoning_agent(pipeline_data):
    """
    Generate an explainable reasoning paragraph from pipeline results.

    Args:
        pipeline_data: dict with keys:
            - query: user's original prompt
            - scene_type: from scene agent
            - lighting: from scene agent
            - n_objects: number of objects in scene
            - reasoning: from attribute agent
            - ambiguity: from attribute agent
            - recommended_prompt: detection prompt used
            - n_candidates: from GroundingDINO
            - n_verified: after CLIP
            - n_rejected: CLIP rejections
            - clip_details: per-candidate CLIP scores
            - spatial_term: spatial filter used
            - n_selected: after spatial filter

    Returns:
        str: reasoning paragraph
    """
    # Build a structured summary of what happened
    query = pipeline_data.get("query", "unknown")
    scene_type = pipeline_data.get("scene_type", "road scene")
    lighting = pipeline_data.get("lighting", "unknown")
    n_objects = pipeline_data.get("n_objects", 0)
    reasoning = pipeline_data.get("reasoning", "")
    ambiguity = pipeline_data.get("ambiguity", "unknown")
    rec_prompt = pipeline_data.get("recommended_prompt", query)
    n_candidates = pipeline_data.get("n_candidates", 0)
    n_verified = pipeline_data.get("n_verified", 0)
    n_rejected = pipeline_data.get("n_rejected", 0)
    clip_details = pipeline_data.get("clip_details", "")
    spatial_term = pipeline_data.get("spatial_term", "none")
    n_selected = pipeline_data.get("n_selected", 0)

    context = f"""A multi-agent vision-language detection pipeline processed a road scene image with the following results:

Query: "{query}"
Scene Analysis: {scene_type} scene, {lighting} lighting, {n_objects} objects identified in the scene.
Attribute Matching: {reasoning}. Ambiguity level: {ambiguity}. Detection prompt refined to: "{rec_prompt}".
Object Detection: GroundingDINO detected {n_candidates} candidate region(s) matching the prompt.
Semantic Verification: CLIP verified {n_verified} candidate(s), rejected {n_rejected}. {clip_details}
Spatial Selection: Spatial filter '{spatial_term}' applied, {n_selected} object(s) selected.
Segmentation: SAM generated pixel-precise segmentation mask for the selected object."""

    prompt = f"""{context}

Based on the above pipeline results, write a clear reasoning paragraph that explains:
1. What the system found in the scene
2. How it identified the target object from the user's query
3. Why certain candidates were accepted or rejected
4. How the final object was selected and how confident the decision is

Write as a single cohesive paragraph. Be specific and reference the actual numbers. Do not use bullet points."""

    messages = [
        {"role": "user", "content": prompt}
    ]

    print("[i] Reasoning Agent: Generating explainable reasoning (LLaVA)...")
    try:
        response = run_vlm(messages, image_path=None)

        # Validate response
        if not response or len(response) < 20:
            print("[WARN] Reasoning agent returned too short response, using fallback")
            return _fallback_reasoning(pipeline_data)

        print("[OK] Reasoning complete")
        return response

    except Exception as e:
        print(f"[WARN] Reasoning agent failed: {e}, using fallback")
        return _fallback_reasoning(pipeline_data)


def _fallback_reasoning(data):
    """Rule-based fallback if LLaVA fails."""
    parts = []
    parts.append(
        f"The system analyzed a {data.get('scene_type', 'road')} scene "
        f"with {data.get('lighting', 'unknown')} lighting and identified "
        f"{data.get('n_objects', 0)} objects."
    )
    parts.append(
        f"For the query \"{data.get('query', '')}\", the attribute matching agent "
        f"determined: {data.get('reasoning', 'target matched')} "
        f"with {data.get('ambiguity', 'unknown')} ambiguity."
    )
    parts.append(
        f"GroundingDINO detected {data.get('n_candidates', 0)} candidates, "
        f"of which {data.get('n_verified', 0)} passed CLIP semantic verification."
    )
    if data.get('n_rejected', 0) > 0:
        parts.append(
            f"{data.get('n_rejected', 0)} candidate(s) were rejected due to "
            f"low visual similarity."
        )
    spatial = data.get('spatial_term', 'none')
    if spatial and spatial != 'none':
        parts.append(
            f"The spatial filter '{spatial}' selected "
            f"{data.get('n_selected', 0)} object(s) as the final detection."
        )
    return " ".join(parts)
