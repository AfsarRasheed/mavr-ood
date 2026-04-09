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

    context = f"""A vision-language system analyzed a road scene image to find a specific object. Here is what happened:

User's query: "{query}"
Scene: A {scene_type} scene with {lighting} lighting. {n_objects} objects were found in the scene.
Attribute analysis: {reasoning}. Ambiguity: {ambiguity}. The search was refined to look for: "{rec_prompt}".
Detection: {n_candidates} candidate object(s) were found matching the description.
Verification: {n_verified} candidate(s) were confirmed as visually matching, {n_rejected} were rejected because they did not look similar enough. {clip_details}
Spatial selection: The '{spatial_term}' positioning rule was used, and {n_selected} object(s) were selected as the final result.
Segmentation: A precise outline mask was generated for the selected object."""

    prompt = f"""{context}

Write a single, clear reasoning paragraph that a non-technical person can understand. Your paragraph should cover:
1. Briefly describe the scene (road type, lighting, what objects are present)
2. Explain how the system understood the user's query and what attributes it looked for
3. Explain which candidates matched and which didn't in plain language (do NOT mention raw similarity scores like 0.283 — instead say "strong match", "weak match", "high confidence", etc.)
4. Describe what the final detected object looks like — its type, color, approximate size, position in the scene, and any notable visual characteristics
5. State how confident the system is in the final result

IMPORTANT RULES:
- Do NOT mention model names like GroundingDINO, CLIP, SAM, or LLaVA. Use plain terms like "the object detector", "visual verification", "segmentation".
- Do NOT include raw numerical scores. Convert them to natural language (e.g., "high confidence", "strong visual match", "moderate certainty").
- Write in a clear, professional tone as if explaining to someone viewing the result.
- Write as one cohesive paragraph. Do not use bullet points or numbered lists."""

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
    query = data.get('query', '')
    scene_type = data.get('scene_type', 'road')
    lighting = data.get('lighting', 'unknown')
    n_objects = data.get('n_objects', 0)
    reasoning = data.get('reasoning', 'target matched')
    ambiguity = data.get('ambiguity', 'unknown')
    n_candidates = data.get('n_candidates', 0)
    n_verified = data.get('n_verified', 0)
    n_rejected = data.get('n_rejected', 0)
    spatial = data.get('spatial_term', 'none')
    n_selected = data.get('n_selected', 0)

    # Confidence in natural language
    if ambiguity == 'low':
        confidence = "high confidence"
    elif ambiguity == 'medium':
        confidence = "moderate confidence"
    else:
        confidence = "some uncertainty"

    parts = []
    parts.append(
        f"The system examined a {scene_type} scene with {lighting} lighting "
        f"and identified {n_objects} objects in the image."
    )
    parts.append(
        f"To locate \"{query}\", the system analyzed the key visual attributes "
        f"and determined: {reasoning}."
    )
    if n_candidates > 0:
        parts.append(
            f"The object detector found {n_candidates} candidate(s) matching "
            f"the description, and visual verification confirmed {n_verified} "
            f"as a strong match."
        )
    if n_rejected > 0:
        parts.append(
            f"{n_rejected} candidate(s) were filtered out because they did not "
            f"visually resemble the target closely enough."
        )
    if spatial and spatial != 'none':
        parts.append(
            f"Based on the \"{spatial}\" positioning cue from the query, "
            f"{n_selected} object(s) were selected as the final detection."
        )
    parts.append(
        f"The system has {confidence} in this result, and a precise "
        f"segmentation mask was generated for the detected object."
    )
    return " ".join(parts)
