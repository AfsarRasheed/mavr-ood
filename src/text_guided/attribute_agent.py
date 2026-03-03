"""
Attribute Matching Agent (Step 2)
Uses LLaVA to reason about which object from the scene analysis
best matches the user's query.
"""

import json

from src.text_guided.scene_agent import parse_json_response


ATTRIBUTE_AGENT_PROMPT = """You are an expert object matching agent. You will be given:
1. A scene analysis listing all objects in an image
2. A user's search query describing a specific object

Your task: Analyze which object(s) from the scene best match the user's query.
Consider: color, type, position, size, and any other attributes mentioned.

Return your analysis as valid JSON with these fields:
- query: the user's original query
- matched_objects: list of matches with name, position, confidence (high/medium/low), reason
- recommended_prompt: a short phrase describing ONLY the object type and color (e.g. "white car", "large zebra", "red truck"). Do NOT write instructions or descriptions, just the object name.
- ambiguity: none, low, or high
- reasoning: brief explanation

Example for query "the white car on the left":
{
  "query": "the white car on the left",
  "matched_objects": [
    {"name": "white sedan", "position": "left", "confidence": "high", "reason": "matches color and position"}
  ],
  "recommended_prompt": "white car",
  "ambiguity": "none",
  "reasoning": "Only one white car visible on the left side"
}"""


def attribute_matching_agent(image_path, scene_result, user_prompt):
    """
    Run LLaVA to reason about which object matches the user's query.

    Args:
        image_path: path to image file
        scene_result: dict from scene_understanding()
        user_prompt: original user query

    Returns:
        dict with matching analysis
    """
    from src.agents.vlm_backend import run_vlm

    # Build context from scene analysis
    scene_text = json.dumps(scene_result, indent=2) if isinstance(scene_result, dict) else str(scene_result)

    messages = [
        {"role": "system", "content": ATTRIBUTE_AGENT_PROMPT},
        {"role": "user", "content": f"Scene Analysis:\n{scene_text}\n\nUser Query: \"{user_prompt}\"\n\nWhich object(s) match this query? Return valid JSON only."}
    ]

    print("[i] Text-Guided: Running attribute matching agent (LLaVA)...")
    output = run_vlm(messages, image_path=image_path)
    print(f"[OK] Attribute matching complete")

    result = parse_json_response(output)

    # Use the agent's recommended prompt if available
    if isinstance(result, dict) and result.get("recommended_prompt"):
        print(f"[i] Agent recommended prompt: '{result['recommended_prompt']}'")

    return result
