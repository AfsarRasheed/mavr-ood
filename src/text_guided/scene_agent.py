"""
Scene Understanding Agent (Step 1)
Uses LLaVA to analyze the image and list all visible objects
with their positions, colors, and sizes.
"""

import re
import json


SCENE_SYSTEM_PROMPT = """You are an expert scene analyzer. Analyze the given image and list ALL visible objects.
For each object, provide:
1. Object name/type
2. Approximate position (left, center-left, center, center-right, right)
3. Color/appearance
4. Size relative to scene (small, medium, large)

You MUST return ONLY valid JSON with NO other text before or after it.
Do NOT include any explanation, markdown, or commentary. ONLY the JSON object.

{
  "scene_type": "description of the scene",
  "lighting": "bright/dim/dark/night",
  "total_objects": N,
  "objects": [
    {"name": "red car", "position": "left", "color": "red", "size": "medium"},
    {"name": "pedestrian", "position": "center", "color": "dark clothing", "size": "small"}
  ]
}

List EVERY visible object. Be thorough."""


def scene_understanding(image_path):
    """
    Run LLaVA scene analysis to list all objects in the image.

    Args:
        image_path: path to the image file

    Returns:
        dict with scene analysis results
    """
    from src.agents.vlm_backend import run_vlm

    messages = [
        {"role": "system", "content": SCENE_SYSTEM_PROMPT},
        {"role": "user", "content": "Analyze this image and list all visible objects with their positions and attributes. Return ONLY valid JSON, nothing else."}
    ]

    print("[i] Text-Guided: Running scene understanding (LLaVA)...")
    output = run_vlm(messages, image_path=image_path)
    print(f"[OK] Scene analysis complete")

    return parse_json_response(output)


def parse_json_response(response):
    """Robustly parse JSON from LLaVA output."""
    if not response or not response.strip():
        return {"scene_type": "unknown", "objects": [], "raw_output": "(empty response)"}

    # Clean up common LLaVA issues
    response = response.replace("\\_", "_")
    response = response.replace("\\n", "\n")
    response = response.strip()

    # Remove markdown code fences if wrapping the entire response
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    # Try direct parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Fix trailing commas (common LLaVA error)
    cleaned = re.sub(r',\s*([}\]])', r'\1', response)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block from mixed text
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{[\s\S]*\}',
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            candidate = match.group(1) if '```' in pattern else match.group(0)
            # Fix trailing commas in extracted block too
            candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, IndexError):
                continue

    # Fallback — store raw output for debugging
    print(f"[WARN] Could not parse LLaVA JSON, using fallback")
    return {"scene_type": "unknown", "objects": [], "raw_output": response[:500]}
