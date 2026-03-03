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

Return your analysis as valid JSON:
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
        {"role": "user", "content": "Analyze this image and list all visible objects with their positions and attributes. Return valid JSON only."}
    ]

    print("[i] Text-Guided: Running scene understanding (LLaVA)...")
    output = run_vlm(messages, image_path=image_path)
    print(f"[OK] Scene analysis complete")

    return parse_json_response(output)


def parse_json_response(response):
    """Robustly parse JSON from LLaVA output."""
    # Clean LaTeX-style underscores
    response = response.replace("\\_", "_")

    # Try direct parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{[\s\S]*\}',
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1) if '```' in pattern else match.group(0))
            except (json.JSONDecodeError, IndexError):
                continue

    # Fallback
    return {"scene_type": "unknown", "objects": [], "raw_output": response}
