#!/usr/bin/env python3
"""
Agent 4: Visual Appearance Evaluator (LLaVA-7B Version)

Detects condition-based visual anomalies such as color, texture,
shape, and material irregularities in road environments.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import time
import argparse
from typing import Dict
from datetime import datetime

from src.agents.vlm_backend import run_vlm



# =========================
# SYSTEM PROMPT
# =========================
VISUAL_APPEARANCE_SYSTEM_PROMPT = """
You are a Visual Appearance Evaluator specializing in detecting anomalies
based on visual characteristics such as color, texture, shape, and material.

CORE RESPONSIBILITIES:
1. Analyze color inconsistencies
2. Examine texture and material irregularities
3. Identify shape deformations
4. Detect condition-based hazards

OUTPUT FORMAT (JSON ONLY):
{
    "visual_analysis": {
        "lighting_conditions": "",
        "overall_visual_quality": "",
        "detected_objects": []
    },
    "color_anomalies": [],
    "texture_irregularities": [],
    "shape_deformations": [],
    "material_condition_issues": [],
    "visual_integrity_assessment": {
        "overall_condition": "",
        "primary_visual_concerns": [],
        "hazard_indicators": [],
        "condition_based_issues": []
    },
    "visual_confidence": 0.0
}
"""





# =========================
# AGENT CLASS
# =========================
class VisualAppearanceEvaluator:
    def __init__(self, delay_between_requests: float = 2.0):
        self.delay_between_requests = delay_between_requests

        print("🚀 AGENT 4: Visual Appearance Evaluator (LLaVA-7B)")
        print("👁️ Detecting condition-based visual anomalies")

    def analyze_image(self, image_path: str) -> Dict:
        print(f"🔍 Analyzing visual appearance: {image_path}")

        try:
            messages = [
                {"role": "system", "content": VISUAL_APPEARANCE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Analyze this road scene image for visual appearance anomalies, "
                        "including color, texture, shape, and material condition issues."
                    )
                }
            ]

            output = run_vlm(messages, image_path=image_path)
            return self._parse_json_response(output)

        except Exception as e:
            print(f"❌ Error: {e}")
            return {"error": str(e)}

    def _parse_json_response(self, response: str) -> Dict:
        """Robustly parse JSON from LLaVA output"""
        import re
        try:
            return json.loads(response)
        except Exception:
            pass
        code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1).strip())
            except Exception:
                pass
        brace_start = response.find('{')
        if brace_start != -1:
            depth = 0
            for i in range(brace_start, len(response)):
                if response[i] == '{': depth += 1
                elif response[i] == '}': depth -= 1
                if depth == 0:
                    try:
                        return json.loads(response[brace_start:i+1])
                    except Exception:
                        break
        cleaned = response.strip()
        cleaned = re.sub(r'^[^{]*', '', cleaned)
        cleaned = re.sub(r'[^}]*$', '', cleaned)
        if cleaned:
            try:
                return json.loads(cleaned)
            except Exception:
                pass
        return {"error": "JSON parsing failed", "raw_response": response}

    def process_batch(self, image_directory: str, output_file: str):
        image_files = [
            f for f in os.listdir(image_directory)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_files:
            print("❌ No images found")
            return

        results = {
            "metadata": {
                "agent": "visual_appearance_evaluator",
                "agent_number": 4,
                "model": "llava-7b",
                "date": datetime.now().isoformat(),
                "total_images": len(image_files)
            },
            "results": []
        }

        for i, img_name in enumerate(image_files):
            print(f"\n📸 {i+1}/{len(image_files)}: {img_name}")
            img_path = os.path.join(image_directory, img_name)

            start = time.time()
            analysis = self.analyze_image(img_path)
            duration = time.time() - start

            results["results"].append({
                "image": img_name,
                "processing_time": duration,
                "visual_appearance_analysis": analysis
            })

            print(f"✅ Done in {duration:.2f}s")

            if i < len(image_files) - 1:
                time.sleep(self.delay_between_requests)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to {output_file}")


# =========================
# CLI ENTRY
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent 4: Visual Appearance Evaluator (LLaVA-7B)")
    parser.add_argument("--image_directory", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--delay_between_requests", type=float, default=60.0)

    args = parser.parse_args()

    agent = VisualAppearanceEvaluator(delay_between_requests=args.delay_between_requests)
    agent.process_batch(args.image_directory, args.output_file)

    print("\n✅ Agent 4 completed successfully")
