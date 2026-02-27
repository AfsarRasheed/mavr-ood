#!/usr/bin/env python3
"""
Agent 1: Scene Context Analyzer (LLaVA-7B Version)

Establishes contextual baselines for road environments using LLaVA-7B.
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
SCENE_CONTEXT_SYSTEM_PROMPT = """
You are a Scene Context Analyzer that establishes contextual baselines for road environments.
Your primary role is to understand what constitutes "normal" for the given road environment.

CORE RESPONSIBILITIES:
1. Determine scene type (urban/rural/highway/intersection/residential)
2. Assess environmental conditions (weather, lighting, time of day)
3. Identify expected object inventory based on context
4. Establish context-dependent normality criteria

OUTPUT FORMAT (JSON ONLY):
{
    "scene_analysis": {
        "scene_type": "",
        "road_infrastructure": "",
        "environmental_conditions": {
            "weather": "",
            "lighting": "",
            "time_period": ""
        }
    },
    "contextual_baseline": {
        "expected_objects": [],
        "expected_behaviors": [],
        "infrastructure_elements": [],
        "typical_layout": ""
    },
    "environmental_factors": {
        "visibility_conditions": "",
        "seasonal_indicators": "",
        "special_circumstances": ""
    },
    "normality_criteria": {
        "object_appropriateness": "",
        "spatial_expectations": "",
        "behavioral_norms": ""
    },
    "context_confidence": 0.0
}
"""





# =========================
# AGENT CLASS
# =========================
class SceneContextAnalyzer:
    def __init__(self, delay_between_requests: float = 2.0):
        self.delay_between_requests = delay_between_requests

        print("🚀 AGENT 1: Scene Context Analyzer (LLaVA-7B)")
        print("🧠 Using LLaVA-7B for contextual reasoning")

    def analyze_image(self, image_path: str) -> Dict:
        """Analyze scene context using LLaVA-7B"""

        print(f"🔍 Analyzing: {image_path}")

        try:
            messages = [
                {"role": "system", "content": SCENE_CONTEXT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Analyze the following road scene image and return ONLY valid JSON."
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

        # Pre-clean: Fix LLaVA's LaTeX-style escaped underscores (\_  ->  _)
        response = response.replace("\\_", "_")

        # Strategy 1: Try direct parse
        try:
            return json.loads(response)
        except Exception:
            pass

        # Strategy 2: Extract from code blocks (```json ... ``` or ``` ... ```)
        code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1).strip())
            except Exception:
                pass

        # Strategy 3: Find JSON object by matching braces
        brace_start = response.find('{')
        if brace_start != -1:
            depth = 0
            for i in range(brace_start, len(response)):
                if response[i] == '{':
                    depth += 1
                elif response[i] == '}':
                    depth -= 1
                if depth == 0:
                    try:
                        return json.loads(response[brace_start:i+1])
                    except Exception:
                        break

        # Strategy 4: Clean common LLaVA artifacts and retry
        cleaned = response.strip()
        cleaned = re.sub(r'^[^{]*', '', cleaned)  # remove text before first {
        cleaned = re.sub(r'[^}]*$', '', cleaned)   # remove text after last }
        if cleaned:
            try:
                return json.loads(cleaned)
            except Exception:
                pass

        return {
            "error": "JSON parsing failed",
            "raw_response": response
        }

    def process_batch(self, image_directory: str, output_file: str):
        """Process all images in directory"""

        image_files = [
            f for f in os.listdir(image_directory)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_files:
            print("❌ No images found")
            return

        results = {
            "metadata": {
                "agent": "scene_context_analyzer",
                "agent_number": 1,
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
                "scene_context_analysis": analysis
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
    parser = argparse.ArgumentParser(description="Agent 1: Scene Context Analyzer (LLaVA-7B)")
    parser.add_argument("--image_directory", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--delay_between_requests", type=float, default=60.0)

    args = parser.parse_args()

    agent = SceneContextAnalyzer(delay_between_requests=args.delay_between_requests)
    agent.process_batch(args.image_directory, args.output_file)

    print("\n✅ Agent 1 completed successfully")
