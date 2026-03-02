#!/usr/bin/env python3
"""
Agent 3: Semantic Inconsistency Analyzer (LLaVA-7B Version)

Evaluates domain appropriateness and contextual fitness of objects
within road environments.
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
SEMANTIC_INCONSISTENCY_SYSTEM_PROMPT = """
You are a Semantic Inconsistency Analyzer for road environments.

YOUR JOB: Find objects that do NOT belong on a road. Animals (horses, cows, zebras, deer, dogs), fallen debris, unusual vehicles, and any non-standard road objects are INAPPROPRIATE and must be flagged.

CATEGORIZATION RULES:
- road_appropriate: cars, trucks, buses, traffic signs, lane markings, pedestrians on sidewalks
- questionable: pedestrians on the road, bicycles in car lanes
- inappropriate: ANY animal on or near the road, debris, fallen objects, construction materials outside work zones

CRITICAL INSTRUCTIONS:
1. Each object must appear in EXACTLY ONE category (road_appropriate OR questionable OR inappropriate). Never put the same object in two categories.
2. You MUST set semantic_confidence to a number between 0.1 and 1.0 based on how confident you are. Do NOT leave it as 0.0.
3. If you see animals on the road, they are ALWAYS inappropriate and should be the primary concern.

OUTPUT FORMAT (respond with ONLY this JSON, no other text):
{
    "semantic_analysis": {
        "detected_objects": ["object1", "object2"],
        "object_categorization": {
            "road_appropriate": ["object1"],
            "questionable": [],
            "inappropriate": ["object2"]
        }
    },
    "domain_violations": ["description of each violation"],
    "appropriateness_assessment": ["assessment of each inappropriate object"],
    "safety_implications": {
        "immediate_hazards": ["list of hazards"],
        "regulatory_violations": ["list of violations"],
        "functional_conflicts": ["list of conflicts"]
    },
    "semantic_reasoning": {
        "overall_assessment": "one paragraph summary",
        "primary_concerns": ["concern1", "concern2"],
        "context_considerations": "additional context"
    },
    "semantic_confidence": 0.85
}
"""





# =========================
# AGENT CLASS
# =========================
class SemanticInconsistencyAnalyzer:
    def __init__(self, delay_between_requests: float = 2.0):
        self.delay_between_requests = delay_between_requests

        print("🚀 AGENT 3: Semantic Inconsistency Analyzer (LLaVA-7B)")
        print("🧠 Evaluating domain appropriateness")

    def analyze_image(self, image_path: str) -> Dict:
        print(f"🔍 Analyzing semantic inconsistencies: {image_path}")

        try:
            messages = [
                {"role": "system", "content": SEMANTIC_INCONSISTENCY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Analyze this road scene image for semantic inconsistencies and "
                        "domain appropriateness violations."
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
        # Pre-clean: Fix LLaVA's LaTeX-style escaped underscores
        response = response.replace("\\_", "_")
        
        # Pre-clean: Fix common LLaVA bracket/quote mistakes
        # Fix: "some string value"], → "some string value",
        response = re.sub(r'"\]\s*,', '",', response)
        # Fix: "some string value"] → "some string value"
        response = re.sub(r'"\]\s*\n', '"\n', response)
        # Fix: trailing comma before closing brace/bracket
        response = re.sub(r',\s*}', '}', response)
        response = re.sub(r',\s*]', ']', response)
        
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
        # Strategy 5: Truncated JSON recovery — close unfinished JSON
        if brace_start is not None and brace_start != -1:
            truncated = response[brace_start:]
            truncated = re.sub(r',\s*"[^"]*$', '', truncated)
            truncated = re.sub(r':\s*"[^"]*$', ': ""', truncated)
            open_braces = truncated.count('{') - truncated.count('}')
            open_brackets = truncated.count('[') - truncated.count(']')
            truncated += ']' * max(0, open_brackets)
            truncated += '}' * max(0, open_braces)
            try:
                return json.loads(truncated)
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
                "agent": "semantic_inconsistency_analyzer",
                "agent_number": 3,
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
                "semantic_inconsistency_analysis": analysis
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
    parser = argparse.ArgumentParser(description="Agent 3: Semantic Inconsistency Analyzer (LLaVA-7B)")
    parser.add_argument("--image_directory", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--delay_between_requests", type=float, default=60.0)

    args = parser.parse_args()

    agent = SemanticInconsistencyAnalyzer(delay_between_requests=args.delay_between_requests)
    agent.process_batch(args.image_directory, args.output_file)

    print("\n✅ Agent 3 completed successfully")
