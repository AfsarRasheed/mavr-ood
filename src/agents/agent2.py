#!/usr/bin/env python3
"""
Agent 2: Spatial Anomaly Detector (LLaVA-7B Version)

Detects spatial positioning violations and traffic flow anomalies.
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
SPATIAL_ANOMALY_SYSTEM_PROMPT = """
You are a Spatial Anomaly Detector for road scenes.

Analyze object positioning relative to road infrastructure.
Identify objects that should NOT be on the road.

You MUST output ONLY this exact JSON structure:
{
    "objects_on_road": "list the objects you see on the road",
    "positioning_violations": "describe any objects in wrong positions",
    "traffic_disruptions": "describe any traffic flow issues",
    "safety_hazards": "describe safety concerns",
    "spatial_confidence": 0.95
}

OUTPUT ONLY THE JSON ABOVE. NO OTHER TEXT.
CRITICAL: `spatial_confidence` MUST be a number between 0.0 and 1.0. DO NOT use quotes around it.
"""





# =========================
# AGENT CLASS
# =========================
class SpatialAnomalyDetector:
    def __init__(self, delay_between_requests: float = 2.0):
        self.delay_between_requests = delay_between_requests

        print("🚀 AGENT 2: Spatial Anomaly Detector (LLaVA-7B)")
        print("📐 Detecting spatial positioning violations")

    def analyze_image(self, image_path: str) -> Dict:
        print(f"🔍 Analyzing spatial anomalies: {image_path}")

        try:
            messages = [
                {"role": "system", "content": SPATIAL_ANOMALY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Analyze this road scene image for spatial anomalies and positioning violations."
                }
            ]

            output = run_vlm(messages, image_path=image_path)
            return self._parse_json_response(output)

        except Exception as e:
            print(f"❌ Error: {e}")
            return {"error": str(e)}

    def _parse_json_response(self, response: str) -> Dict:
        """Robustly parse JSON from LLaVA output"""
        parsed = None
        import re
        # Pre-clean: Fix LLaVA's LaTeX-style escaped underscores
        response = response.replace("\\_", "_")
        try:
            parsed = json.loads(response)
        except Exception:
            pass
        code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if code_block:
            try:
                parsed = json.loads(code_block.group(1).strip())
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
                        parsed = json.loads(response[brace_start:i+1])
                    except Exception:
                        break
        cleaned = response.strip()
        cleaned = re.sub(r'^[^{]*', '', cleaned)
        cleaned = re.sub(r'[^}]*$', '', cleaned)
        if cleaned and not parsed:
            try:
                parsed = json.loads(cleaned)
            except Exception:
                pass
        # Strategy 5: Truncated JSON recovery — close unfinished JSON
        if brace_start is not None and brace_start != -1:
            truncated = response[brace_start:]
            # Remove trailing incomplete string (cut mid-value)
            truncated = re.sub(r',\s*"[^"]*$', '', truncated)
            truncated = re.sub(r':\s*"[^"]*$', ': ""', truncated)
            # Count open braces/brackets and close them
            open_braces = truncated.count('{') - truncated.count('}')
            open_brackets = truncated.count('[') - truncated.count(']')
            truncated += ']' * max(0, open_brackets)
            truncated += '}' * max(0, open_braces)
            try:
                parsed = json.loads(truncated)
            except Exception:
                pass
                
        # Enforce float conversion just in case LLaVA returned a string like "0.9"
        if isinstance(parsed, dict) and "spatial_confidence" in parsed:
            try:
                parsed["spatial_confidence"] = float(parsed["spatial_confidence"])
            except:
                pass
                
        if parsed is not None:
            return parsed
            
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
                "agent": "spatial_anomaly_detector",
                "agent_number": 2,
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
                "spatial_anomaly_analysis": analysis
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
    parser = argparse.ArgumentParser(description="Agent 2: Spatial Anomaly Detector (LLaVA-7B)")
    parser.add_argument("--image_directory", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--delay_between_requests", type=float, default=60.0)

    args = parser.parse_args()

    agent = SpatialAnomalyDetector(delay_between_requests=args.delay_between_requests)
    agent.process_batch(args.image_directory, args.output_file)

    print("\n✅ Agent 2 completed successfully")
