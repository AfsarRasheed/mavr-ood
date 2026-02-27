#!/usr/bin/env python3
"""
Agent 5: Reasoning Synthesizer (LLaVA-7B Version)

Integrates multi-agent findings and generates optimized GroundedSAM prompts.
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
REASONING_SYNTHESIZER_SYSTEM_PROMPT = """
You are a Reasoning Synthesizer responsible for integrating multi-agent findings
into coherent final judgments and generating optimized prompts for GroundedSAM.

CRITICAL PROMPT RULES:
- V1: EXACTLY adjective + noun
- V2: EXACTLY single noun
- Choose ONLY the TOP 1 most anomalous object
- Priority: Animals > Misplaced vehicles > Obstacles > Others

OUTPUT JSON ONLY.
"""


# =========================
# AGENT CLASS
# =========================
class ReasoningSynthesizer:
    def __init__(self, delay_between_requests: float = 2.0):
        self.delay_between_requests = delay_between_requests

        print("🚀 AGENT 5: Reasoning Synthesizer (LLaVA-7B)")
        print("🔗 Integrating multi-agent reasoning")

    def synthesize_analysis(self, agent_results: Dict) -> Dict:
        try:
            synthesis_prompt = f"""
Synthesize the following agent outputs and produce the final decision.

AGENT 1 (Scene Context):
{json.dumps(agent_results.get('agent1_scene_context', {}), indent=2)}

AGENT 2 (Spatial Anomaly):
{json.dumps(agent_results.get('agent2_spatial_anomaly', {}), indent=2)}

AGENT 3 (Semantic Inconsistency):
{json.dumps(agent_results.get('agent3_semantic_inconsistency', {}), indent=2)}

AGENT 4 (Visual Appearance):
{json.dumps(agent_results.get('agent4_visual_appearance', {}), indent=2)}

Return ONLY valid JSON following the required schema.
"""

            messages = [
                {"role": "system", "content": REASONING_SYNTHESIZER_SYSTEM_PROMPT},
                {"role": "user", "content": synthesis_prompt}
            ]

            output = run_vlm(messages)
            return self._parse_json_response(output)

        except Exception as e:
            return {"error": str(e)}

    def _parse_json_response(self, response: str) -> Dict:
        """Robustly parse JSON from LLaVA output"""
        import re
        # Pre-clean: Fix LLaVA's LaTeX-style escaped underscores
        response = response.replace("\\_", "_")
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

    def load_json(self, path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def process_batch_synthesis(
        self,
        agent1_file: str,
        agent2_file: str,
        agent3_file: str,
        agent4_file: str,
        output_file: str
    ):
        agent1 = self.load_json(agent1_file)["results"]
        agent2 = self.load_json(agent2_file)["results"]
        agent3 = self.load_json(agent3_file)["results"]
        agent4 = self.load_json(agent4_file)["results"]

        a1 = {x["image"]: x for x in agent1}
        a2 = {x["image"]: x for x in agent2}
        a3 = {x["image"]: x for x in agent3}
        a4 = {x["image"]: x for x in agent4}

        common_files = set(a1) & set(a2) & set(a3) & set(a4)

        results = {
            "metadata": {
                "agent": "reasoning_synthesizer",
                "agent_number": 5,
                "model": "llava-7b",
                "total_images": len(common_files),
                "date": datetime.now().isoformat()
            },
            "results": []
        }

        for i, fname in enumerate(sorted(common_files)):
            print(f"\n📸 Synthesizing {i+1}/{len(common_files)}: {fname}")

            combined = {
                "agent1_scene_context": a1[fname].get("scene_context_analysis", a1[fname]),
                "agent2_spatial_anomaly": a2[fname].get("spatial_anomaly_analysis", a2[fname]),
                "agent3_semantic_inconsistency": a3[fname].get("semantic_inconsistency_analysis", a3[fname]),
                "agent4_visual_appearance": a4[fname].get("visual_appearance_analysis", a4[fname])
            }

            start = time.time()
            synthesis = self.synthesize_analysis(combined)
            duration = time.time() - start

            results["results"].append({
                "image": fname,
                "synthesis_result": synthesis,
                "processing_time": duration
            })

            print("  📝 V1:", synthesis.get("grounded_sam_prompts", {}).get("prompt_v1", "N/A"))
            print("  🎯 Confidence:", synthesis.get("overall_confidence", 0.0))

            time.sleep(self.delay_between_requests)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Final synthesis saved to {output_file}")


# =========================
# CLI ENTRY
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent 5: Reasoning Synthesizer (LLaVA-7B)")
    parser.add_argument("--agent1_file", required=True)
    parser.add_argument("--agent2_file", required=True)
    parser.add_argument("--agent3_file", required=True)
    parser.add_argument("--agent4_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--delay_between_requests", type=float, default=30.0)

    args = parser.parse_args()

    agent = ReasoningSynthesizer(delay_between_requests=args.delay_between_requests)
    agent.process_batch_synthesis(
        args.agent1_file,
        args.agent2_file,
        args.agent3_file,
        args.agent4_file,
        args.output_file
    )

    print("\n✅ Agent 5 completed successfully")
