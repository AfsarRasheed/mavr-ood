#!/usr/bin/env python3
"""
Master Script for Multi-Agent OOD Detection Framework (LLaVA-7B Version)

This script runs all 5 agents sequentially using local LLaVA-7B VLM:
1. Agent 1: Scene Context Analyzer
2. Agent 2: Spatial Anomaly Detector
3. Agent 3: Semantic Inconsistency Analyzer
4. Agent 4: Visual Appearance Evaluator
5. Agent 5: Reasoning Synthesizer

Usage:
    python src/agents/run_all_agents.py --image_dir /path/to/images --output_dir ./output --delay 2
"""

import os
import sys
import argparse
import time
from datetime import datetime


def run_agent(agent_script: str, description: str, args_str: str) -> bool:
    """Run an individual agent script with arguments"""
    print(f"\n{'='*80}")
    print(f"🚀 STARTING: {description}")
    print(f"📄 Script: {agent_script}")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    start_time = time.time()

    command = f"python {agent_script} {args_str}"
    print(f"Executing command: {command}")
    exit_code = os.system(command)

    duration = time.time() - start_time

    if exit_code == 0:
        print(f"\n✅ COMPLETED: {description}")
        print(f"⏱️ Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        return True
    else:
        print(f"\n❌ FAILED: {description}")
        print(f"💥 Exit code: {exit_code}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run Multi-Agent OOD Detection Framework (Gemini Backend)"
    )

    parser.add_argument("--image_dir", required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", required=True, help="Directory for output files")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between inference calls (seconds)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("🌟 MULTI-AGENT OOD DETECTION FRAMEWORK (LLaVA-7B LOCAL)")
    print("=" * 80)
    print(f"📂 Input Directory: {args.image_dir}")
    print(f"📁 Output Directory: {args.output_dir}")
    print(f"⏱️ Request Delay: {args.delay} seconds")
    print(f"🔗 Backend: HuggingFace LLaVA-1.5-7B (local)")
    print(f"🕐 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    agent_outputs = {
        "agent1": "agent1_scene_context_results.json",
        "agent2": "agent2_spatial_anomaly_results.json",
        "agent3": "agent3_semantic_inconsistency_results.json",
        "agent4": "agent4_visual_appearance_results.json",
        "agent5": "agent5_final_synthesis_results.json"
    }

    agents = [
        {"script": "src/agents/agent1.py", "description": "Agent 1: Scene Context Analyzer"},
        {"script": "src/agents/agent2.py", "description": "Agent 2: Spatial Anomaly Detector"},
        {"script": "src/agents/agent3.py", "description": "Agent 3: Semantic Inconsistency Analyzer"},
        {"script": "src/agents/agent4.py", "description": "Agent 4: Visual Appearance Evaluator"},
        {"script": "src/agents/agent5.py", "description": "Agent 5: Reasoning Synthesizer"}
    ]

    successful_agents = []
    failed_agents = []

    # Run Agents 1–4
    for i, agent in enumerate(agents[:4]):
        print(f"\n🔄 STEP {i+1}/5: {agent['description']}")

        output_file = os.path.join(args.output_dir, agent_outputs[f"agent{i+1}"])
        agent_args = (
            f"--image_directory {args.image_dir} "
            f"--output_file {output_file} "
            f"--delay_between_requests {args.delay}"
        )

        success = run_agent(agent["script"], agent["description"], agent_args)

        if success:
            successful_agents.append(agent["description"])
        else:
            failed_agents.append(agent["description"])
            print("🛑 Stopping execution due to failure.")
            sys.exit(1)

        print("\n⏳ Waiting 10 seconds before next agent...")
        time.sleep(10)

    # Run Agent 5
    print(f"\n🔄 STEP 5/5: {agents[4]['description']}")

    agent5_args = (
        f"--agent1_file {os.path.join(args.output_dir, agent_outputs['agent1'])} "
        f"--agent2_file {os.path.join(args.output_dir, agent_outputs['agent2'])} "
        f"--agent3_file {os.path.join(args.output_dir, agent_outputs['agent3'])} "
        f"--agent4_file {os.path.join(args.output_dir, agent_outputs['agent4'])} "
        f"--output_file {os.path.join(args.output_dir, agent_outputs['agent5'])} "
        f"--delay_between_requests {args.delay}"
    )

    success = run_agent(agents[4]["script"], agents[4]["description"], agent5_args)

    if success:
        successful_agents.append(agents[4]["description"])
    else:
        failed_agents.append(agents[4]["description"])

    print(f"\n{'='*80}")
    print("🎊 EXECUTION SUMMARY")
    print(f"✅ Successful Agents: {len(successful_agents)}/5")
    print(f"❌ Failed Agents: {len(failed_agents)}/5")

    final_output = os.path.join(args.output_dir, agent_outputs["agent5"])
    if os.path.exists(final_output):
        print(f"\n🎯 FINAL OUTPUT READY:")
        print(f"📄 {final_output}")
        print("🚀 Ready for GroundedSAM evaluation!")
    else:
        print("\n⚠️ Final synthesis file not found!")

    print(f"\n🏁 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
