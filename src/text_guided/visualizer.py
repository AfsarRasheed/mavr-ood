"""
Step Visualization Generator
Creates annotated images for each step of the text-guided detection pipeline.
"""

import cv2
import numpy as np
import torch


def generate_step_visualizations(image_np, scene_result, parsed_query,
                                  all_boxes_xyxy, all_labels, all_scores,
                                  clip_mask, clip_scores,
                                  selected_idx, final_masks):
    """
    Generate visualization images for each pipeline step.

    Returns:
        dict of step_name -> numpy RGB image
    """
    H, W = image_np.shape[:2]
    steps = {}

    # ---- Step 1: Scene Understanding ----
    step1 = image_np.copy()
    scene_text = []
    if isinstance(scene_result, dict):
        scene_type = scene_result.get("scene_type", "Unknown scene")
        lighting = scene_result.get("lighting", "unknown")
        objects = scene_result.get("objects", [])
        scene_text.append(f"Scene: {scene_type}")
        scene_text.append(f"Lighting: {lighting}")
        scene_text.append(f"Objects found: {len(objects)}")
        for i, obj in enumerate(objects[:10]):
            name = obj.get("name", "?")
            pos = obj.get("position", "?")
            color = obj.get("color", "")
            scene_text.append(f"  {i+1}. {name} ({color}, {pos})")
    else:
        scene_text.append("Scene analysis unavailable")

    overlay = step1.copy()
    box_h = min(30 + len(scene_text) * 25, H - 20)
    cv2.rectangle(overlay, (10, 10), (min(450, W - 10), box_h), (0, 0, 0), -1)
    step1 = cv2.addWeighted(overlay, 0.7, step1, 0.3, 0)
    for i, text in enumerate(scene_text):
        cv2.putText(step1, text, (20, 35 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(step1, "STEP 1: Scene Understanding (LLaVA)", (10, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    steps["step1_scene"] = step1

    # ---- Step 2: Attribute Matching Agent ----
    step2 = image_np.copy()
    overlay2 = step2.copy()
    cv2.rectangle(overlay2, (10, 10), (min(500, W - 10), 220), (0, 0, 0), -1)
    step2 = cv2.addWeighted(overlay2, 0.7, step2, 0.3, 0)

    agent2_info = [
        f"Query: \"{parsed_query['original']}\"",
        f"Detection Prompt: \"{parsed_query['object_prompt']}\"",
        f"Attribute: {parsed_query.get('attribute', 'None')}",
        f"Spatial: {parsed_query.get('spatial', 'None (detect all)')}",
    ]
    attr_result = parsed_query.get('attr_agent_result', {})
    if isinstance(attr_result, dict):
        reasoning = attr_result.get('reasoning', '')
        ambiguity = attr_result.get('ambiguity', 'N/A')
        if reasoning:
            agent2_info.append(f"Reasoning: {reasoning[:80]}")
        agent2_info.append(f"Ambiguity: {ambiguity}")
        matched = attr_result.get('matched_objects', [])
        for m in matched[:3]:
            agent2_info.append(f"  Match: {m.get('name','')} ({m.get('position','')}) [{m.get('confidence','')}]")

    for i, text in enumerate(agent2_info):
        cv2.putText(step2, text, (20, 35 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)
    cv2.putText(step2, "STEP 2: Attribute Matching Agent (LLaVA)", (10, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    steps["step2_query"] = step2

    # ---- Step 3: All Candidates (GroundingDINO) ----
    step3 = image_np.copy()
    if all_boxes_xyxy is not None and len(all_boxes_xyxy) > 0:
        boxes_np = all_boxes_xyxy.numpy() if torch.is_tensor(all_boxes_xyxy) else np.array(all_boxes_xyxy)
        for i, (box, label) in enumerate(zip(boxes_np, all_labels)):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(step3, (x1, y1), (x2, y2), (0, 255, 255), 3)
            score_txt = f"#{i+1} {label}"
            cv2.putText(step3, score_txt, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        n_found = len(all_boxes_xyxy)
    else:
        n_found = 0
    cv2.putText(step3, f"STEP 3: GroundingDINO - {n_found} candidates found", (10, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    steps["step3_candidates"] = step3

    # ---- Step 4: CLIP Verification ----
    step4 = image_np.copy()
    if all_boxes_xyxy is not None and len(all_boxes_xyxy) > 0:
        boxes_np = all_boxes_xyxy.numpy() if torch.is_tensor(all_boxes_xyxy) else np.array(all_boxes_xyxy)
        n_passed = 0
        n_rejected = 0
        for i, box in enumerate(boxes_np):
            x1, y1, x2, y2 = box.astype(int)
            passed = clip_mask[i] if clip_mask is not None and i < len(clip_mask) else True
            if passed:
                color = (0, 255, 0)
                status = "PASS"
                score = clip_scores[i] if clip_scores and i < len(clip_scores) else 0
                n_passed += 1
            else:
                color = (0, 0, 255)
                status = "REJECT"
                score = clip_scores[i] if clip_scores and i < len(clip_scores) else 0
                n_rejected += 1
            cv2.rectangle(step4, (x1, y1), (x2, y2), color, 3)
            cv2.putText(step4, f"#{i+1} {status} ({score:.2f})", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        summary = f"STEP 4: CLIP Verification - {n_passed} passed, {n_rejected} rejected"
    else:
        summary = "STEP 4: CLIP Verification - No candidates"
    cv2.putText(step4, summary, (10, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    steps["step4_clip"] = step4

    # ---- Step 5: Spatial Selection ----
    step5 = image_np.copy()
    if all_boxes_xyxy is not None and len(all_boxes_xyxy) > 0 and selected_idx is not None:
        boxes_np = all_boxes_xyxy.numpy() if torch.is_tensor(all_boxes_xyxy) else np.array(all_boxes_xyxy)
        for i, box in enumerate(boxes_np):
            if clip_mask is not None and i < len(clip_mask) and not clip_mask[i]:
                continue
            x1, y1, x2, y2 = box.astype(int)
            if isinstance(selected_idx, list):
                is_selected = i in selected_idx
            else:
                is_selected = (i == selected_idx)

            if is_selected:
                cv2.rectangle(step5, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(step5, "SELECTED", (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(step5, (x1, y1), (x2, y2), (128, 128, 128), 2)
                cv2.putText(step5, "skipped", (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA)

    spatial_text = parsed_query.get('spatial', 'all')
    if spatial_text:
        cv2.putText(step5, f"STEP 5: Spatial Filter - '{spatial_text}'", (10, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(step5, "STEP 5: No spatial filter - keeping all", (10, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    steps["step5_spatial"] = step5

    # ---- Step 6: Final Segmentation ----
    step6 = image_np.copy().astype(np.float32)
    if final_masks is not None and len(final_masks) > 0:
        colors = [
            [0, 255, 100],
            [255, 0, 100],
            [255, 165, 0],
            [0, 255, 255],
        ]
        for i, mask in enumerate(final_masks):
            mask_np = mask.squeeze().numpy().astype(bool) if torch.is_tensor(mask) else mask.astype(bool)
            color = np.array(colors[i % len(colors)], dtype=np.float32)
            step6[mask_np] = step6[mask_np] * 0.35 + color * 0.65
    step6 = step6.astype(np.uint8)

    if all_boxes_xyxy is not None and selected_idx is not None:
        boxes_np = all_boxes_xyxy.numpy() if torch.is_tensor(all_boxes_xyxy) else np.array(all_boxes_xyxy)
        indices = selected_idx if isinstance(selected_idx, list) else [selected_idx]
        for idx in indices:
            if idx < len(boxes_np):
                x1, y1, x2, y2 = boxes_np[idx].astype(int)
                cv2.rectangle(step6, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.putText(step6, "STEP 6: SAM Segmentation - Final Result", (10, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    steps["step6_final"] = step6

    return steps
