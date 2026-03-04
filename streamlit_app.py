#!/usr/bin/env python3
"""
MAVR Streamlit Frontend
Multi-Agent Vision-Language Reasoning for Reliable Object Localization in Road Environments

Two tabs:
  1. Text-Guided Detection — user provides text prompt to detect specific objects
  2. OOD Detection — autonomous multi-agent anomaly detection

Run on Colab:
    !pip install streamlit pyngrok
    !streamlit run streamlit_app.py &
    from pyngrok import ngrok
    print(ngrok.connect(8501))

Run locally:
    streamlit run streamlit_app.py
"""

import os
import sys
import gc
import json
import tempfile
import numpy as np
import torch
import cv2
from PIL import Image
import streamlit as st

# ============================================================
# Lazy imports — models loaded only when needed (not at startup)
# This lets the UI render even without GPU/models installed
# ============================================================
def _load_backend():
    """Lazy-load backend functions from app.py and model_loader."""
    from src.model_loader import load_gdino_model, load_sam_predictor, load_clip_verifier
    from app import (
        preprocess_image, get_grounding_output, run_sam_segmentation,
        create_detection_visualization, create_mask_visualization,
        create_binary_mask_visualization, run_agents_on_image, extract_prompts,
        generate_pipeline_visualization_img,
    )
    return {
        "load_gdino_model": load_gdino_model,
        "load_sam_predictor": load_sam_predictor,
        "load_clip_verifier": load_clip_verifier,
        "preprocess_image": preprocess_image,
        "get_grounding_output": get_grounding_output,
        "run_sam_segmentation": run_sam_segmentation,
        "create_detection_visualization": create_detection_visualization,
        "create_mask_visualization": create_mask_visualization,
        "create_binary_mask_visualization": create_binary_mask_visualization,
        "run_agents_on_image": run_agents_on_image,
        "extract_prompts": extract_prompts,
        "generate_pipeline_visualization_img": generate_pipeline_visualization_img,
    }


# =====================
# Page Config
# =====================
st.set_page_config(
    page_title="MAVR-OOD",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================
# Custom CSS — Dark theme
# =====================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.2rem 0;
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #30363d;
    }
    .main-header h1 {
        font-size: 1.8rem;
        margin-bottom: 0.3rem;
        color: #58a6ff;
        letter-spacing: 2px;
    }
    .main-header p {
        font-size: 0.9rem;
        color: #8b949e;
        margin: 0;
    }
    .main-header .pipeline {
        font-size: 0.75rem;
        color: #6e7681;
        margin-top: 0.5rem;
    }
    .step-header {
        color: #58a6ff;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        color: #6e7681;
        font-size: 0.8rem;
        border-top: 1px solid #30363d;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# =====================
# Header
# =====================
st.markdown("""
<div class="main-header">
    <h1>🔍 MAVR</h1>
    <p>Multi-Agent Vision-Language Reasoning for Reliable Object Localization in Road Environments</p>
    <p class="pipeline">
        Pipeline: LLaVA-7B → GroundingDINO → CLIP Verification → SAM Segmentation
    </p>
</div>
""", unsafe_allow_html=True)


# =====================
# Sidebar
# =====================
with st.sidebar:
    st.header("⚙️ Settings")

    clip_threshold = st.slider(
        "CLIP Threshold",
        min_value=0.10, max_value=0.50, value=0.25, step=0.05,
        help="Lower = more detections, Higher = stricter filtering"
    )
    box_threshold = st.slider(
        "Box Threshold (GroundingDINO)",
        min_value=0.15, max_value=0.50, value=0.35, step=0.05,
        help="Confidence threshold for GroundingDINO candidates"
    )

    st.divider()
    st.markdown("### 🧠 Pipeline Agents")
    st.markdown("""
    **Text-Guided Mode:**
    - Scene Understanding Agent (LLaVA)
    - Attribute Matching Agent (LLaVA)
    - Object Grounding (GroundingDINO)
    - Semantic Verification (CLIP)
    - Spatial Filter
    - Segmentation (SAM)

    **OOD Mode:**
    - Agent 1: Scene Context
    - Agent 2: Spatial Anomaly
    - Agent 3: Semantic Analysis
    - Agent 4: Visual Appearance
    - Agent 5: Reasoning Synthesis
    - GroundingDINO + CLIP + SAM
    """)

    st.divider()
    st.caption("MAVR-OOD | LLaVA-7B · GroundingDINO · CLIP · SAM")


# =====================
# Tabs
# =====================
tab1, tab2 = st.tabs(["🎯 Text-Guided Detection", "🔬 OOD Detection"])


# =====================
# Tab 1: Text-Guided Detection
# =====================
with tab1:
    st.markdown("Upload any image and describe the object to find. Supports spatial terms like **left/right/center/largest**.")

    col_img, col_prompt = st.columns([2, 1])

    with col_img:
        tg_uploaded = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"],
            help="Upload a road scene or any image",
            key="tg_upload"
        )

    with col_prompt:
        user_prompt = st.text_input(
            "Text Prompt",
            placeholder="e.g. the white car on the left",
            help="Describe the object with color, type, and spatial terms"
        )
        st.markdown("""
        **Example prompts:**
        - `the yellow car in the middle`
        - `the largest zebra`
        - `the red truck on the right`
        """)

    if tg_uploaded is not None:
        tg_image_pil = Image.open(tg_uploaded).convert("RGB")
        tg_image_np = np.array(tg_image_pil)

        st.image(tg_image_pil, caption="Uploaded Image", use_container_width=True)

        tg_run = st.button("🔍 Detect Object", type="primary", use_container_width=True, key="tg_run")

        if tg_run:
            if not user_prompt or not user_prompt.strip():
                st.error("Please enter a text prompt first.")
            else:
                from src.text_guided import run_text_guided_pipeline

                # Save temp image for LLaVA
                temp_path = os.path.join(tempfile.gettempdir(), "tg_input.jpg")
                tg_image_pil.save(temp_path)

                # Phase 1: LLaVA agents
                status1 = st.status("Phase 1: Running LLaVA agents...", expanded=True)
                with status1:
                    from src.text_guided.scene_agent import scene_understanding
                    from src.text_guided.attribute_agent import attribute_matching_agent

                    st.write("🔄 Running Scene Understanding Agent...")
                    scene_result = scene_understanding(temp_path)
                    st.write("✅ Scene analysis complete")

                    st.write("🔄 Running Attribute Matching Agent...")
                    attr_result = attribute_matching_agent(temp_path, scene_result, user_prompt)
                    st.write("✅ Attribute matching complete")

                    status1.update(label="Phase 1: LLaVA agents complete!", state="complete")

                # Free LLaVA from GPU
                try:
                    import src.agents.vlm_backend as vlm_mod
                    if hasattr(vlm_mod, '_model') and vlm_mod._model is not None:
                        del vlm_mod._model
                        vlm_mod._model = None
                    if hasattr(vlm_mod, '_processor') and vlm_mod._processor is not None:
                        del vlm_mod._processor
                        vlm_mod._processor = None
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception:
                    gc.collect()
                    torch.cuda.empty_cache()

                # Phase 2: Detection pipeline
                status2 = st.status("Phase 2: Running detection pipeline...", expanded=True)
                with status2:
                    st.write("🔄 Loading detection models...")
                    backend = _load_backend()
                    gdino = backend["load_gdino_model"]()
                    sam = backend["load_sam_predictor"]()
                    clip_v = backend["load_clip_verifier"]()

                    st.write(f"🔍 Running pipeline: '{user_prompt}'...")
                    results = run_text_guided_pipeline(
                        image_np=tg_image_np,
                        user_prompt=user_prompt,
                        image_path=temp_path,
                        gdino_model=gdino,
                        sam_predictor=sam,
                        clip_verifier=clip_v,
                        box_threshold=box_threshold,
                        clip_threshold=clip_threshold,
                        precomputed_scene=scene_result,
                        precomputed_attr=attr_result,
                    )
                    st.write("✅ Detection complete!")
                    status2.update(label="Phase 2: Detection complete!", state="complete")

                # === Show Results ===
                st.divider()
                st.subheader("📊 Step-by-Step Pipeline Results")

                step_images = results.get("step_images", {})
                step_names = [
                    ("step1_scene", "Step 1: Scene Understanding (LLaVA)"),
                    ("step2_query", "Step 2: Attribute Matching (LLaVA)"),
                    ("step3_candidates", "Step 3: Candidates (GroundingDINO)"),
                    ("step4_clip", "Step 4: CLIP Verification"),
                    ("step5_spatial", "Step 5: Spatial Selection"),
                    ("step6_final", "Step 6: Final Segmentation (SAM)"),
                ]

                # Row 1: Steps 1-3
                c1, c2, c3 = st.columns(3)
                for col, (key, title) in zip([c1, c2, c3], step_names[:3]):
                    with col:
                        img = step_images.get(key)
                        if img is not None:
                            st.image(img, caption=title, use_container_width=True)

                # Row 2: Steps 4-6
                c4, c5, c6 = st.columns(3)
                for col, (key, title) in zip([c4, c5, c6], step_names[3:]):
                    with col:
                        img = step_images.get(key)
                        if img is not None:
                            st.image(img, caption=title, use_container_width=True)

                # Pipeline Summary
                st.divider()
                with st.expander("📋 Pipeline Summary", expanded=True):
                    # Build clean summary from results
                    summary_parts = []

                    # Query info
                    summary_parts.append(f"**Query:** \"{user_prompt}\"")

                    # Step 1: Scene
                    scene = results.get("scene_result", {})
                    if isinstance(scene, dict) and scene.get("scene_type"):
                        n_objs = len(scene.get("objects", []))
                        summary_parts.append(
                            f"**Step 1 — Scene Understanding:** {scene.get('scene_type', 'N/A')} scene, "
                            f"{scene.get('lighting', 'N/A')} lighting, {n_objs} objects identified"
                        )
                    else:
                        summary_parts.append("**Step 1 — Scene Understanding:** Scene analyzed")

                    # Step 2: Attribute
                    attr = results.get("attr_result", {})
                    if isinstance(attr, dict) and attr.get("reasoning"):
                        summary_parts.append(
                            f"**Step 2 — Attribute Matching:** {attr.get('reasoning', 'N/A')} "
                            f"(Ambiguity: {attr.get('ambiguity', 'N/A')})"
                        )
                    else:
                        summary_parts.append("**Step 2 — Attribute Matching:** Prompt refined")

                    # Step 3-4: Detection + CLIP
                    n_det = results.get("n_detected", 0)
                    n_ver = results.get("n_verified", 0)
                    summary_parts.append(f"**Step 3 — GroundingDINO:** {n_det} candidate(s) detected")
                    summary_parts.append(f"**Step 4 — CLIP Verification:** {n_ver}/{n_det} passed semantic check")

                    # Step 5: Spatial
                    spatial = results.get("parsed", {}).get("spatial", None)
                    n_sel = results.get("n_selected", 0)
                    anchor = results.get("parsed", {}).get("anchor", None)
                    if spatial and anchor:
                        summary_parts.append(f"**Step 5 — Spatial Filter:** Selected object '{spatial}' the '{anchor}' → {n_sel} selected")
                    elif spatial:
                        summary_parts.append(f"**Step 5 — Spatial Filter:** '{spatial}' → {n_sel} selected")
                    else:
                        summary_parts.append(f"**Step 5 — Spatial Filter:** No filter (keeping all {n_sel})")

                    # Step 6: SAM
                    summary_parts.append(f"**Step 6 — SAM Segmentation:** ✅ Complete")

                    # Render
                    for part in summary_parts:
                        st.markdown(part)


# =====================
# Tab 2: OOD Detection
# =====================
with tab2:
    st.markdown("Upload a road scene to autonomously detect out-of-distribution (anomalous) objects.")

    col_upload, col_gt = st.columns([3, 1])

    with col_upload:
        ood_uploaded = st.file_uploader(
            "Upload Road Scene",
            type=["jpg", "jpeg", "png"],
            help="Upload an image to detect out-of-distribution objects",
            key="ood_upload"
        )

    with col_gt:
        gt_mask_file = st.file_uploader(
            "Ground Truth Mask (optional)",
            type=["png"],
            help="Upload ground truth for metric computation",
            key="ood_gt"
        )

    if ood_uploaded is not None:
        ood_image_pil = Image.open(ood_uploaded).convert("RGB")
        ood_image_np = np.array(ood_image_pil)

        st.image(ood_image_pil, caption="Uploaded Image", use_container_width=True)

        ood_run = st.button("🔬 Run OOD Detection", type="primary", use_container_width=True, key="ood_run")

        if ood_run:
            # Save temp image
            os.makedirs("outputs", exist_ok=True)
            tmp_path = os.path.join("outputs", "temp_upload.jpg")
            ood_image_pil.save(tmp_path)

            # === Stage 1: Agents ===
            status = st.status("Running multi-agent analysis...", expanded=True)

            with status:
                st.write("🔄 Running 5 LLaVA agents...")
                backend = _load_backend()
                agent_results = backend["run_agents_on_image"](tmp_path)
                st.write("✅ All 5 agents completed")

                prompt_v1, prompt_v2 = backend["extract_prompts"](agent_results)
                st.write(f"Prompt V1: **{prompt_v1}**")
                st.write(f"Prompt V2: **{prompt_v2}**")

                status.update(label="Agent analysis complete!", state="complete")

            # Free LLaVA
            try:
                import src.agents.vlm_backend as vlm_mod
                if hasattr(vlm_mod, '_model') and vlm_mod._model is not None:
                    del vlm_mod._model
                    vlm_mod._model = None
                if hasattr(vlm_mod, '_processor') and vlm_mod._processor is not None:
                    del vlm_mod._processor
                    vlm_mod._processor = None
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                gc.collect()
                torch.cuda.empty_cache()

            # === Stage 2: Detection ===
            status2 = st.status("Running detection pipeline...", expanded=True)

            with status2:
                st.write("🔄 Loading detection models...")
                gdino = backend["load_gdino_model"]()
                predictor = backend["load_sam_predictor"]()
                clip_verifier = backend["load_clip_verifier"]()

                st.write(f"🔍 Detecting with prompt: '{prompt_v1}'...")
                image_tensor = backend["preprocess_image"](ood_image_pil)

                boxes, labels, scores = backend["get_grounding_output"](
                    gdino, image_tensor, prompt_v1,
                    box_threshold=box_threshold, text_threshold=0.25
                )

                if len(boxes) == 0 and prompt_v2 != prompt_v1:
                    st.write(f"🔄 Trying prompt V2: '{prompt_v2}'...")
                    boxes, labels, scores = backend["get_grounding_output"](
                        gdino, image_tensor, prompt_v2,
                        box_threshold=box_threshold, text_threshold=0.25
                    )

                # CLIP verification
                if len(boxes) > 0:
                    st.write("🔍 CLIP semantic verification...")
                    try:
                        H, W = ood_image_np.shape[:2]
                        clip_boxes = boxes.clone()
                        clip_boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * W
                        clip_boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * H
                        clip_boxes[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * W
                        clip_boxes[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * H

                        clip_verifier.similarity_threshold = clip_threshold
                        filtered_boxes, filtered_phrases, clip_scores, _ = clip_verifier.verify_detections(
                            ood_image_np, clip_boxes, labels, prompt_v1
                        )
                        if len(filtered_boxes) > 0:
                            boxes_back = torch.zeros(len(filtered_boxes), 4)
                            boxes_back[:, 0] = ((filtered_boxes[:, 0] + filtered_boxes[:, 2]) / 2) / W
                            boxes_back[:, 1] = ((filtered_boxes[:, 1] + filtered_boxes[:, 3]) / 2) / H
                            boxes_back[:, 2] = (filtered_boxes[:, 2] - filtered_boxes[:, 0]) / W
                            boxes_back[:, 3] = (filtered_boxes[:, 3] - filtered_boxes[:, 1]) / H
                            boxes = boxes_back
                            labels = filtered_phrases
                    except Exception as e:
                        st.write(f"⚠️ CLIP warning: {e}")

                # SAM segmentation
                masks = None
                boxes_xyxy = None
                if len(boxes) > 0:
                    st.write("🔄 Running SAM segmentation...")
                    masks, boxes_xyxy = backend["run_sam_segmentation"](predictor, ood_image_np, boxes)
                    st.write(f"✅ Found {len(boxes)} detections")
                else:
                    st.write("⚠️ No detections found")

                status2.update(label="Detection complete!", state="complete")

            # === Results ===
            st.divider()
            st.subheader("📊 Detection Results")

            # Metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Detections", len(boxes) if boxes is not None else 0)
            with m2:
                st.metric("Prompt V1", prompt_v1)
            with m3:
                st.metric("Prompt V2", prompt_v2)

            # Visualizations
            if masks is not None and len(masks) > 0:
                # Pipeline visualization (3-panel: Reasoning + Boxes + Masks)
                anomaly_reasoning = agent_results.get("agent5", {}).get("reasoning", "N/A")
                clip_scores_list = []
                try:
                    clip_scores_list = [0.0] * len(boxes_xyxy)
                except Exception:
                    pass

                pipeline_img = backend["generate_pipeline_visualization_img"](
                    ood_image_np, anomaly_reasoning, prompt_v1, prompt_v2,
                    boxes_xyxy, clip_scores_list, masks
                )
                st.image(pipeline_img, caption="Pipeline Visualization", use_container_width=True)

                st.divider()

                # Individual visualizations
                det_img = backend["create_detection_visualization"](ood_image_np, boxes_xyxy, labels)
                mask_img = backend["create_mask_visualization"](ood_image_np, masks)
                binary_img = backend["create_binary_mask_visualization"](ood_image_np, masks)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(det_img, caption="Bounding Boxes", use_container_width=True)
                with c2:
                    st.image(mask_img, caption="SAM Masks", use_container_width=True)
                with c3:
                    st.image(binary_img, caption="OOD Mask", use_container_width=True)

                # Ground truth comparison
                if gt_mask_file is not None:
                    st.divider()
                    st.subheader("📐 Evaluation Metrics")

                    gt_mask = np.array(Image.open(gt_mask_file).convert("L"))
                    gt_binary = (gt_mask > 0).astype(np.float32)

                    pred_binary = np.zeros((ood_image_np.shape[0], ood_image_np.shape[1]), dtype=np.float32)
                    for mask in masks:
                        m = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
                        if m.ndim == 3:
                            m = m.squeeze(0)
                        if m.shape != pred_binary.shape:
                            m = cv2.resize(m.astype(np.float32), (pred_binary.shape[1], pred_binary.shape[0]))
                        pred_binary = np.maximum(pred_binary, m)

                    pred_binary = (pred_binary > 0.5).astype(np.float32)

                    if gt_binary.shape != pred_binary.shape:
                        gt_binary = cv2.resize(gt_binary, (pred_binary.shape[1], pred_binary.shape[0]))

                    intersection = (pred_binary * gt_binary).sum()
                    union = ((pred_binary + gt_binary) > 0).sum()
                    iou = intersection / (union + 1e-8)
                    tp = intersection
                    fp = (pred_binary * (1 - gt_binary)).sum()
                    fn = ((1 - pred_binary) * gt_binary).sum()
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)

                    mc1, mc2, mc3, mc4 = st.columns(4)
                    with mc1:
                        st.metric("mIoU", f"{iou:.4f}")
                    with mc2:
                        st.metric("F1 Score", f"{f1:.4f}")
                    with mc3:
                        st.metric("Precision", f"{precision:.4f}")
                    with mc4:
                        st.metric("Recall", f"{recall:.4f}")

                    gc1, gc2 = st.columns(2)
                    with gc1:
                        st.image(gt_binary, caption="Ground Truth Mask", use_container_width=True, clamp=True)
                    with gc2:
                        st.image(pred_binary, caption="Predicted Mask", use_container_width=True, clamp=True)

            else:
                st.warning("No objects detected in the image.")

            # Agent Analysis
            st.divider()
            with st.expander("🧠 Agent Analysis", expanded=False):
                for i, (name, key) in enumerate([
                    ("Agent 1 — Scene Context", "agent1"),
                    ("Agent 2 — Spatial Anomaly", "agent2"),
                    ("Agent 3 — Semantic Analysis", "agent3"),
                    ("Agent 4 — Visual Appearance", "agent4"),
                    ("Agent 5 — Reasoning Synthesis", "agent5"),
                ]):
                    data = agent_results.get(key, {})
                    st.markdown(f"**{name}**")
                    if isinstance(data, dict):
                        if "error" in data:
                            st.error(f"Error: {data['error']}")
                        else:
                            st.json(data)
                    else:
                        st.text(str(data)[:500])


# =====================
# Footer
# =====================
st.markdown("""
<div class="footer">
    <strong>MAVR</strong> | LLaVA-7B · GroundingDINO · CLIP · SAM<br>
    Multi-Agent Vision-Language Reasoning for Reliable Object Localization in Road Environments
</div>
""", unsafe_allow_html=True)
