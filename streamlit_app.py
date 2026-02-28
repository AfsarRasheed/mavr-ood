#!/usr/bin/env python3
"""
MAVR-OOD Streamlit Frontend
Multi-Agent Vision-Language Reasoning for OOD Object Localization

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
import json
import numpy as np
import torch
import cv2
from PIL import Image
import streamlit as st

# ============================================================
# CRITICAL: Monkey-patch transformers BEFORE importing GroundingDINO
# ============================================================
import transformers
_orig_fn = getattr(transformers.PreTrainedModel, 'get_extended_attention_mask', None)
if _orig_fn is not None:
    def _safe_get_extended_attention_mask(self, attention_mask, input_shape, device_or_dtype=None):
        if attention_mask.dim() == 3:
            extended = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong attention_mask shape: {attention_mask.shape}")
        extended = extended.to(dtype=torch.float32)
        extended = (1.0 - extended) * torch.finfo(torch.float32).min
        return extended
    transformers.PreTrainedModel.get_extended_attention_mask = _safe_get_extended_attention_mask

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "GroundingDINO"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "segment_anything"))

# Reuse backend functions from app.py
from app import (
    load_gdino_model,
    load_sam_predictor,
    load_clip_verifier,
    preprocess_image,
    get_grounding_output,
    run_sam_segmentation,
    create_detection_visualization,
    create_mask_visualization,
    create_binary_mask_visualization,
    run_agents_on_image,
    extract_prompts,
)

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
# Custom CSS
# =====================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-size: 1.6rem;
        margin-bottom: 0.2rem;
        color: #1a1a2e;
    }
    .main-header p {
        font-size: 0.95rem;
        color: #555;
        margin: 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1a73e8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
    }
    .agent-box {
        background: #f0f4f8;
        border-left: 4px solid #1a73e8;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0 8px 8px 0;
    }
    .agent-error {
        border-left-color: #ea4335;
        background: #fef0ef;
    }
    .pipeline-step {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 6px;
        font-size: 0.9rem;
    }
    .step-complete {
        background: #e6f4ea;
        color: #137333;
    }
    .step-running {
        background: #e8f0fe;
        color: #1a73e8;
    }
    .step-pending {
        background: #f1f3f4;
        color: #999;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        color: #888;
        font-size: 0.85rem;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# =====================
# Header
# =====================
st.markdown("""
<div class="main-header">
    <h1>MAVR-OOD</h1>
    <p>Multi-Agent Vision-Language Reasoning for Reliable Out-of-Distribution Object Localization in Road Environments</p>
    <p style="font-size: 0.8rem; color: #888; margin-top: 0.5rem;">
        Pipeline: Multi-Agent Analysis (LLaVA-7B) → Object Grounding (GroundingDINO) → Semantic Verification (CLIP) → Segmentation (SAM)
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
        min_value=0.05, max_value=0.50, value=0.20, step=0.05,
        help="Lower = more detections, Higher = more precise"
    )
    box_threshold = st.slider(
        "Box Threshold",
        min_value=0.10, max_value=0.50, value=0.30, step=0.05,
        help="GroundingDINO confidence threshold"
    )

    st.divider()
    st.markdown("### Pipeline Components")
    st.markdown("""
    - **Agent 1**: Scene Context
    - **Agent 2**: Spatial Anomaly
    - **Agent 3**: Semantic Analysis
    - **Agent 4**: Visual Appearance
    - **Agent 5**: Reasoning Synthesis
    - **GroundingDINO**: Object Detection
    - **CLIP**: Semantic Verification
    - **SAM**: Segmentation
    """)

    st.divider()
    st.caption("MAVR-OOD | LLaVA-7B · GroundingDINO · CLIP · SAM")


# =====================
# Tabs
# =====================
tab1, tab2 = st.tabs(["📷 Single Image", "📁 Batch Evaluation"])


# =====================
# Tab 1: Single Image
# =====================
with tab1:
    col_upload, col_settings = st.columns([3, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload a road scene image",
            type=["jpg", "jpeg", "png"],
            help="Upload an image to detect out-of-distribution objects"
        )

    with col_settings:
        gt_mask_file = st.file_uploader(
            "Ground truth mask (optional)",
            type=["png"],
            help="Upload ground truth for metric computation"
        )

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image_pil)

        st.image(image_pil, caption="Uploaded Image", use_container_width=True)

        run_button = st.button("🚀 Run Detection", type="primary", use_container_width=True)

        if run_button:
            # Save temp image
            os.makedirs("outputs", exist_ok=True)
            tmp_path = os.path.join("outputs", "temp_upload.jpg")
            image_pil.save(tmp_path)

            # === Stage 1: Agents ===
            status = st.status("Running multi-agent analysis...", expanded=True)

            with status:
                st.write("🤖 Running Agent 1: Scene Context Analyzer...")
                agent_results = run_agents_on_image(tmp_path)
                st.write("✅ All 5 agents completed")

                prompt_v1, prompt_v2 = extract_prompts(agent_results)
                st.write(f"📝 Prompt V1: **{prompt_v1}**")
                st.write(f"📝 Prompt V2: **{prompt_v2}**")

            # === Stage 2: Detection ===
            status2 = st.status("Running detection pipeline...", expanded=True)

            with status2:
                st.write("📦 Loading GroundingDINO...")
                gdino = load_gdino_model()
                predictor = load_sam_predictor()
                clip_verifier = load_clip_verifier()

                st.write(f"🎯 Detecting with prompt: '{prompt_v1}'...")
                image_tensor = preprocess_image(image_pil)

                boxes, labels, scores = get_grounding_output(
                    gdino, image_tensor, prompt_v1,
                    box_threshold=box_threshold, text_threshold=0.25
                )

                if len(boxes) == 0 and prompt_v2 != prompt_v1:
                    st.write(f"🔄 Trying prompt V2: '{prompt_v2}'...")
                    boxes, labels, scores = get_grounding_output(
                        gdino, image_tensor, prompt_v2,
                        box_threshold=box_threshold, text_threshold=0.25
                    )

                # CLIP verification
                if len(boxes) > 0:
                    st.write("🔍 CLIP semantic verification...")
                    try:
                        H, W = image_np.shape[:2]
                        clip_boxes = boxes.clone()
                        clip_boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * W
                        clip_boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * H
                        clip_boxes[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * W
                        clip_boxes[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * H

                        clip_verifier.similarity_threshold = clip_threshold
                        filtered_boxes, filtered_phrases, clip_scores, _ = clip_verifier.verify_detections(
                            image_np, clip_boxes, labels, prompt_v1
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
                    st.write("🎨 Running SAM segmentation...")
                    masks, boxes_xyxy = run_sam_segmentation(predictor, image_np, boxes)
                    st.write(f"✅ Found {len(boxes)} detections")
                else:
                    st.write("⚠️ No detections found")

                status2.update(label="Detection complete!", state="complete")

            # === Results ===
            st.divider()
            st.subheader("Detection Results")

            # Metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Detections", len(boxes) if boxes is not None else 0)
            with m2:
                st.metric("Prompt V1", prompt_v1)
            with m3:
                st.metric("Prompt V2", prompt_v2)

            # Visualization columns
            if masks is not None and len(masks) > 0:
                det_img = create_detection_visualization(image_np, boxes_xyxy, labels)
                mask_img = create_mask_visualization(image_np, masks)
                binary_img = create_binary_mask_visualization(image_np, masks)

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
                    st.subheader("📊 Evaluation Metrics")

                    gt_mask = np.array(Image.open(gt_mask_file).convert("L"))
                    gt_binary = (gt_mask > 0).astype(np.float32)

                    # Create predicted binary mask
                    pred_binary = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)
                    for mask in masks:
                        m = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
                        if m.ndim == 3:
                            m = m.squeeze(0)
                        if m.shape != pred_binary.shape:
                            m = cv2.resize(m.astype(np.float32), (pred_binary.shape[1], pred_binary.shape[0]))
                        pred_binary = np.maximum(pred_binary, m)

                    pred_binary = (pred_binary > 0.5).astype(np.float32)

                    # Resize gt to match if needed
                    if gt_binary.shape != pred_binary.shape:
                        gt_binary = cv2.resize(gt_binary, (pred_binary.shape[1], pred_binary.shape[0]))

                    # Compute metrics
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

                    # Side-by-side mask comparison
                    gc1, gc2 = st.columns(2)
                    with gc1:
                        st.image(gt_binary, caption="Ground Truth Mask", use_container_width=True, clamp=True)
                    with gc2:
                        st.image(pred_binary, caption="Predicted Mask", use_container_width=True, clamp=True)

            else:
                st.warning("No objects detected in the image.")

            # Agent Analysis
            st.divider()
            st.subheader("🤖 Agent Analysis")

            for i, (name, key) in enumerate([
                ("Agent 1 — Scene Context", "agent1"),
                ("Agent 2 — Spatial Anomaly", "agent2"),
                ("Agent 3 — Semantic Analysis", "agent3"),
                ("Agent 4 — Visual Appearance", "agent4"),
                ("Agent 5 — Reasoning Synthesis", "agent5"),
            ]):
                data = agent_results.get(key, {})
                with st.expander(name, expanded=(i == 4)):
                    if isinstance(data, dict):
                        if "error" in data:
                            st.error(f"Error: {data['error']}")
                        else:
                            st.json(data)
                    else:
                        st.text(str(data)[:500])


# =====================
# Tab 2: Batch Evaluation
# =====================
with tab2:
    st.markdown("Run the full evaluation pipeline on a dataset folder.")

    dataset_dir = st.text_input(
        "Dataset Directory",
        value="./data/challenging_subset",
        help="Path to dataset with original/ and labels/ subfolders"
    )

    prompts_file = st.text_input(
        "Agent Prompts JSON",
        value="./outputs/challenging_subset_prompts/agent5_final_synthesis_results.json",
        help="Path to agent5 synthesis results"
    )

    if st.button("🚀 Run Batch Evaluation", type="primary"):
        st.info("For batch evaluation, use the command line:")
        st.code(f"""python run_evaluate.py \\
    --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \\
    --grounded_checkpoint weights/groundingdino_swint_ogc.pth \\
    --sam_checkpoint weights/sam_vit_h_4b8939.pth \\
    --dataset_dir {dataset_dir} \\
    --dataset_type road_anomaly \\
    --multiagent_prompts {prompts_file} \\
    --output_dir ./outputs/evaluation_results \\
    --clip_threshold {clip_threshold} \\
    --device cuda""")

    # Show existing results if available
    results_file = "./outputs/evaluation_results/multiagent_evaluation_results.json"
    if os.path.exists(results_file):
        st.divider()
        st.subheader("📊 Previous Evaluation Results")
        with open(results_file) as f:
            results = json.load(f)

        m = results.get("average_metrics", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("mIoU", f"{m.get('mIoU', 0):.4f}")
        with c2:
            st.metric("F1", f"{m.get('F1', 0):.4f}")
        with c3:
            st.metric("Precision", f"{m.get('Precision', 0):.4f}")
        with c4:
            st.metric("Recall", f"{m.get('Recall', 0):.4f}")

        st.metric("Detection Rate", f"{results.get('detection_rate', 0)}%")

        # Show per-image results
        if "per_image_results" in results:
            st.subheader("Per-Image Results")
            for img_result in results["per_image_results"]:
                name = img_result.get("image_name", "Unknown")
                status = img_result.get("status", "unknown")
                metrics = img_result.get("metrics", {})
                icon = "✅" if status != "failure_no_detection" else "❌"
                with st.expander(f"{icon} {name}"):
                    st.json(metrics)


# =====================
# Footer
# =====================
st.markdown("""
<div class="footer">
    <strong>MAVR-OOD</strong> | LLaVA-7B · GroundingDINO · CLIP · SAM
</div>
""", unsafe_allow_html=True)
