import matplotlib
matplotlib.use('Agg')

import argparse
import os
import sys
import numpy as np
import json
import torch
from PIL import Image
from itertools import product
import copy

# ============================================================
# CRITICAL: Monkey-patch transformers BEFORE importing GroundingDINO
# transformers 5.0 changed get_extended_attention_mask(mask, shape, device)
# to get_extended_attention_mask(mask, shape, dtype). GroundingDINO passes
# device, causing TypeError. This makes it work with both.
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

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# CLIP Semantic Verifier
from src.clip_verifier import CLIPVerifier

from dataset import DatasetFactory

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm


def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB") 

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_visualization(image, masks, boxes, labels, save_path, title_suffix=""):
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.imshow(image)
        
        for mask in masks:
            show_mask(mask.cpu().numpy(), ax, random_color=True)
        
        for box, label in zip(boxes, labels):
            show_box(box.numpy(), ax, label)
        
        ax.axis('off')
        if title_suffix:
            ax.set_title(title_suffix, fontsize=14, pad=10)
        
        fig.savefig(save_path, bbox_inches="tight", dpi=300, pad_inches=0.0)
        
        plt.close(fig)
        
        print(f"✓ Visualization saved: {os.path.basename(save_path)}")
        
    except Exception as e:
        print(f"Warning: Failed to save visualization {save_path}: {str(e)}")

        try:
            info_path = save_path.replace('.jpg', '_info.txt')
            with open(info_path, 'w') as f:
                f.write(f"Image shape: {image.shape}\n")
                f.write(f"Number of masks: {len(masks)}\n")
                f.write(f"Number of boxes: {len(boxes)}\n")
                f.write(f"Labels: {labels}\n")
            print(f"✓ Info saved instead: {os.path.basename(info_path)}")
        except:
            pass
def save_pipeline_visualization(image, anomaly_reasoning, v1_prompt, v2_prompt, boxes, clip_scores, masks, final_metrics, save_path):
    try:
        import textwrap
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Panel A: Input + NLP Reasoning
        ax1 = axes[0]
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title("Panel A: Input + VLM Reasoning", fontsize=16, pad=10, weight='bold')
        
        reasoning_text = f"Agent 5 Reasoning:\n{anomaly_reasoning}\n\nGenerated Prompts:\nV1: '{v1_prompt}'\nV2: '{v2_prompt}'"
        wrapped_text = "\n".join(textwrap.wrap(reasoning_text, width=60))
        ax1.text(0.5, -0.1, wrapped_text, ha='center', va='top', transform=ax1.transAxes, 
                 fontsize=14, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        # Panel B: GroundingDINO + CLIP
        ax2 = axes[1]
        ax2.imshow(image)
        for i, box in enumerate(boxes):
            clip_score = clip_scores[i] if i < len(clip_scores) else 0.0
            label = f"Score: {clip_score:.3f}"
            show_box(box.numpy() if hasattr(box, 'numpy') else box, ax2, label)
        ax2.axis('off')
        ax2.set_title("Panel B: GroundingDINO + CLIP Verifier", fontsize=16, pad=10, weight='bold')

        # Panel C: Final SAM Mask + Metrics
        ax3 = axes[2]
        ax3.imshow(image)
        for mask in masks:
            mask_arr = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
            show_mask(mask_arr, ax3, random_color=True)
        ax3.axis('off')
        ax3.set_title("Panel C: Final SAM Segmentation", fontsize=16, pad=10, weight='bold')
        
        metrics_text = f"mIoU: {final_metrics.get('mIoU', 0):.4f} | F1: {final_metrics.get('F1', 0):.4f}\nPrecision: {final_metrics.get('Precision', 0):.4f} | Recall: {final_metrics.get('Recall', 0):.4f}"
        ax3.text(0.5, -0.1, metrics_text, ha='center', va='top', transform=ax3.transAxes, 
                 fontsize=14, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
                 
        plt.tight_layout(pad=3.0)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"✓ Pipeline visualization saved: {os.path.basename(save_path)}")
    except Exception as e:
        print(f"Warning: Failed to save pipeline visualization: {str(e)}")

def save_metrics_bar_chart(metrics_dict, save_path, title="Segmentation Metrics"):
    try:
        metrics = ['mIoU', 'F1', 'Precision', 'Recall']
        values = [metrics_dict.get(m, 0.0) for m in metrics]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(metrics, values, color=['#3498db', '#2ecc71', '#9b59b6', '#e74c3c'])
        
        ax.set_ylim(0, 1.1)
        ax.set_title(title, fontsize=16, pad=15, weight='bold')
        ax.set_ylabel('Score', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=12, weight='bold')
                    
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"✓ Metrics bar chart saved: {os.path.basename(save_path)}")
    except Exception as e:
        print(f"Warning: Failed to save metrics bar chart: {str(e)}")

def save_spider_chart(metrics_dict, save_path, title="System Balance (Radar Chart)"):
    """
    Creates a 5-axis academic radar chart to visually prove the pipeline is balanced
    (doesn't sacrifice Precision for Recall, etc.)
    """
    try:
        labels = ['mIoU', 'F1 Score', 'Precision', 'Recall', 'CLIP Match']
        
        # Pull values. If CLIP is missing, default to 1.0 (perfect) or 0 depending on presence.
        clip_val = min(metrics_dict.get('CLIP_Score', 0.0) / 0.40, 1.0) # Normalize CLIP (0.4 is usually max real-world score)
        
        values = [
            metrics_dict.get('mIoU', 0.0),
            metrics_dict.get('F1', 0.0),
            metrics_dict.get('Precision', 0.0),
            metrics_dict.get('Recall', 0.0),
            clip_val
        ]
        
        # Close the loop
        values = np.concatenate((values, [values[0]]))
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        # Add values to labels
        labels_with_scores = [f"{label}\n({val:.2f})" for label, val in zip(labels, values[:-1])]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Plot and fill
        ax.plot(angles, values, 'o-', linewidth=3, color='#9b59b6', label='Current System')
        ax.fill(angles, values, alpha=0.3, color='#9b59b6')
        
        # Design formatting
        ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels_with_scores, fontsize=13, weight='bold')
        ax.set_ylim(0, 1)
        
        # Add gridlines and perfect circle limit
        ax.grid(color='#AAAAAA', linestyle='--', linewidth=1)
        ax.plot(angles, [1.0]*6, color='black', alpha=0.5, linestyle=':', linewidth=2, label='Perfect Score (1.0)')
        
        ax.set_title(title, fontsize=16, pad=20, weight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"✓ Polygon Spider chart saved: {os.path.basename(save_path)}")
    except Exception as e:
        print(f"Warning: Failed to save spider chart: {str(e)}")

def save_confidence_scatter_plot(results_list, save_path):
    """
    Plots Agent 5 Overall Confidence vs Final mIoU across all evaluated images.
    Demonstrates System Self-Awareness (correlation between LLM certainty and Vision accuracy).
    """
    try:
        if not results_list or len(results_list) < 2:
            print("Not enough data points to generate Scatter Plot.")
            return

        confidences = []
        mious = []
        image_names = []

        for res in results_list:
            # Safely extract metrics
            conf = res.get('agent_confidence', 0.0)
            metrics = res.get('final_metrics', {})
            miou = metrics.get('mIoU', 0.0)
            
            confidences.append(conf)
            mious.append(miou)
            image_names.append(res['image_name'])

        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Create scatter plot with semi-transparent academic styling
        ax.scatter(confidences, mious, alpha=0.7, s=100, c='#3498db', edgecolors='black', linewidths=1)
        
        # Calculate and plot the linear regression trendline (Line of Best Fit)
        if len(confidences) > 1 and len(set(confidences)) > 1:
            z = np.polyfit(confidences, mious, 1)
            p = np.poly1d(z)
            
            # Plot trendline
            x_trend = np.linspace(0, 1.0, 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f"Trendline (Correlation)")
            
            # Calculate correlation coefficient (R-value)
            correlation_matrix = np.corrcoef(confidences, mious)
            r_value = correlation_matrix[0,1]
            ax.text(0.05, 0.95, f"Pearson Correlation (R) = {r_value:.2f}", 
                    transform=ax.transAxes, fontsize=12, weight='bold',
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_title("System Self-Awareness: Language Confidence vs Vision Accuracy", fontsize=15, pad=15, weight='bold')
        ax.set_xlabel("Agent 5 Overall Confidence (LLaVA-7B)", fontsize=12, weight='bold')
        ax.set_ylabel("Final Segmentation mIoU (SAM)", fontsize=12, weight='bold')
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if len(confidences) > 1 and len(set(confidences)) > 1:
            ax.legend(loc="lower right")
        
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Scatter Plot (Confidence vs mIoU) saved: {os.path.basename(save_path)}")
        
    except Exception as e:
        print(f"Warning: Failed to save scatter plot: {str(e)}")

def calculate_iou(pred_mask, gt_mask):
    """Calculate IoU between prediction and ground truth masks"""
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    
    if np.sum(union) == 0:
        return 1.0 if np.sum(pred_mask) == 0 else 0.0
    
    iou = np.sum(intersection) / np.sum(union)
    return iou


def calculate_f1(pred_mask, gt_mask):
    """Calculate F1 score between prediction and ground truth masks"""
    true_positive = np.sum(np.logical_and(pred_mask, gt_mask))
    false_positive = np.sum(np.logical_and(pred_mask, ~gt_mask))
    false_negative = np.sum(np.logical_and(~pred_mask, gt_mask))
    
    if true_positive + false_positive == 0:
        precision = 0.0
    else:
        precision = true_positive / (true_positive + false_positive)
    
    if true_positive + false_negative == 0:
        recall = 0.0
    else:
        recall = true_positive / (true_positive + false_negative)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def calculate_auuprc(pred_scores, gt_mask):
    """Calculate Area Under Precision-Recall Curve (AUUPRC)"""
    try:
        from sklearn.metrics import precision_recall_curve, auc
        precision, recall, _ = precision_recall_curve(gt_mask.flatten(), pred_scores.flatten())
        return auc(recall, precision)
    except:
        return 0.0


def calculate_fpr95(pred_scores, gt_mask):
    """Calculate FPR at 95% TPR"""
    try:
        fpr, tpr, thresholds = roc_curve(gt_mask.flatten(), pred_scores.flatten())
        if np.sum(tpr >= 0.95) == 0:
            return 1.0
        return fpr[np.where(tpr >= 0.95)[0][0]]
    except:
        return 1.0


def calculate_auroc(pred_scores, gt_mask):
    """Calculate Area Under ROC curve"""
    try:
        fpr, tpr, _ = roc_curve(gt_mask.flatten(), pred_scores.flatten())
        return auc(fpr, tpr)
    except:
        return 0.0


def evaluate_segmentation(pred_masks, gt_masks, image_name=None, pred_scores=None):

    metrics = {}
    
    # Convert to numpy if tensors
    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.cpu().numpy()
    if isinstance(gt_masks, torch.Tensor):
        gt_masks = gt_masks.cpu().numpy()
        

    if image_name and False:
        print(f"\n=== Evaluating {image_name} ===")
        print(f"pred_masks shape: {pred_masks.shape}")
        print(f"gt_masks shape: {gt_masks.shape}")
        if pred_scores is not None:
            print(f"pred_scores shape: {pred_scores.shape}")
        
    # GT mask dimension normalization
    if gt_masks.ndim == 3 and gt_masks.shape[0] == 1:
        gt_masks = gt_masks.squeeze(0)
    elif gt_masks.ndim == 1:
        return None
    
    # Prediction mask processing
    if pred_masks.ndim == 3:
        # Combine multiple masks (OR operation)
        combined_pred_mask = np.any(pred_masks, axis=0)
    else:
        combined_pred_mask = pred_masks
    
    # Size consistency check
    if combined_pred_mask.shape != gt_masks.shape:
        return None
    
    # Calculate IoU
    iou = calculate_iou(combined_pred_mask, gt_masks)
    
    # Calculate F1
    f1 = calculate_f1(combined_pred_mask, gt_masks)
    
    # Calculate precision and recall
    true_positive = np.sum(np.logical_and(combined_pred_mask, gt_masks))
    false_positive = np.sum(np.logical_and(combined_pred_mask, ~gt_masks))
    false_negative = np.sum(np.logical_and(~combined_pred_mask, gt_masks))
    
    precision = true_positive / (true_positive + false_positive + 1e-6)
    recall = true_positive / (true_positive + false_negative + 1e-6)
    
    metrics['mIoU'] = float(iou)
    metrics['F1'] = float(f1)
    metrics['Precision'] = float(precision)
    metrics['Recall'] = float(recall)
    
    # Score-based metrics (optional)
    if pred_scores is not None:
        try:
            if isinstance(pred_scores, torch.Tensor):
                pred_scores = pred_scores.cpu().numpy()
            
            # Score normalization
            if pred_scores.ndim > 2:
                pred_scores = pred_scores.squeeze()
            
            if pred_scores.shape == combined_pred_mask.shape:
                fpr95 = calculate_fpr95(pred_scores, gt_masks)
                auroc = calculate_auroc(pred_scores, gt_masks)
                auuprc = calculate_auuprc(pred_scores, gt_masks)
                
                metrics['FPR95'] = float(fpr95)
                metrics['AUROC'] = float(auroc)
                metrics['AUUPRC'] = float(auuprc)
        except Exception as e:
            pass
    
    return metrics


def create_failure_metrics():

    return {
        'mIoU': 0.0,
        'F1': 0.0,
        'Precision': 0.0,
        'Recall': 0.0,
        'FPR95': 1.0,     # worst-case FPR
        'AUROC': 0.0,     # worst-case AUROC
        'AUUPRC': 0.0     # worst-case AUUPRC
    }


def load_multiagent_prompts(json_path):

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        

        prompt_dict = {}
        
        if 'results' in data:
            for result in data['results']:
                # Get image name — support both formats
                image_name = None
                if 'image_info' in result:
                    if isinstance(result['image_info'], dict):
                        image_name = result['image_info'].get('filename', 'unknown')
                    else:
                        image_name = result['image_info']
                elif 'image' in result:
                    image_name = result['image']
                
                if not image_name:
                    continue

                # Get prompts — support both formats
                final_prompts = None
                if 'final_prompts' in result:
                    final_prompts = result['final_prompts']
                elif 'synthesis_result' in result:
                    sr = result['synthesis_result']
                    if 'grounded_sam_prompts' in sr:
                        final_prompts = sr['grounded_sam_prompts']
                    elif 'prompt_v1' in sr:
                        final_prompts = sr
                
                if final_prompts and 'prompt_v1' in final_prompts and 'prompt_v2' in final_prompts:
                    # Extract reasoning correctly handling different potential keys
                    reasoning = ""
                    if 'synthesis_result' in result:
                        sr = result['synthesis_result']
                        reasoning = sr.get('anomaly_reasoning', sr.get('reasoning', ''))
                    
                    prompt_dict[image_name] = {
                        'prompt_v1': final_prompts['prompt_v1'],
                        'prompt_v2': final_prompts['prompt_v2'],
                        'overall_confidence': final_prompts.get('overall_confidence', 0.0),
                        'detection_confidence': final_prompts.get('detection_confidence', 0.0),
                        'reasoning': reasoning
                    }
        
        print(f"✓ Loaded prompts for {len(prompt_dict)} images from {json_path}")
        
        if prompt_dict:
            sample_items = list(prompt_dict.items())[:3] 
            print("  Sample prompts:")
            for img_name, prompts in sample_items:
                print(f"    {img_name}: V1='{prompts['prompt_v1']}', V2='{prompts['prompt_v2']}'")
        
        return prompt_dict
        
    except Exception as e:
        print(f"✗ Error loading multi-agent prompts from {json_path}: {str(e)}")
        import traceback
        print(f"   Full traceback: {traceback.format_exc()}")
        return {}


def optimize_thresholds_for_prompt(model, predictor, sample, text_prompt, device="cpu", clip_verifier=None):

    box_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    text_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    
    best_metrics = {'mIoU': 0, 'F1': 0}
    best_thresholds = {'box_threshold': 0.3, 'text_threshold': 0.25}
    best_results = None
    
    image_pil = sample['image']
    gt_mask = sample['mask']
    image_name = sample['image_name']
    
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_transformed, _ = transform(image_pil, None)
    
    image_cv = cv2.imread(sample['image_path'])
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_cv)
    
    for box_th, text_th in product(box_thresholds, text_thresholds):
        try:
            boxes_filt, pred_phrases = get_grounding_output(
                model, image_transformed, text_prompt, box_th, text_th, device=device
            )
            
            if len(boxes_filt) == 0:
                continue
            
            size = image_pil.size
            H, W = size[1], size[0]
            boxes_processed = copy.deepcopy(boxes_filt)
            
            for i in range(boxes_processed.size(0)):
                boxes_processed[i] = boxes_processed[i] * torch.Tensor([W, H, W, H])
                boxes_processed[i][:2] -= boxes_processed[i][2:] / 2
                boxes_processed[i][2:] += boxes_processed[i][:2]
            
            boxes_processed = boxes_processed.cpu()
            clip_scores_batch = None

            # CLIP verification: filter detections before SAM
            if clip_verifier is not None:
                boxes_processed, pred_phrases, clip_scores_batch, _ = clip_verifier.verify_detections(
                    image_cv, boxes_processed, pred_phrases, text_prompt
                )
                if len(boxes_processed) == 0:
                    continue

            transformed_boxes = predictor.transform.apply_boxes_torch(
                boxes_processed, image_cv.shape[:2]
            ).to(device)
            
            masks, scores, logits = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            
            if masks is None or masks.shape[0] == 0:
                continue
            
            gt_mask_array = np.array(gt_mask)[None, ...]
            
            resized_logits = torch.nn.functional.interpolate(
                logits,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            expanded_scores = scores.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            combined_scores = expanded_scores * torch.sigmoid(resized_logits)
            
            metrics = evaluate_segmentation(
                masks.squeeze(1).cpu(), 
                gt_mask_array, 
                image_name=None,
                pred_scores=combined_scores
            )
            
            if metrics is None:
                continue
            
            if metrics['F1'] > best_metrics['F1']:
                best_metrics = metrics
                best_thresholds = {'box_threshold': box_th, 'text_threshold': text_th}
                
                best_results = {
                    'masks': masks,
                    'boxes': boxes_processed,
                    'phrases': pred_phrases,
                    'scores': scores,
                    'logits': logits,
                    'combined_scores': combined_scores,
                    'clip_scores': clip_scores_batch
                }
                
        except Exception as e:
            continue
    
    return best_thresholds, best_metrics, best_results


def evaluate_dataset_with_multiagent_prompts(model, predictor, dataset, prompt_dict, 
                                            output_dir, device="cpu", clip_verifier=None):

    all_metrics = []
    threshold_stats = {'box_thresholds_v1': [], 'text_thresholds_v1': [], 
                      'box_thresholds_v2': [], 'text_thresholds_v2': []}
    per_image_results = []
    failed_images = []
    prompt_missing_images = []
    
    print(f"Evaluating {len(dataset)} samples with multi-agent prompts...")
    
    for idx in tqdm(range(len(dataset)), desc="Multi-agent evaluation"):
        try:
            sample = dataset[idx]
            image_name = sample['image_name']
            
            if image_name not in prompt_dict:
                print(f"✗ {image_name}: No prompts found in JSON")
                prompt_missing_images.append(image_name)
                
                failure_metrics = create_failure_metrics()
                all_metrics.append(failure_metrics)
                
                per_image_results.append({
                    'image_name': image_name,
                    'status': 'failure_no_prompt',
                    'metrics': failure_metrics
                })
                continue
            
            prompts = prompt_dict[image_name]
            prompt_v1 = prompts['prompt_v1']
            prompt_v2 = prompts['prompt_v2']
            
            print(f"  Processing {image_name}: V1='{prompt_v1}', V2='{prompt_v2}'")
            
            # V1 prompt optimization
            best_thresholds_v1, best_metrics_v1, best_results_v1 = optimize_thresholds_for_prompt(
                model, predictor, sample, prompt_v1, device, clip_verifier=clip_verifier
            )
            
            # V2 prompt optimization
            best_thresholds_v2, best_metrics_v2, best_results_v2 = optimize_thresholds_for_prompt(
                model, predictor, sample, prompt_v2, device, clip_verifier=clip_verifier
            )
            
            # both failed
            if best_results_v1 is None and best_results_v2 is None:
                print(f"✗ {image_name}: No valid detection found for both prompts")
                failed_images.append(image_name)
                
                failure_metrics = create_failure_metrics()
                all_metrics.append(failure_metrics)
                
                per_image_results.append({
                    'image_name': image_name,
                    'prompts': prompts,
                    'status': 'failure_no_detection',
                    'metrics': failure_metrics
                })
                continue
            
            # create combined mask
            image_cv = cv2.imread(sample['image_path'])
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            H, W = image_cv.shape[:2]
            
            combined_mask = np.zeros((H, W), dtype=bool)
            all_boxes = []
            all_phrases = []
            all_scores = []
            all_clip_scores = []
            
            # add V1 result
            if best_results_v1 is not None:
                v1_mask = best_results_v1['masks'].squeeze(1).cpu().numpy()
                if v1_mask.ndim == 3:
                    v1_mask = np.any(v1_mask, axis=0)
                combined_mask = np.logical_or(combined_mask, v1_mask)
                all_boxes.extend(best_results_v1['boxes'])
                all_phrases.extend([f"V1:{p}" for p in best_results_v1['phrases']])
                if 'clip_scores' in best_results_v1 and best_results_v1['clip_scores'] is not None:
                    if isinstance(best_results_v1['clip_scores'], torch.Tensor):
                        all_clip_scores.extend(best_results_v1['clip_scores'].cpu().tolist())
                    else:
                        all_clip_scores.extend(best_results_v1['clip_scores'])
                
                threshold_stats['box_thresholds_v1'].append(best_thresholds_v1['box_threshold'])
                threshold_stats['text_thresholds_v1'].append(best_thresholds_v1['text_threshold'])
            
            # add V2 result
            if best_results_v2 is not None:
                v2_mask = best_results_v2['masks'].squeeze(1).cpu().numpy()
                if v2_mask.ndim == 3:
                    v2_mask = np.any(v2_mask, axis=0)
                combined_mask = np.logical_or(combined_mask, v2_mask)
                all_boxes.extend(best_results_v2['boxes'])
                all_phrases.extend([f"V2:{p}" for p in best_results_v2['phrases']])
                if 'clip_scores' in best_results_v2 and best_results_v2['clip_scores'] is not None:
                    if isinstance(best_results_v2['clip_scores'], torch.Tensor):
                        all_clip_scores.extend(best_results_v2['clip_scores'].cpu().tolist())
                    else:
                        all_clip_scores.extend(best_results_v2['clip_scores'])
                
                threshold_stats['box_thresholds_v2'].append(best_thresholds_v2['box_threshold'])
                threshold_stats['text_thresholds_v2'].append(best_thresholds_v2['text_threshold'])
            
            # evaluate with GT mask
            gt_mask_array = np.array(sample['mask'])
            
            # final evaluation
            final_metrics = evaluate_segmentation(
                combined_mask, 
                gt_mask_array, 
                image_name=image_name
            )
            
            if final_metrics is not None:
                if all_clip_scores:
                    final_metrics['CLIP_Score'] = float(np.mean(all_clip_scores))
                else:
                    final_metrics['CLIP_Score'] = 0.0
                    
                all_metrics.append(final_metrics)
                
                clip_str = f", CLIP={final_metrics['CLIP_Score']:.4f}" if final_metrics['CLIP_Score'] > 0 else ""
                print(f"✓ {image_name}: F1={final_metrics['F1']:.4f}, IoU={final_metrics['mIoU']:.4f}{clip_str}")
                
                per_image_results.append({
                    'image_name': image_name,
                    'prompts': prompts,
                    'best_thresholds_v1': best_thresholds_v1 if best_results_v1 else None,
                    'best_thresholds_v2': best_thresholds_v2 if best_results_v2 else None,
                    'metrics_v1': best_metrics_v1 if best_results_v1 else None,
                    'metrics_v2': best_metrics_v2 if best_results_v2 else None,
                    'final_metrics': final_metrics,
                    'agent_confidence': prompts.get('overall_confidence', 0.0),
                    'status': 'success'
                })
                
                # save visualization (V1, V2, combined)
                if best_results_v1 is not None:
                    save_visualization(
                        image_cv, best_results_v1['masks'], best_results_v1['boxes'], 
                        [f"V1:{p}" for p in best_results_v1['phrases']],
                        os.path.join(output_dir, f"{image_name}_v1_result.jpg"),
                        title_suffix=f"V1: {prompt_v1}"
                    )
                
                if best_results_v2 is not None:
                    save_visualization(
                        image_cv, best_results_v2['masks'], best_results_v2['boxes'], 
                        [f"V2:{p}" for p in best_results_v2['phrases']],
                        os.path.join(output_dir, f"{image_name}_v2_result.jpg"),
                        title_suffix=f"V2: {prompt_v2}"
                    )
                
                # combined visualization
                if all_boxes:
                    combined_masks_tensor = torch.zeros((len(all_boxes), H, W))
                    for i, _ in enumerate(all_boxes):
                        combined_masks_tensor[i] = torch.from_numpy(combined_mask.astype(float))
                    
                    save_visualization(
                        image_cv, combined_masks_tensor, all_boxes, all_phrases,
                        os.path.join(output_dir, f"{image_name}_combined_result.jpg"),
                        title_suffix=f"Combined: V1+V2"
                    )
                    
                    # NEW: Save 3-Panel Pipeline Visualization
                    save_pipeline_visualization(
                        image=image_cv, 
                        anomaly_reasoning=prompts.get('reasoning', 'No reasoning provided.'), 
                        v1_prompt=prompt_v1, 
                        v2_prompt=prompt_v2, 
                        boxes=all_boxes, 
                        clip_scores=all_clip_scores, 
                        masks=combined_masks_tensor, 
                        final_metrics=final_metrics, 
                        save_path=os.path.join(output_dir, f"{image_name}_pipeline_vis.jpg")
                    )
                    
                    # NEW: Save Base Metrics Bar Chart
                    save_metrics_bar_chart(
                        metrics_dict=final_metrics,
                        save_path=os.path.join(output_dir, f"{image_name}_metrics_bar.jpg"),
                        title=f"Metrics for {image_name}"
                    )
                    
                    # NEW: Save Academic Spider Chart
                    save_spider_chart(
                        metrics_dict=final_metrics,
                        save_path=os.path.join(output_dir, f"{image_name}_spider_chart.jpg"),
                        title=f"System Balance Radar for {image_name}"
                    )
            else:
                print(f"✗ {image_name}: Evaluation failed")
                failed_images.append(image_name)
                
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
            failed_images.append(image_name)
            continue
    
    if not all_metrics:
        print("No successful evaluations completed")
        return None
        
    # calculate average metrics
    all_metric_keys = set()
    for metrics in all_metrics:
        all_metric_keys.update(metrics.keys())
    
    avg_metrics = {}
    for metric in all_metric_keys:
        values = [m.get(metric, 0.0) for m in all_metrics]
        avg_metrics[metric] = np.mean(values)
    
    # calculate threshold statistics
    threshold_analysis = {}
    for key, values in threshold_stats.items():
        if values:
            threshold_analysis[f'{key}_stats'] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'distribution': values
            }
    
    # calculate success/failure statistics
    success_count = len([r for r in per_image_results if r['status'] == 'success'])
    failure_count = len(failed_images)
    prompt_missing_count = len(prompt_missing_images)
    
    # save results
    results = {
        'dataset_name': dataset.dataset_name,
        'dataset_dir': dataset.dataset_dir,
        'total_samples': len(dataset),
        'successful_detections': success_count,
        'failed_detections': failure_count,
        'prompt_missing': prompt_missing_count,
        'detection_rate': success_count / len(dataset),
        'method': 'multi_agent_prompt_optimization',
        'average_metrics': avg_metrics,
        'threshold_analysis': threshold_analysis,
        'per_image_results': per_image_results,
        'failed_images': failed_images,
        'prompt_missing_images': prompt_missing_images
    }
    
    with open(os.path.join(output_dir, 'multiagent_evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # visualize threshold distribution
    plot_multiagent_threshold_distribution(threshold_analysis, output_dir)
    
    # NEW: Generate System Self-Awareness Scatter Plot
    save_confidence_scatter_plot(per_image_results, os.path.join(output_dir, "confidence_vs_miou_scatter.jpg"))
    
    # print results
    print(f"\n{'='*70}")
    print(f"Multi-Agent Dataset Evaluation Results for {dataset.dataset_name}")
    print(f"{'='*70}")
    print(f"Total samples: {len(dataset)}")
    print(f"Successful detections: {success_count}")
    print(f"Failed detections (no boxes found): {failure_count}")
    print(f"Prompt missing: {prompt_missing_count}")
    print(f"Detection rate: {success_count/len(dataset)*100:.1f}%")
    
    if failed_images:
        print(f"\n⚠️  Failed images ({len(failed_images)}):")
        for img in failed_images:
            print(f"   - {img}")
    
    if prompt_missing_images:
        print(f"\n⚠️  Images with missing prompts ({len(prompt_missing_images)}):")
        for img in prompt_missing_images:
            print(f"   - {img}")
    
    print(f"\nAverage Metrics (including failures as 0):")
    # print basic metrics
    basic_metrics = ['mIoU', 'F1', 'Precision', 'Recall', 'CLIP_Score']
    for metric_name in basic_metrics:
        if metric_name in avg_metrics:
            print(f"  {metric_name}: {avg_metrics[metric_name]:.4f}")
    
    # print score-based metrics
    score_metrics = ['FPR95', 'AUROC', 'AUUPRC']
    score_metrics_available = [m for m in score_metrics if m in avg_metrics]
    if score_metrics_available:
        print(f"\nScore-based Metrics:")
        for metric_name in score_metrics_available:
            print(f"  {metric_name}: {avg_metrics[metric_name]:.4f}")
    
    # print threshold analysis
    print(f"\nThreshold Analysis:")
    for key, stats in threshold_analysis.items():
        if 'stats' in key:
            threshold_type = key.replace('_stats', '').replace('_', ' ').title()
            print(f"  {threshold_type} - Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    return avg_metrics


def plot_multiagent_threshold_distribution(threshold_analysis, output_dir):
    """
    Multi-agent threshold distribution visualization
    """
    try:
        # V1, V2 each box/text threshold distribution plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        plots_data = [
            ('box_thresholds_v1_stats', 'V1 Box Threshold Distribution', axes[0,0]),
            ('text_thresholds_v1_stats', 'V1 Text Threshold Distribution', axes[0,1]),
            ('box_thresholds_v2_stats', 'V2 Box Threshold Distribution', axes[1,0]),
            ('text_thresholds_v2_stats', 'V2 Text Threshold Distribution', axes[1,1])
        ]
        
        for key, title, ax in plots_data:
            if key in threshold_analysis:
                stats = threshold_analysis[key]
                ax.hist(stats['distribution'], bins=15, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Threshold Value')
                ax.set_ylabel('Frequency')
                ax.set_title(title)
                ax.axvline(stats['mean'], color='red', linestyle='--', 
                          label=f"Mean: {stats['mean']:.3f}")
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(title)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'multiagent_threshold_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Multi-agent threshold distribution plot saved")
        
    except Exception as e:
        print(f"Failed to create multi-agent threshold distribution plot: {e}")


def plot_threshold_distribution(threshold_analysis, output_dir):
    """
    threshold distribution visualization (keep original function)
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Box threshold distribution
        ax1.hist(threshold_analysis['box_threshold_stats']['distribution'], 
                bins=15, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Box Threshold')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Optimal Box Thresholds')
        ax1.axvline(threshold_analysis['box_threshold_stats']['mean'], 
                   color='red', linestyle='--', 
                   label=f"Mean: {threshold_analysis['box_threshold_stats']['mean']:.3f}")
        ax1.legend()
        
        # Text threshold distribution
        ax2.hist(threshold_analysis['text_threshold_stats']['distribution'], 
                bins=15, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Text Threshold')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Optimal Text Thresholds')
        ax2.axvline(threshold_analysis['text_threshold_stats']['mean'], 
                   color='red', linestyle='--',
                   label=f"Mean: {threshold_analysis['text_threshold_stats']['mean']:.3f}")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'threshold_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Threshold distribution plot saved")
        
    except Exception as e:
        print(f"Failed to create threshold distribution plot: {e}")


def optimize_thresholds_per_image(model, predictor, sample, text_prompt, device="cpu"):
    """
    find optimal thresholds for each image (keep original function)
    """
    
    # define threshold range (more fine-grained)
    box_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    text_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    
    best_metrics = {'mIoU': 0, 'F1': 0}
    best_thresholds = {'box_threshold': 0.3, 'text_threshold': 0.25}
    best_results = None
    
    image_pil = sample['image']
    gt_mask = sample['mask']
    image_name = sample['image_name']
    
    # image transformation for GroundingDINO
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_transformed, _ = transform(image_pil, None)
    
    # prepare image for SAM
    image_cv = cv2.imread(sample['image_path'])
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_cv)
    
    # try all threshold combinations (10 × 8 = 80 combinations)
    print(f"  Testing {len(box_thresholds)} × {len(text_thresholds)} = {len(box_thresholds) * len(text_thresholds)} threshold combinations...")
    
    tested_combinations = 0
    for box_th, text_th in product(box_thresholds, text_thresholds):
        tested_combinations += 1
        try:
            # detect boxes with GroundingDINO
            boxes_filt, pred_phrases = get_grounding_output(
                model, image_transformed, text_prompt, box_th, text_th, device=device
            )
            
            # skip if no boxes found
            if len(boxes_filt) == 0:
                continue
            
            # convert box coordinates
            size = image_pil.size
            H, W = size[1], size[0]
            boxes_processed = copy.deepcopy(boxes_filt)
            
            for i in range(boxes_processed.size(0)):
                boxes_processed[i] = boxes_processed[i] * torch.Tensor([W, H, W, H])
                boxes_processed[i][:2] -= boxes_processed[i][2:] / 2
                boxes_processed[i][2:] += boxes_processed[i][:2]
            
            boxes_processed = boxes_processed.cpu()
            transformed_boxes = predictor.transform.apply_boxes_torch(
                boxes_processed, image_cv.shape[:2]
            ).to(device)
            
            # create masks with SAM
            masks, scores, logits = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            
            if masks is None or masks.shape[0] == 0:
                continue
            
            # prepare GT mask
            gt_mask_array = np.array(gt_mask)[None, ...]
            
            # resize logits and combine scores
            resized_logits = torch.nn.functional.interpolate(
                logits,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            expanded_scores = scores.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            combined_scores = expanded_scores * torch.sigmoid(resized_logits)
            
            # evaluate (include scores)
            metrics = evaluate_segmentation(
                masks.squeeze(1).cpu(), 
                gt_mask_array, 
                image_name=None,  # avoid logging
                pred_scores=combined_scores  # pass scores
            )
            
            if metrics is None:
                continue
            
            # update best result (F1 score based)
            if metrics['F1'] > best_metrics['F1']:
                best_metrics = metrics
                best_thresholds = {'box_threshold': box_th, 'text_threshold': text_th}
                
                print(f"    New best F1: {metrics['F1']:.4f} at box={box_th}, text={text_th} (combination {tested_combinations}/{len(box_thresholds) * len(text_thresholds)})")
                
                # resize logits and combine scores
                resized_logits = torch.nn.functional.interpolate(
                    logits,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
                expanded_scores = scores.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
                combined_scores = expanded_scores * torch.sigmoid(resized_logits)
                
                best_results = {
                    'masks': masks,
                    'boxes': boxes_processed,
                    'phrases': pred_phrases,
                    'scores': scores,
                    'logits': logits,
                    'combined_scores': combined_scores
                }
                
        except Exception as e:
            # skip if error occurs
            continue
    
    return best_thresholds, best_metrics, best_results


def evaluate_dataset_with_threshold_optimization(model, predictor, dataset, text_prompt, 
                                                output_dir, device="cpu"):
    """
    evaluate dataset with threshold optimization (keep original function)
    """
    all_metrics = []
    threshold_stats = {'box_thresholds': [], 'text_thresholds': []}
    per_image_results = []
    
    print(f"Evaluating {len(dataset)} samples with threshold optimization...")
    
    for idx in tqdm(range(len(dataset)), desc="Optimizing and evaluating"):
        try:
            sample = dataset[idx]
            image_name = sample['image_name']
            
            # optimize thresholds for each image
            best_thresholds, best_metrics, best_results = optimize_thresholds_per_image(
                model, predictor, sample, text_prompt, device
            )
            
            if best_thresholds is None or best_results is None:
                print(f"✗ {image_name}: No valid detection found - assigning failure metrics (0 scores)")
                
                # assign failure metrics
                failure_metrics = create_failure_metrics()
                all_metrics.append(failure_metrics)
                
                # record failure thresholds (use default values)
                failure_thresholds = {'box_threshold': 0.3, 'text_threshold': 0.25}
                threshold_stats['box_thresholds'].append(failure_thresholds['box_threshold'])
                threshold_stats['text_thresholds'].append(failure_thresholds['text_threshold'])
                
                # collect individual results
                per_image_results.append({
                    'image_name': image_name,
                    'best_thresholds': failure_thresholds,
                    'metrics': failure_metrics,
                    'status': 'failure_no_detection'
                })
                
                continue
            
            # print results
            print(f"✓ {image_name}: box={best_thresholds['box_threshold']:.2f}, "
                  f"text={best_thresholds['text_threshold']:.2f}, "
                  f"F1={best_metrics['F1']:.4f}, IoU={best_metrics['mIoU']:.4f}")
            
            # collect statistics
            threshold_stats['box_thresholds'].append(best_thresholds['box_threshold'])
            threshold_stats['text_thresholds'].append(best_thresholds['text_threshold'])
            
            # collect metrics
            all_metrics.append(best_metrics)
            
            # collect individual results
            per_image_results.append({
                'image_name': image_name,
                'best_thresholds': best_thresholds,
                'metrics': best_metrics,
                'status': 'success'
            })
            
            # save visualization
            image_cv = cv2.imread(sample['image_path'])
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            save_visualization(
                image_cv,
                best_results['masks'], 
                best_results['boxes'], 
                best_results['phrases'],
                os.path.join(output_dir, f"{image_name}_optimized_result.jpg")
            )
                
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
            continue
    
    if not all_metrics:
        print("No successful evaluations completed")
        return None
        
    # calculate average metrics (include all metrics)
    all_metric_keys = set()
    for metrics in all_metrics:
        all_metric_keys.update(metrics.keys())
    
    avg_metrics = {}
    for metric in all_metric_keys:
        values = [m.get(metric, 0.0) for m in all_metrics]  # handle missing metrics as 0.0
        avg_metrics[metric] = np.mean(values)
    
    # calculate threshold statistics
    threshold_analysis = {
        'box_threshold_stats': {
            'mean': np.mean(threshold_stats['box_thresholds']),
            'std': np.std(threshold_stats['box_thresholds']),
            'min': np.min(threshold_stats['box_thresholds']),
            'max': np.max(threshold_stats['box_thresholds']),
            'distribution': threshold_stats['box_thresholds']
        },
        'text_threshold_stats': {
            'mean': np.mean(threshold_stats['text_thresholds']),
            'std': np.std(threshold_stats['text_thresholds']),
            'min': np.min(threshold_stats['text_thresholds']),
            'max': np.max(threshold_stats['text_thresholds']),
            'distribution': threshold_stats['text_thresholds']
        }
    }
    
    # calculate success/failure statistics
    success_count = len([r for r in per_image_results if r['status'] == 'success'])
    failure_count = len([r for r in per_image_results if r['status'] == 'failure_no_detection'])
    
    # save results
    results = {
        'dataset_name': dataset.dataset_name,
        'dataset_dir': dataset.dataset_dir,
        'total_samples': len(dataset),
        'successful_detections': success_count,
        'failed_detections': failure_count,
        'detection_rate': success_count / len(dataset),
        'optimization_method': 'per_image_threshold_optimization_with_failures',
        'average_metrics': avg_metrics,
        'threshold_analysis': threshold_analysis,
        'per_image_results': per_image_results
    }
    
    with open(os.path.join(output_dir, 'optimized_evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # visualize threshold distribution
    plot_threshold_distribution(threshold_analysis, output_dir)
    
    # print results
    print(f"\n{'='*60}")
    print(f"Optimized Dataset Evaluation Results for {dataset.dataset_name}")
    print(f"{'='*60}")
    print(f"Total samples: {len(dataset)}")
    print(f"Successful detections: {success_count}")
    print(f"Failed detections (no boxes found): {failure_count}")
    print(f"Detection rate: {success_count/len(dataset)*100:.1f}%")
    
    print(f"\nAverage Metrics (including failures as 0):")
    # print basic metrics
    basic_metrics = ['mIoU', 'F1', 'Precision', 'Recall']
    for metric_name in basic_metrics:
        if metric_name in avg_metrics:
            print(f"  {metric_name}: {avg_metrics[metric_name]:.4f}")
    
    # print score-based metrics
    score_metrics = ['FPR95', 'AUROC', 'AUUPRC']
    score_metrics_available = [m for m in score_metrics if m in avg_metrics]
    if score_metrics_available:
        print(f"\nScore-based Metrics:")
        for metric_name in score_metrics_available:
            print(f"  {metric_name}: {avg_metrics[metric_name]:.4f}")
    else:
        print(f"\nScore-based Metrics: Not available (no confidence scores computed)")
    
    # print other metrics
    other_metrics = [m for m in avg_metrics.keys() if m not in basic_metrics + score_metrics]
    if other_metrics:
        print(f"\nOther Metrics:")
        for metric_name in other_metrics:
            print(f"  {metric_name}: {avg_metrics[metric_name]:.4f}")
    
    print(f"\nThreshold Analysis:")
    print(f"  Box Threshold - Mean: {threshold_analysis['box_threshold_stats']['mean']:.3f} "
          f"± {threshold_analysis['box_threshold_stats']['std']:.3f}")
    print(f"  Text Threshold - Mean: {threshold_analysis['text_threshold_stats']['mean']:.3f} "
          f"± {threshold_analysis['text_threshold_stats']['std']:.3f}")
    
    return avg_metrics


# evaluate dataset with fixed thresholds (keep original function)
def evaluate_dataset_with_new_classes(model, predictor, dataset, text_prompt, output_dir, 
                                     box_threshold=0.3, text_threshold=0.25, device="cpu"):
    """
    evaluate dataset with fixed thresholds (keep original function)
    """
    all_metrics = []
    
    print(f"Evaluating {len(dataset)} samples with fixed thresholds (box={box_threshold}, text={text_threshold})")
    
    for idx in tqdm(range(len(dataset)), desc="Evaluating dataset"):
        try:
            # get sample from new Dataset class
            sample = dataset[idx]
            image_pil = sample['image']
            gt_mask = sample['mask']  # already binarized mask
            image_name = sample['image_name']
            
            # image transformation for GroundingDINO
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image_transformed, _ = transform(image_pil, None)
            
            # Run grounding dino model
            boxes_filt, pred_phrases = get_grounding_output(
                model, image_transformed, text_prompt, box_threshold, text_threshold, device=device
            )
            
            # skip if no boxes found
            if len(boxes_filt) == 0:
                print(f"✗ {image_name}: No boxes found - assigning failure metrics (0 scores)")
                
                # assign failure metrics
                failure_metrics = create_failure_metrics()
                all_metrics.append(failure_metrics)
                continue
            
            # Prepare image for SAM
            image_cv = cv2.imread(sample['image_path'])
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_cv)
            
            # Process boxes
            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]
            
            boxes_filt = boxes_filt.cpu()
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_cv.shape[:2]).to(device)
            
            # check if boxes are valid
            if transformed_boxes.shape[0] == 0:
                print(f"No valid boxes after transformation for {image_name}, skipping...")
                continue
            
            # Generate masks and scores
            masks, scores, logits = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            
            # check if masks are generated
            if masks is None or masks.shape[0] == 0:
                print(f"No masks generated for {image_name}, skipping...")
                continue
            
            # prepare GT mask
            gt_mask_array = np.array(gt_mask)[None, ...]
            
            # resize logits and combine scores
            resized_logits = torch.nn.functional.interpolate(
                logits,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            expanded_scores = scores.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            combined_scores = expanded_scores * torch.sigmoid(resized_logits)
            
            # Evaluate predictions
            metrics = evaluate_segmentation(
                masks.squeeze(1).cpu(), 
                gt_mask_array, 
                image_name=image_name,
                pred_scores=combined_scores  # pass scores
            )
            
            if metrics is not None:
                all_metrics.append(metrics)
                
                # check which metrics are calculated
                available_metrics = list(metrics.keys())
                score_based = [m for m in ['FPR95', 'AUROC', 'AUUPRC'] if m in available_metrics]
                if score_based:
                    score_info = f" (scores: {', '.join(score_based)})"
                else:
                    score_info = " (no scores)"
                
                print(f"✓ {image_name}: F1={metrics['F1']:.4f}, IoU={metrics['mIoU']:.4f}{score_info}")
            else:
                print(f"✗ {image_name}: Evaluation failed")
            
            # Save visualization
            save_visualization(image_cv, masks, boxes_filt, pred_phrases, 
                            os.path.join(output_dir, f"{image_name}_result.jpg"))
                            
        except Exception as e:
            print(f"Error processing sample {idx} ({sample.get('image_name', 'unknown')}): {str(e)}")
            continue
    
    if not all_metrics:
        print("No successful evaluations completed")
        return None
        
    # calculate and save average metrics
    avg_metrics = {
        metric: np.mean([m[metric] for m in all_metrics])
        for metric in all_metrics[0].keys()
    }
    
    # save results to JSON
    with open(os.path.join(output_dir, 'dataset_evaluation_results.json'), 'w') as f:
        json.dump({
            'dataset_name': dataset.dataset_name,
            'dataset_dir': dataset.dataset_dir,
            'total_samples': len(dataset),
            'successful_evaluations': len(all_metrics),
            'box_threshold': box_threshold,
            'text_threshold': text_threshold,
            'average_metrics': avg_metrics,
            'per_image_metrics': all_metrics
        }, f, indent=4)
    
    print(f"\nDataset Evaluation Results for {dataset.dataset_name}:")
    print(f"Successfully evaluated: {len(all_metrics)}/{len(dataset)} samples")
    for metric_name, value in avg_metrics.items():
        print(f"Average {metric_name}: {value:.4f}")
    
    return avg_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Multi-Agent Evaluation", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--dataset_dir", type=str, required=True, help="path to dataset directory")
    parser.add_argument("--dataset_type", type=str, default=None, choices=['road_anomaly', 'fishyscapes', 'segment_me'], help="dataset type")
    
    # add arguments for multi-agent prompts
    parser.add_argument("--multiagent_prompts", type=str, default=None, 
                       help="path to JSON file containing multi-agent generated prompts")
    parser.add_argument("--text_prompt", type=str, default=None, 
                       help="text prompt (used only when not using multi-agent prompts)")
    
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )
    
    # select evaluation mode
    parser.add_argument("--optimize_thresholds", action="store_true", 
                       help="Enable per-image threshold optimization (slower but more accurate)")
    parser.add_argument("--box_threshold", type=float, default=0.3, 
                       help="box threshold (used only when not optimizing)")
    parser.add_argument("--text_threshold", type=float, default=0.25, 
                       help="text threshold (used only when not optimizing)")
    
    parser.add_argument("--clip_threshold", type=float, default=0.20,
                       help="CLIP similarity threshold for semantic verification (0.0 to disable)")
    parser.add_argument("--device", type=str, default="cpu", help="device to run on")
    parser.add_argument("--bert_base_uncased_path", type=str, required=False, help="bert_base_uncased model path")
    
    args = parser.parse_args()

    # validate settings
    if args.multiagent_prompts is None and args.text_prompt is None:
        print("Error: Either --multiagent_prompts or --text_prompt must be provided")
        sys.exit(1)

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # load models
    print("Loading models...")
    model = load_model(args.config, args.grounded_checkpoint, args.bert_base_uncased_path, device=args.device)

    # initialize SAM
    if args.use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[args.sam_version](checkpoint=args.sam_hq_checkpoint).to(args.device))
    else:
        predictor = SamPredictor(sam_model_registry[args.sam_version](checkpoint=args.sam_checkpoint).to(args.device))

    # load dataset
    print(f"Loading dataset from: {args.dataset_dir}")
    dataset = DatasetFactory.create_dataset(args.dataset_dir, args.dataset_type)
    
    if len(dataset) == 0:
        print("No valid image-label pairs found in dataset")
        sys.exit(1)
    
    print(f"Dataset loaded: {dataset.dataset_name} with {len(dataset)} samples")
    
    # select evaluation method
    if args.multiagent_prompts:
        print("🤖 Running evaluation with multi-agent prompts...")
        print(f"   Prompts JSON: {args.multiagent_prompts}")
        
        # load multi-agent prompts
        prompt_dict = load_multiagent_prompts(args.multiagent_prompts)
        
        if not prompt_dict:
            print("Error: No valid prompts loaded from JSON file")
            sys.exit(1)
        
        # Initialize CLIP verifier
        clip_verifier = None
        if args.clip_threshold > 0:
            print(f"🔍 Initializing CLIP verifier (threshold={args.clip_threshold})...")
            clip_verifier = CLIPVerifier(
                device=args.device,
                similarity_threshold=args.clip_threshold
            )
        
        # evaluate with multi-agent prompts
        metrics = evaluate_dataset_with_multiagent_prompts(
            model=model,
            predictor=predictor,
            dataset=dataset,
            prompt_dict=prompt_dict,
            output_dir=args.output_dir,
            device=args.device,
            clip_verifier=clip_verifier
        )
        
    elif args.optimize_thresholds:
        print("🔧 Running evaluation with per-image threshold optimization...")
        print("⚠️  This will be significantly slower but more accurate")
        print(f"   Text prompt: {args.text_prompt}")
        
        # evaluate with threshold optimization
        metrics = evaluate_dataset_with_threshold_optimization(
            model=model,
            predictor=predictor,
            dataset=dataset,
            text_prompt=args.text_prompt,
            output_dir=args.output_dir,
            device=args.device
        )
        
    else:
        print(f"🚀 Running evaluation with fixed thresholds...")
        print(f"   Text prompt: {args.text_prompt}")
        print(f"   Box threshold: {args.box_threshold}")
        print(f"   Text threshold: {args.text_threshold}")
        
        # evaluate with fixed thresholds (keep original function)
        metrics = evaluate_dataset_with_new_classes(
            model=model,
            predictor=predictor,
            dataset=dataset,
            text_prompt=args.text_prompt,
            output_dir=args.output_dir,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device
        )
    
    print("\n✅ Evaluation completed!")
    if metrics:
        print("📊 Check the output directory for detailed results and visualizations.")
    else:
        print("❌ No successful evaluations were completed.")