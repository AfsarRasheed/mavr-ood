"""
Florence-2 text-guided grounding backend.

This module is intentionally isolated so the rest of the text-guided pipeline
can keep using the same CLIP / semantic judging / reliability / SAM flow.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from src.text_guided.candidate_adapter import normalize_candidate_output


FLORENCE2_TASK = "<CAPTION_TO_PHRASE_GROUNDING>"


def _ensure_rgb(image_pil: Image.Image) -> Image.Image:
    return image_pil if image_pil.mode == "RGB" else image_pil.convert("RGB")


def _extract_grounding_items(processed_output: Any) -> tuple[list[list[float]], list[str], list[float]]:
    """
    Normalize the various Florence-2 post-processing payload shapes into
    simple box/label/score lists.
    """
    if not processed_output:
        return [], [], []

    if isinstance(processed_output, dict):
        candidates = None
        for key in ("bboxes", "boxes", "candidates"):
            if key in processed_output:
                candidates = processed_output[key]
                break
        if candidates is None and "answer" in processed_output:
            candidates = processed_output["answer"]
    else:
        candidates = processed_output

    boxes: list[list[float]] = []
    labels: list[str] = []
    scores: list[float] = []

    for item in candidates or []:
        if isinstance(item, dict):
            box = item.get("bbox") or item.get("box") or item.get("xyxy")
            label = item.get("label") or item.get("phrase") or item.get("text") or "florence2"
            score = item.get("score", item.get("confidence", 0.0))
        elif isinstance(item, (list, tuple)) and len(item) >= 4:
            box = item[:4]
            label = "florence2"
            score = item[4] if len(item) > 4 else 0.0
        else:
            continue

        if not box or len(box) < 4:
            continue

        boxes.append([float(v) for v in box[:4]])
        labels.append(str(label))
        scores.append(float(score or 0.0))

    return boxes, labels, scores


def run_florence2_grounding(
    *,
    florence_model,
    florence_processor,
    image_pil: Image.Image,
    prompt: str,
    device: str,
    image_size: tuple[int, int],
    max_new_tokens: int = 256,
) -> dict:
    """
    Run Florence-2 phrase grounding and return the shared candidate format.
    """
    image_pil = _ensure_rgb(image_pil)
    task_prompt = f"{FLORENCE2_TASK}{prompt.strip()}"

    inputs = florence_processor(
        text=task_prompt,
        images=image_pil,
        return_tensors="pt",
    )
    inputs = {name: value.to(device) for name, value in inputs.items()}

    generated_ids = florence_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        num_beams=3,
        do_sample=False,
    )
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    processed = florence_processor.post_process_generation(
        generated_text,
        task=FLORENCE2_TASK,
        image_size=(image_size[1], image_size[0]),
    )

    task_output = processed.get(FLORENCE2_TASK, processed) if isinstance(processed, dict) else processed
    boxes_xyxy, labels, scores = _extract_grounding_items(task_output)

    return normalize_candidate_output(
        backend_name="florence2",
        boxes_xyxy=boxes_xyxy,
        image_size=image_size,
        labels=labels,
        scores=scores,
        raw_response={
            "task_prompt": task_prompt,
            "generated_text": generated_text,
            "processed": processed,
        },
    )
