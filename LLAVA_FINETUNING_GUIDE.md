# LLaVA-7B Fine-Tuning Guide for MAVR-OOD

## Goal

This guide explains whether it makes sense to fine-tune the current LLaVA-7B model used in MAVR-OOD, what improvement to realistically expect from only about 50 selected BDD images, and how to do that safely.

It is written for the current repo, where LLaVA is used in two places:

- `src/agents/` for OOD multi-agent reasoning
- `src/text_guided/` for scene understanding, attribute matching, and final reasoning

---

## Short Answer

Yes, you can fine-tune LLaVA-7B on your own road-scene data.

Yes, it can improve performance, but only if the training data is formatted well and the task is narrow enough.

With only about 50 images, the best approach is:

- use **LoRA / QLoRA instruction tuning**
- keep the scope narrow
- train for your specific reasoning style or driving-scene vocabulary
- do not expect a huge general intelligence jump

With 50 images, you are much more likely to improve:

- scene descriptions for your domain
- anomaly reasoning phrasing
- attribute matching on road scenes
- consistency of JSON outputs

You are much less likely to improve:

- broad visual knowledge
- true object detection ability
- general multimodal reasoning outside your task

---

## Is LLaVA Already Trained on Your 50 BDD Images?

We cannot prove whether your exact 50 selected BDD images were seen unless you have a full training-data trace or hash-level overlap check.

Practically, the answer is:

- **the exact 50 images are probably not something you should assume were used**
- **BDD100K itself is a public, well-known driving dataset, but LLaVA's official training recipe does not list BDD100K as a core named dataset**

So the safe assumption is:

- LLaVA already knows general road-scene concepts
- it may have seen visually similar internet driving images
- it may or may not have seen some BDD-style examples
- you should treat your selected 50-image subset as **custom adaptation data**, not something guaranteed to already be in the official training mix

That means fine-tuning is still a valid experiment.

---

## What Official LLaVA Training Looks Like

Official LLaVA training is generally described as a two-stage process:

1. feature alignment between the vision encoder and language model
2. visual instruction tuning on multimodal instruction-following data

The official LLaVA materials describe training on large-scale image-text and multimodal instruction datasets, not on a small driving-scene dataset alone. That is why your 50-image set should be viewed as a **task adaptation layer**, not full model retraining.

---

## Will Fine-Tuning Improve Performance?

### When it can help

Fine-tuning can help if your current failures look like these:

- scene agent misses important road-specific context
- attribute agent gives vague or template-like prompts
- reasoning agent explains anomalies poorly
- JSON outputs are inconsistent
- road-scene anomalies are described too generically

### When it will not help much

Fine-tuning will not directly fix:

- GroundingDINO localization misses
- SAM segmentation errors
- CLIP verification threshold problems
- spatial filter logic bugs

Those pieces are outside LLaVA.

So if your problem is mainly:

- "LLaVA is describing the wrong thing"

then fine-tuning may help.

If your problem is mainly:

- "the detector box is wrong"

then fine-tuning LLaVA will only help indirectly, if better prompts improve grounding.

---

## Best Strategy with Only 50 Images

With 50 images, the right goal is **specialization**, not full retraining.

Recommended options, from best to worst:

1. **LoRA fine-tuning**
2. **QLoRA fine-tuning**
3. prompt engineering plus synthetic data expansion
4. full fine-tuning

### Recommendation

Use **LoRA or QLoRA**.

Why:

- much cheaper
- less overfitting than full fine-tuning
- realistic on Colab or A100
- easier to merge or keep as an adapter

Do **not** full-fine-tune the whole 7B model on only 50 images.

---

## Biggest Risk: Overfitting

With 50 images, overfitting is the main risk.

Symptoms of overfitting:

- model repeats training-style wording
- confidence looks high but generalization gets worse
- works on your 50 images but fails on new road scenes
- JSON output looks better but reasoning becomes less flexible

To reduce overfitting:

- keep epochs low
- use a validation split
- use LoRA, not full fine-tuning
- expand the dataset with multiple instruction variants per image
- keep labels high quality

---

## What Data Should You Prepare?

For this project, your fine-tuning data should match the way MAVR uses LLaVA.

That means you should create samples for tasks like:

- scene understanding
- anomaly reasoning
- attribute matching
- text-guided object identification
- short grounding prompt generation
- strict JSON output

### Best data format idea

Each image should produce several supervised examples, not only one.

For example, for one road image:

1. scene description example
2. anomaly analysis example
3. attribute matching example
4. grounding prompt generation example
5. final reasoning example

So 50 images can become perhaps:

- 200 to 500 instruction samples

That is much more useful than only 50 one-line samples.

---

## Suggested Dataset Design for MAVR-OOD

For each image, create entries like the following.

### Task 1: Scene Understanding

Input:

- image
- instruction: "Analyze this road scene and return JSON with scene type, lighting, and visible objects."

Target:

- strict JSON

### Task 2: Attribute Matching

Input:

- image
- scene JSON
- user query like `"the white car on the right"`

Target:

- match reasoning JSON
- recommended prompt like `"white car"`

### Task 3: OOD Reasoning

Input:

- image
- instruction: "Identify what object is unusual in this road scene and explain why."

Target:

- anomaly reasoning JSON or concise explanation

### Task 4: Grounding Prompt Generation

Input:

- image
- agent summaries

Target:

- `prompt_v1`
- `prompt_v2`

### Task 5: Final Explanation

Input:

- image
- pipeline summary text

Target:

- a clean final explanation paragraph

This is better aligned with MAVR than generic captioning.

---

## Train/Validation Split

Do not train on all 50.

Use something like:

- 40 images for train
- 10 images for validation

If possible:

- keep a separate hidden test set later

Because otherwise you will not know whether the model truly improved.

---

## LoRA vs QLoRA for This Project

### LoRA

Good if:

- you have strong GPU memory
- you want stable training

### QLoRA

Good if:

- memory is limited
- you want lower VRAM usage

For Colab A100, either can work.

For smaller GPU budgets, QLoRA is safer.

---

## Practical Fine-Tuning Workflow

## Stage 1: Define the target task clearly

Before training, decide what exactly you want to improve:

- OOD agent reasoning
- text-guided scene analysis
- attribute matching
- JSON reliability

If you try to improve everything at once with only 50 images, the signal will be weak.

Best first target:

- scene understanding + anomaly reasoning + JSON consistency

---

## Stage 2: Convert your data into instruction format

Typical sample structure:

```json
{
  "id": "bdd_001_scene",
  "image": "images/bdd_001.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nAnalyze this road scene and return JSON with scene_type, lighting, and visible objects."
    },
    {
      "from": "gpt",
      "value": "{\"scene_type\":\"urban road\",\"lighting\":\"daylight\",\"objects\":[{\"name\":\"white car\",\"position\":\"right\",\"color\":\"white\",\"size\":\"medium\"}]}"
    }
  ]
}
```

For another task:

```json
{
  "id": "bdd_001_ood",
  "image": "images/bdd_001.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nIdentify any anomalous object in this road scene and explain why it is inappropriate."
    },
    {
      "from": "gpt",
      "value": "{\"inappropriate_objects\":\"cow\",\"reasoning\":\"A cow is not expected on an active urban roadway and creates a safety hazard.\"}"
    }
  ]
}
```

This format is close to common LLaVA conversation-style fine-tuning data.

---

## Stage 3: Train with LoRA

High-level recipe:

1. start from a LLaVA-1.5 7B base checkpoint
2. freeze most model weights
3. train LoRA adapters on your custom instruction data
4. evaluate on held-out road scenes
5. use the adapter during inference

### Reasonable starter hyperparameters

These are practical starting points, not fixed rules:

- LoRA rank: `16` or `32`
- LoRA alpha: `32` or `64`
- learning rate: `1e-4` to `2e-4` for LoRA-only
- epochs: start with `2` to `5`
- batch size: as memory allows
- image resolution: keep aligned with your LLaVA training codebase
- early stopping: yes, if validation gets worse

With only 50 images, fewer epochs is usually safer.

---

## Stage 4: Evaluate carefully

Do not decide using only training samples.

Measure whether the tuned model improves:

- scene JSON quality
- prompt quality
- reasoning quality
- anomaly naming accuracy
- consistency on unseen road images

For MAVR, compare:

- base LLaVA outputs
- fine-tuned LLaVA outputs
- downstream MAVR pipeline effect

Especially check whether better LLaVA outputs actually improve:

- GroundingDINO prompt quality
- candidate selection
- final success rate

---

## What Improvement Is Realistic?

With a good 50-image dataset and LoRA:

- modest but useful improvement is realistic
- strong improvement on very similar scenes is realistic
- dramatic general-purpose improvement is not realistic

Expected gains are most likely in:

- formatting consistency
- road-scene vocabulary
- anomaly reasoning style
- better short prompts for your pipeline

Expected gains are least likely in:

- open-world visual grounding
- generic multimodal intelligence
- robust generalization to very different datasets

---

## Best Way to Make 50 Images More Valuable

If you only have 50 images, increase supervision quality rather than just collecting more raw images.

Do this:

- create multiple tasks per image
- create multiple user-query variants per image
- include both positive and ambiguous examples
- include strict JSON targets
- include hard cases where the model currently fails

For example, one image can yield:

- 3 text-guided queries
- 1 scene JSON
- 1 anomaly reasoning sample
- 1 grounding prompt sample

That is how a small image set becomes useful.

---

## Should You Use BDD Images for This?

Yes, selectively chosen BDD images are reasonable for this project because:

- they are driving-domain images
- they match the road-scene context of MAVR
- they can teach road-scene-specific reasoning style

But you should choose them intentionally:

- edge cases
- unusual object placement
- hard attribute matches
- difficult lighting
- crowded scenes

If your 50 images are too easy, the gain will be small.

---

## Should You Fine-Tune Now?

My honest recommendation:

Yes, but only as a controlled experiment.

Best sequence:

1. establish baseline performance first
2. prepare high-quality instruction data from the 50 images
3. run LoRA fine-tuning
4. compare base vs tuned model on held-out images
5. keep the tuned adapter only if downstream MAVR improves

Do not assume fine-tuning always helps. Measure it.

---

## Recommended Experiment Plan

### Experiment A: JSON reliability

Train only for:

- scene JSON
- attribute matching JSON
- anomaly JSON

Goal:

- fewer malformed outputs

### Experiment B: prompt quality

Train for:

- better `recommended_prompt`
- better `prompt_v1` and `prompt_v2`

Goal:

- better downstream grounding

### Experiment C: final reasoning

Train for:

- concise anomaly explanations
- consistent road-scene hazard reasoning

Goal:

- better interpretability

This staged approach is better than one large unfocused fine-tune.

---

## Example Minimal Folder Layout

```text
finetune/
├── images/
│   ├── bdd_001.jpg
│   ├── bdd_002.jpg
│   └── ...
├── train.json
├── val.json
└── notes.md
```

---

## Example Training Commands

The exact command depends on which LLaVA training codebase you use, but conceptually it will look like:

```bash
bash scripts/finetune_lora.sh \
  --model_name_or_path liuhaotian/llava-v1.5-7b \
  --data_path /path/to/train.json \
  --image_folder /path/to/images \
  --lora_enable True \
  --lora_r 16 \
  --lora_alpha 32
```

If you use QLoRA, the command will include 4-bit loading and related settings.

For this repo, the main decision is not the exact shell command yet. The main decision is:

- what task data to create
- how to evaluate whether the tuned model actually helps MAVR

---

## What I Recommend for MAVR Specifically

If we fine-tune for this project, I would target these outputs first:

1. `src/text_guided/scene_agent.py`
2. `src/text_guided/attribute_agent.py`
3. `src/agents/agent1.py` to `src/agents/agent5.py`

Priority order:

1. scene understanding
2. anomaly reasoning
3. prompt generation
4. final explanation style

I would not fine-tune first for generic captioning.

---

## Final Recommendation

### Yes, try fine-tuning if:

- you use LoRA or QLoRA
- you create high-quality instruction data
- you keep a validation set
- you evaluate on downstream MAVR behavior

### No, do not do it this way:

- full fine-tune on only 50 images
- no validation split
- only one annotation per image
- assuming any improvement without measuring it

If you want, the next best step is to build:

1. a training data schema for your 50 BDD images
2. a JSON template for each MAVR task
3. a Colab fine-tuning notebook for LoRA/QLoRA

---

## Sources

- LLaVA official repository: https://github.com/haotian-liu/LLaVA
- LLaVA project page: https://llava-vl.github.io/
- BDD100K official repository: https://github.com/bdd100k/bdd100k
- BDD100K dataset site: https://bdd-data.berkeley.edu/
