# Fine-Tuning LLaVA-7B with LoRA on Road Scene Images

Complete end-to-end guide to fine-tune LLaVA-7B on your BDD road images using LoRA on Google Colab (free T4 GPU).

---

## Step 1: Prepare Your Dataset

### Folder Structure

```
/content/finetune_data/
├── images/
│   ├── bdd_001.jpg
│   ├── bdd_002.jpg
│   ├── bdd_003.jpg
│   └── ... (your 50 BDD images)
└── train_data.json
```

### JSON Format (train_data.json)

Create **3-5 Q&A pairs per image**. Here are the 5 types to cover:

```json
[
  {
    "image": "bdd_001.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nDescribe the objects in this road scene in detail."},
      {"from": "gpt", "value": "This is a daytime highway scene with 3 vehicles: a white sedan traveling in the left lane, a silver SUV in the center lane, and a blue pickup truck in the right lane. The road has 3 lanes separated by white dashed lines. Highway barriers are visible on both sides. The sky is partly cloudy."}
    ]
  },
  {
    "image": "bdd_001.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nWhat are the attributes of the vehicle on the left?"},
      {"from": "gpt", "value": "The vehicle on the left is a white four-door sedan, mid-size, positioned in the leftmost lane. It appears to be moving at highway speed. Its color is white and it has a standard sedan body shape."}
    ]
  },
  {
    "image": "bdd_001.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nDescribe the spatial arrangement of all vehicles in the scene."},
      {"from": "gpt", "value": "There are 3 vehicles arranged across the lanes. The white sedan is on the far left. The silver SUV is in the center, slightly ahead of the sedan. The blue pickup truck is on the far right, roughly parallel to the SUV. No vehicles are directly behind or in front of each other."}
    ]
  },
  {
    "image": "bdd_002.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nIs there anything unusual or out-of-distribution in this scene?"},
      {"from": "gpt", "value": "Yes, there is a stray dog on the right shoulder of the road, which is unusual for a highway environment. This could be classified as an out-of-distribution object. All other elements (vehicles, road markings, barriers) appear normal."}
    ]
  },
  {
    "image": "bdd_002.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nHow many vehicles are visible and what are their types?"},
      {"from": "gpt", "value": "There are 5 vehicles visible: 2 sedans (one red, one black), 1 white SUV, 1 delivery van, and 1 motorcycle. The motorcycle is in the left lane, and the delivery van is the largest vehicle in the right lane."}
    ]
  }
]
```

---

## Step 2: Run Fine-Tuning on Colab

### Cell 1 — Install Dependencies

```python
!pip install -q torch torchvision
!pip install -q transformers==4.36.2 accelerate bitsandbytes
!pip install -q peft==0.7.1
!pip install -q pillow sentencepiece protobuf
print("[OK] All dependencies installed")
```

### Cell 2 — Upload Your Dataset

```python
import os, json

# Create directories
os.makedirs("/content/finetune_data/images", exist_ok=True)

# Option A: Upload images manually
from google.colab import files
print("Upload your BDD images (select all 50 at once):")
uploaded = files.upload()
for fname in uploaded:
    os.rename(fname, f"/content/finetune_data/images/{fname}")

print(f"[OK] Uploaded {len(uploaded)} images")
```

### Cell 3 — Upload or Create train_data.json

```python
# Option A: Upload your pre-made JSON
from google.colab import files
print("Upload train_data.json:")
uploaded = files.upload()
os.rename("train_data.json", "/content/finetune_data/train_data.json")

# Option B: Auto-generate basic training data from images
# (Use this if you don't want to manually write Q&A pairs)
# Uncomment below to auto-generate:

# image_files = sorted(os.listdir("/content/finetune_data/images"))
# train_data = []
# for img_file in image_files:
#     # Type 1: Scene description
#     train_data.append({
#         "image": img_file,
#         "conversations": [
#             {"from": "human", "value": "<image>\nDescribe all the objects in this road scene."},
#             {"from": "gpt", "value": "FILL_THIS_IN"}  # ← You fill this manually
#         ]
#     })
#     # Type 2: Vehicle attributes
#     train_data.append({
#         "image": img_file,
#         "conversations": [
#             {"from": "human", "value": "<image>\nList all vehicles with their color, type, and position."},
#             {"from": "gpt", "value": "FILL_THIS_IN"}
#         ]
#     })
#     # Type 3: Spatial arrangement
#     train_data.append({
#         "image": img_file,
#         "conversations": [
#             {"from": "human", "value": "<image>\nDescribe the spatial arrangement of objects in this scene."},
#             {"from": "gpt", "value": "FILL_THIS_IN"}
#         ]
#     })
# with open("/content/finetune_data/train_data.json", "w") as f:
#     json.dump(train_data, f, indent=2)
# print(f"[OK] Generated {len(train_data)} training samples (fill in the answers!)")

# Verify
with open("/content/finetune_data/train_data.json") as f:
    data = json.load(f)
print(f"[OK] {len(data)} training samples loaded")
print(f"[OK] Sample: {data[0]['conversations'][0]['value'][:60]}...")
```

### Cell 4 — Load LLaVA Model (4-bit Quantized)

```python
import torch
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

model_id = "llava-hf/llava-1.5-7b-hf"

# 4-bit quantization for Colab T4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading LLaVA-7B (4-bit)... this takes ~3 minutes")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.pad_token = processor.tokenizer.eos_token

print(f"[OK] Model loaded on {model.device}")
print(f"[OK] VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")
```

### Cell 5 — Apply LoRA Adapters

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Prepare for training
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "v_proj",     # attention layers
        "k_proj", "o_proj",     # more attention layers
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Should show ~0.06% trainable params (about 4M out of 7B)

print("[OK] LoRA adapters added")
```

### Cell 6 — Create Training Dataset

```python
import json
from PIL import Image
from torch.utils.data import Dataset

class RoadSceneDataset(Dataset):
    def __init__(self, json_path, image_dir, processor):
        with open(json_path) as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item["image"])
        image = Image.open(image_path).convert("RGB")

        # Build conversation text
        human_msg = item["conversations"][0]["value"]
        gpt_msg = item["conversations"][1]["value"]

        # Format as LLaVA prompt
        prompt = f"USER: {human_msg}\nASSISTANT: {gpt_msg}"

        # Process with LLaVA processor
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True,
        )

        # Create labels (same as input_ids, model learns to predict next token)
        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()

        # Mask the prompt part so model only learns to generate answers
        # Find where ASSISTANT: starts
        assistant_token = self.processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)
        prompt_tokens = self.processor.tokenizer.encode(
            f"USER: {human_msg}\nASSISTANT:",
            add_special_tokens=True
        )
        # Set prompt tokens to -100 (ignored in loss)
        labels[:len(prompt_tokens)] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": labels,
        }


# Create dataset
dataset = RoadSceneDataset(
    json_path="/content/finetune_data/train_data.json",
    image_dir="/content/finetune_data/images",
    processor=processor,
)

print(f"[OK] Dataset created: {len(dataset)} samples")
print(f"[OK] Sample shape: {dataset[0]['input_ids'].shape}")
```

### Cell 7 — Train!

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="/content/lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=10,
    fp16=True,
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

print("Starting training...")
print(f"  Samples: {len(dataset)}")
print(f"  Epochs: 3")
print(f"  Batch size: 1 (× 4 accumulation = effective 4)")
print(f"  Estimated time: 30-60 minutes on T4\n")

trainer.train()

print("\n[OK] Training complete!")
```

### Cell 8 — Save LoRA Adapter

```python
# Save only the LoRA adapter weights (~50MB, not the full 7B model)
adapter_path = "/content/lora_road_adapter"
model.save_pretrained(adapter_path)
processor.save_pretrained(adapter_path)

print(f"[OK] Adapter saved to {adapter_path}")

# Check size
import subprocess
size = subprocess.check_output(["du", "-sh", adapter_path]).decode().split()[0]
print(f"[OK] Adapter size: {size} (vs 14GB full model)")

# Download to your computer
import shutil
shutil.make_archive("/content/lora_road_adapter", "zip", adapter_path)
from google.colab import files
files.download("/content/lora_road_adapter.zip")
print("[OK] Downloaded adapter zip")
```

### Cell 9 — Test Fine-Tuned Model

```python
# Load a test image
from PIL import Image

test_image = Image.open("/content/finetune_data/images/bdd_001.jpg").convert("RGB")
test_prompt = "USER: <image>\nDescribe all vehicles in this scene with their colors and positions.\nASSISTANT:"

inputs = processor(text=test_prompt, images=test_image, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

response = processor.tokenizer.decode(output[0], skip_special_tokens=True)
print("Fine-tuned model response:")
print(response.split("ASSISTANT:")[-1].strip())
```

---

## Step 3: Use Fine-Tuned Model in MAVR Pipeline

### Cell 10 — Copy Adapter to MAVR

```python
# Copy adapter into your project
!cp -r /content/lora_road_adapter /content/mavr-ood/weights/lora_road_adapter
print("[OK] Adapter copied to MAVR project")
```

### What to modify in MAVR (future integration)

In `src/agents/vlm_backend.py`, you would change the model loading to:

```python
# Before (base model):
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", ...)

# After (with LoRA adapter):
from peft import PeftModel
base_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", ...)
model = PeftModel.from_pretrained(base_model, "./weights/lora_road_adapter")
```

This loads the base model + applies your fine-tuned adapter on top.

---

## Summary

| Step | What | Time |
|------|------|------|
| 1 | Prepare 50 images + JSON Q&A pairs | 2-4 hours (manual) |
| 2 | Install & load model | 5 minutes |
| 3 | Apply LoRA | 1 minute |
| 4 | Train (3 epochs) | 30-60 minutes |
| 5 | Save adapter (~50MB) | 1 minute |
| 6 | Integrate into MAVR | 10 minutes |

**Total effort: ~4-5 hours (mostly data preparation)**
