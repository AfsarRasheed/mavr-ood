# Running MAVR Streamlit App on Google Colab

Two versions available:
- **Original (main)** — Rule-based query parser (left/right/center keywords)
- **Improved (llava-parser)** — LLaVA-based parser (any natural language query)

---

## Common Setup (Run Once)

### Cell 1 — Clone Repository

```python
!git clone https://github.com/AfsarRasheed/mavr-ood.git
%cd /content/mavr-ood
```

### Cell 2 — Install Dependencies

```python
!pip install -q gradio torch torchvision transformers accelerate bitsandbytes
!pip install -q streamlit pyngrok addict yapf
```

### Cell 3 — Build GroundingDINO & SAM

```python
!cd GroundingDINO && pip install -e . -q
!cd segment_anything && pip install -e . -q
!pip install git+https://github.com/openai/CLIP.git -q
```

### Cell 4 — Verify GPU & Dependencies

```python
import torch, bitsandbytes
print(f"[OK] torch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
print(f"[OK] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("[OK] All dependencies installed!")
```

### Cell 5 — Download Model Weights

```python
import os
if not os.path.exists("weights/groundingdino_swint_ogc.pth"):
    !mkdir -p weights
    !wget -q -P weights/ https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    !wget -q -P weights/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    print("[OK] Weights downloaded!")
else:
    print("[OK] Weights already exist!")
```

---

## Option A: Run Original Version (main branch)

### Cell 6A — Switch to Original

```python
%cd /content/mavr-ood
!git checkout main
print("[OK] On ORIGINAL version (rule-based parser)")
print("Supported queries: left, right, center, largest, nearest, next to, behind")
```

### Cell 7A — Launch Streamlit

```python
# Set your ngrok auth token (get free one from https://dashboard.ngrok.com/signup)
!ngrok authtoken YOUR_NGROK_TOKEN_HERE    # ← Replace with your token

!nohup streamlit run streamlit_app.py --server.port 8501 --server.headless true &

import time
time.sleep(8)

from pyngrok import ngrok
ngrok.kill()
url = ngrok.connect(8501)
print(f"\n🚀 Open this URL in your browser:\n{url}\n")
print("Example queries:")
print('  • "the red car on the left"')
print('  • "the white truck on the right"')
print('  • "the largest vehicle"')
print('  • "the car next to the truck"')
```

---

## Option B: Run Improved Version (LLaVA parser branch)

### Cell 6B — Switch to Improved

```python
%cd /content/mavr-ood
!git fetch origin
!git checkout improvement/llava-parser
print("[OK] On IMPROVED version (LLaVA-based parser)")
print("Supported queries: ANY natural language description!")
```

### Cell 7B — Launch Streamlit

```python
# Set your ngrok auth token (get free one from https://dashboard.ngrok.com/signup)
!ngrok authtoken YOUR_NGROK_TOKEN_HERE    # ← Replace with your token

!nohup streamlit run streamlit_app.py --server.port 8501 --server.headless true &

import time
time.sleep(8)

from pyngrok import ngrok
ngrok.kill()
url = ngrok.connect(8501)
print(f"\n🚀 Open this URL in your browser:\n{url}\n")
print("Try these ADVANCED queries:")
print('  • "the car parked between the truck and the bus"')
print('  • "the second vehicle from the right"')
print('  • "the damaged car near the traffic signal"')
print('  • "the red car on the left"  (basic queries also work)')
```

---

## Switching Between Versions

> **Important:** After switching branches, you MUST restart the runtime.

### Switch from Original → Improved

```python
# Kill existing Streamlit
!pkill -f streamlit
from pyngrok import ngrok
ngrok.kill()

# Switch branch
!git fetch origin
!git checkout improvement/llava-parser
print("[OK] Switched to IMPROVED version")
print("Now restart runtime (Runtime → Restart runtime) and run Cell 7B")
```

### Switch from Improved → Original

```python
# Kill existing Streamlit
!pkill -f streamlit
from pyngrok import ngrok
ngrok.kill()

# Switch branch
!git checkout main
print("[OK] Switched to ORIGINAL version")
print("Now restart runtime (Runtime → Restart runtime) and run Cell 7A")
```

---

## What's Different Between Versions?

| Feature | Original (main) | Improved (llava-parser) |
|---------|-----------------|------------------------|
| Query parser | Rule-based keywords | LLaVA understands any query |
| "left/right/center" | ✅ | ✅ |
| "between X and Y" | ❌ | ✅ |
| "second from right" | ❌ | ✅ |
| "damaged car near signal" | ❌ | ✅ |
| "the vehicle ahead" | ❌ | ✅ |
| Fallback if parser fails | N/A | Falls back to rule-based |
| Speed | Faster (no extra LLaVA call) | ~3-5s slower (extra LLaVA parse) |
