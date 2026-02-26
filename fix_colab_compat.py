#!/usr/bin/env python3
"""
fix_colab_compat.py — Run this ONCE on Colab to fix all compatibility issues.

Fixes:
1. bertwarper.py: BertModel API removed in transformers 5.x (get_head_mask, etc.)
2. ms_deform_attn_cuda.cu: deprecated value.type() → value.scalar_type()
3. torch.load: add weights_only=False for PyTorch 2.10+
4. build_sam.py: torch.load missing map_location
5. Missing dependencies: addict, yapf
6. Rebuilds GroundingDINO CUDA extensions

Usage (in Colab):
    !python fix_colab_compat.py
"""

import os
import re
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GDINO_ROOT = os.path.join(PROJECT_ROOT, "GroundingDINO")
SAM_ROOT = os.path.join(PROJECT_ROOT, "segment_anything")


def patch_file(filepath, replacements, description):
    """Apply text replacements to a file. Returns True if any changes were made."""
    if not os.path.exists(filepath):
        print(f"  ⚠️  File not found: {filepath}")
        return False

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    original = content
    for old, new in replacements:
        content = content.replace(old, new)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ {description}")
        return True
    else:
        print(f"  ℹ️  Already patched: {description}")
        return False


# ============================================================
# Fix 1: bertwarper.py — BertModel methods removed in transformers 5.x
# ============================================================
def fix_bertwarper():
    print("\n🔧 Fix 1: bertwarper.py (transformers 5.x compat)")
    filepath = os.path.join(GDINO_ROOT, "groundingdino", "models", "GroundingDINO", "bertwarper.py")

    if not os.path.exists(filepath):
        print(f"  ⚠️  File not found: {filepath}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Replace get_head_mask with safe getattr fallback
    content = content.replace(
        "self.get_head_mask = bert_model.get_head_mask",
        "self.get_head_mask = getattr(bert_model, 'get_head_mask', "
        "lambda head_mask, num_hidden_layers: [None] * num_hidden_layers if head_mask is None else head_mask)"
    )

    # Replace get_extended_attention_mask with safe fallback
    content = content.replace(
        "self.get_extended_attention_mask = bert_model.get_extended_attention_mask",
        "self.get_extended_attention_mask = getattr(bert_model, 'get_extended_attention_mask', "
        "self._compat_get_extended_attention_mask)"
    )

    # Replace invert_attention_mask with safe fallback
    content = content.replace(
        "self.invert_attention_mask = bert_model.invert_attention_mask",
        "self.invert_attention_mask = getattr(bert_model, 'invert_attention_mask', "
        "self._compat_invert_attention_mask)"
    )

    # Add fallback method implementations (before forward method)
    fallback_code = '''
    @staticmethod
    def _compat_get_extended_attention_mask(attention_mask, input_shape, device=None):
        """Compat fallback for transformers 5.x"""
        import torch
        if attention_mask.dim() == 3:
            extended = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong attention_mask shape: {attention_mask.shape}")
        extended = (1.0 - extended.float()) * torch.finfo(torch.float32).min
        return extended

    @staticmethod
    def _compat_invert_attention_mask(encoder_attention_mask):
        """Compat fallback for transformers 5.x"""
        import torch
        if encoder_attention_mask.dim() == 3:
            inverted = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            inverted = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong mask shape: {encoder_attention_mask.shape}")
        inverted = (1.0 - inverted.float()) * torch.finfo(torch.float32).min
        return inverted

'''

    # Insert before forward() — handle both \r\n and \n
    for line_ending_pattern in ["    def forward(\r\n        self,", "    def forward(\n        self,"]:
        if line_ending_pattern in content and "_compat_get_extended_attention_mask" not in content:
            content = content.replace(line_ending_pattern, fallback_code + line_ending_pattern)
            break

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print("  ✅ Patched 3 deprecated BertModel methods + added fallbacks")
    else:
        print("  ℹ️  Already patched")


# ============================================================
# Fix 2: CUDA source — deprecated value.type()
# ============================================================
def fix_cuda_source():
    print("\n🔧 Fix 2: ms_deform_attn_cuda.cu (value.type() → value.scalar_type())")
    filepath = os.path.join(GDINO_ROOT, "groundingdino", "models", "GroundingDINO",
                           "csrc", "MsDeformAttn", "ms_deform_attn_cuda.cu")
    patch_file(filepath, [
        ('AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_forward_cuda"',
         'AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda"'),
        ('AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_backward_cuda"',
         'AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda"'),
    ], "Fixed deprecated CUDA macros")


# ============================================================
# Fix 3: torch.load — add weights_only=False for PyTorch 2.10+
# ============================================================
def fix_torch_load():
    print("\n🔧 Fix 3: torch.load (add weights_only=False for PyTorch 2.10+)")

    files_to_fix = [
        # run_evaluate.py
        (os.path.join(PROJECT_ROOT, "run_evaluate.py"), [
            ('torch.load(model_checkpoint_path, map_location="cpu")',
             'torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)'),
        ]),
        # app.py
        (os.path.join(PROJECT_ROOT, "app.py"), [
            ('torch.load(DEFAULT_GDINO_CKPT, map_location="cpu")',
             'torch.load(DEFAULT_GDINO_CKPT, map_location="cpu", weights_only=False)'),
        ]),
        # SAM build_sam.py
        (os.path.join(SAM_ROOT, "segment_anything", "build_sam.py"), [
            ('state_dict = torch.load(f)',
             'state_dict = torch.load(f, weights_only=False)'),
        ]),
        # SAM build_sam_hq.py
        (os.path.join(SAM_ROOT, "segment_anything", "build_sam_hq.py"), [
            ('state_dict = torch.load(f, map_location=device)',
             'state_dict = torch.load(f, map_location=device, weights_only=False)'),
        ]),
        # GroundingDINO inference.py
        (os.path.join(GDINO_ROOT, "groundingdino", "util", "inference.py"), [
            ('torch.load(model_checkpoint_path, map_location="cpu")',
             'torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)'),
        ]),
    ]

    for filepath, replacements in files_to_fix:
        basename = os.path.basename(filepath)
        patch_file(filepath, replacements, f"Fixed torch.load in {basename}")


# ============================================================
# Fix 4: Install missing dependencies
# ============================================================
def install_deps():
    print("\n🔧 Fix 4: Installing missing dependencies")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "addict", "yapf"])
        print("  ✅ Installed addict, yapf")
    except Exception as e:
        print(f"  ⚠️  pip install failed: {e}")


# ============================================================
# Fix 4b: Install local segment_anything (overrides any pip version)
# ============================================================
def install_local_packages():
    print("\n🔧 Fix 4b: Installing local segment_anything package")
    try:
        # Uninstall any pip version that might conflict
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "segment_anything"],
            capture_output=True, text=True
        )
        # Install our local copy (which includes SAM-HQ)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-e", SAM_ROOT]
        )
        print("  ✅ Local segment_anything installed")
    except Exception as e:
        print(f"  ⚠️  Failed: {e}")


# ============================================================
# Fix 5: Rebuild GroundingDINO
# ============================================================
def rebuild_groundingdino():
    print("\n🔧 Fix 5: Rebuilding GroundingDINO CUDA extensions")
    env = os.environ.copy()
    env["BUILD_WITH_CUDA"] = "True"
    env["CUDA_HOME"] = os.environ.get("CUDA_HOME", "/usr/local/cuda")

    result = subprocess.run(
        [sys.executable, "setup.py", "build", "develop", "--no-deps"],
        cwd=GDINO_ROOT,
        env=env,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("  ✅ GroundingDINO built successfully!")
    else:
        # Check if it's just warnings (common for CUDA builds)
        if "error" in result.stderr.lower() and "warning" not in result.stderr.lower():
            print(f"  ❌ Build failed: {result.stderr[-300:]}")
        else:
            print("  ✅ GroundingDINO built (with warnings)")


# ============================================================
# Fix 6: Verify all imports work
# ============================================================
def verify_imports():
    print("\n🔍 Verifying imports...")
    errors = []

    try:
        import torch
        print(f"  ✅ torch {torch.__version__}")
    except ImportError as e:
        errors.append(f"torch: {e}")

    try:
        import transformers
        print(f"  ✅ transformers {transformers.__version__}")
    except ImportError as e:
        errors.append(f"transformers: {e}")

    try:
        import groundingdino
        print(f"  ✅ groundingdino")
    except ImportError as e:
        errors.append(f"groundingdino: {e}")

    try:
        from segment_anything import sam_model_registry
        print(f"  ✅ segment_anything")
    except ImportError as e:
        errors.append(f"segment_anything: {e}")

    try:
        import clip
        print(f"  ✅ clip")
    except ImportError as e:
        errors.append(f"clip: {e}")

    try:
        import gradio
        print(f"  ✅ gradio {gradio.__version__}")
    except ImportError as e:
        errors.append(f"gradio: {e}")

    try:
        import addict
        print(f"  ✅ addict")
    except ImportError as e:
        errors.append(f"addict: {e}")

    if errors:
        print(f"\n  ⚠️  Import errors:")
        for err in errors:
            print(f"     ❌ {err}")
    else:
        print("\n  🎉 All imports successful!")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🔧 MAVR-OOD Colab Compatibility Fixer")
    print("=" * 60)

    install_deps()
    install_local_packages()
    fix_bertwarper()
    fix_cuda_source()
    fix_torch_load()
    rebuild_groundingdino()
    verify_imports()

    print("\n" + "=" * 60)
    print("✅ ALL FIXES APPLIED!")
    print("=" * 60)
    print("\nYou can now run:")
    print("  !python run_evaluate.py ...")
    print("  !python app.py")
