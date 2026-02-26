#!/usr/bin/env python3
"""
fix_colab_compat.py — Run this ONCE on Colab to fix all compatibility issues.

Fixes:
1. GroundingDINO bertwarper.py: BertModel API changes in transformers 5.x
2. GroundingDINO CUDA source: deprecated value.type() → value.scalar_type()
3. Missing dependencies: addict, yapf

Usage (in Colab):
    !python fix_colab_compat.py
"""

import os
import subprocess
import sys

GDINO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GroundingDINO")

def fix_bertwarper():
    """Fix BertModel API compatibility with transformers 5.x"""
    filepath = os.path.join(GDINO_ROOT, "groundingdino", "models", "GroundingDINO", "bertwarper.py")
    
    if not os.path.exists(filepath):
        print(f"⚠️ File not found: {filepath}")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Fix 1: get_head_mask — provide inline fallback
    content = content.replace(
        "self.get_head_mask = bert_model.get_head_mask",
        "self.get_head_mask = getattr(bert_model, 'get_head_mask', lambda head_mask, num_hidden_layers: [None] * num_hidden_layers if head_mask is None else head_mask)"
    )
    
    # Fix 2: get_extended_attention_mask — provide inline fallback
    old_extended = "self.get_extended_attention_mask = bert_model.get_extended_attention_mask"
    new_extended = """self.get_extended_attention_mask = getattr(bert_model, 'get_extended_attention_mask', self._fallback_get_extended_attention_mask)"""
    content = content.replace(old_extended, new_extended)
    
    # Fix 3: invert_attention_mask — provide inline fallback
    old_invert = "self.invert_attention_mask = bert_model.invert_attention_mask"
    new_invert = """self.invert_attention_mask = getattr(bert_model, 'invert_attention_mask', self._fallback_invert_attention_mask)"""
    content = content.replace(old_invert, new_invert)
    
    # Add fallback methods after __init__ 
    fallback_methods = '''
    @staticmethod
    def _fallback_get_extended_attention_mask(attention_mask, input_shape, device=None):
        """Fallback for transformers 5.x where get_extended_attention_mask was removed."""
        if attention_mask.dim() == 3:
            extended = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")
        extended = (1.0 - extended.float()) * torch.finfo(torch.float32).min
        return extended

    @staticmethod
    def _fallback_invert_attention_mask(encoder_attention_mask):
        """Fallback for transformers 5.x where invert_attention_mask was removed."""
        if encoder_attention_mask.dim() == 3:
            inverted = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            inverted = encoder_attention_mask[:, None, None, :]
        inverted = (1.0 - inverted.float()) * torch.finfo(torch.float32).min
        return inverted
'''
    
    # Insert fallback methods before the forward method
    content = content.replace(
        "    def forward(\n        self,",
        fallback_methods + "\n    def forward(\n        self,"
    )
    # Also handle \r\n line endings
    content = content.replace(
        "    def forward(\r\n        self,",
        fallback_methods + "\n    def forward(\r\n        self,"
    )
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print("✅ Fixed bertwarper.py (3 deprecated BertModel methods)")
        return True
    else:
        print("ℹ️ bertwarper.py already patched or no changes needed")
        return False


def fix_cuda_source():
    """Fix deprecated value.type() → value.scalar_type() in CUDA source."""
    filepath = os.path.join(GDINO_ROOT, "groundingdino", "models", "GroundingDINO", 
                           "csrc", "MsDeformAttn", "ms_deform_attn_cuda.cu")
    
    if not os.path.exists(filepath):
        print(f"⚠️ CUDA source not found: {filepath}")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    content = content.replace(
        'AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_forward_cuda"',
        'AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda"'
    )
    content = content.replace(
        'AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_backward_cuda"',
        'AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda"'
    )
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print("✅ Fixed ms_deform_attn_cuda.cu (value.type() → value.scalar_type())")
        return True
    else:
        print("ℹ️ CUDA source already patched")
        return False


def install_deps():
    """Install missing GroundingDINO dependencies."""
    print("📦 Installing missing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "addict", "yapf"])
    print("✅ Installed addict, yapf")


def build_groundingdino():
    """Build GroundingDINO with CUDA extensions."""
    print("🔨 Building GroundingDINO...")
    env = os.environ.copy()
    env["BUILD_WITH_CUDA"] = "True"
    env["CUDA_HOME"] = "/usr/local/cuda"
    
    result = subprocess.run(
        [sys.executable, "setup.py", "build", "develop", "--no-deps"],
        cwd=GDINO_ROOT,
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ GroundingDINO built successfully!")
        return True
    else:
        print(f"⚠️ Build had issues: {result.stderr[-500:] if result.stderr else 'unknown'}")
        return False


if __name__ == "__main__":
    print("🔧 MAVR-OOD Colab Compatibility Fixer")
    print("=" * 50)
    
    install_deps()
    fix_bertwarper()
    fix_cuda_source()
    build_groundingdino()
    
    print("\n" + "=" * 50)
    print("✅ All fixes applied! You can now run:")
    print("   !python run_evaluate.py ...")
