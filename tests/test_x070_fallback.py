import subprocess
import sys
from pathlib import Path


def test_x070_torch_fallback_forward_backward():
    """x070 should run without CUDA build when torch fallback is forced."""
    src_dir = Path(__file__).parent.parent / "src"
    script = f"""
import os
import sys
import torch

sys.path.insert(0, r"{src_dir}")
os.environ["RWKV_MY_TESTING"] = "x070"
os.environ["RWKV_HEAD_SIZE_A"] = "64"
os.environ["RWKV_CTXLEN"] = "64"
os.environ["RWKV_TRAIN_TYPE"] = ""
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_TORCH_FALLBACK"] = "1"

from model import model as m

assert getattr(m, "_X070_USE_TORCH_FALLBACK", False) is True
assert m.MyModule is not torch.jit.ScriptModule

B, T, HC = 2, 3, 64
inputs = [torch.randn(B, T, HC, dtype=torch.float32, requires_grad=True) for _ in range(6)]
y = m.RUN_CUDA_RWKV7g(*inputs)
assert y.shape == (B, T, HC)
assert torch.isfinite(y).all()

loss = y.sum()
loss.backward()
for x in inputs:
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
