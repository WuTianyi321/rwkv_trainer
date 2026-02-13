import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent


def test_train_help_without_rwkv_env_vars():
    """train.py should be import-safe even if RWKV_* env vars are absent."""
    env = os.environ.copy()
    for key in [
        "RWKV_MY_TESTING",
        "RWKV_JIT_ON",
        "RWKV_HEAD_SIZE_A",
        "RWKV_CTXLEN",
        "RWKV_TRAIN_TYPE",
    ]:
        env.pop(key, None)

    result = subprocess.run(
        [sys.executable, "train.py", "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_x070_optimizer_fallback_without_deepspeed():
    """
    x070 fallback should work without deepspeed by falling back to torch.optim.AdamW.
    """
    script = f"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, r"{ROOT / 'src'}")
os.environ["RWKV_MY_TESTING"] = "x070"
os.environ["RWKV_HEAD_SIZE_A"] = "64"
os.environ["RWKV_CTXLEN"] = "16"
os.environ["RWKV_TRAIN_TYPE"] = ""
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_TORCH_FALLBACK"] = "1"

from model.model import RWKV, FusedAdam, _HAS_DEEPSPEED

class Args:
    n_layer=2
    n_embd=64
    vocab_size=128
    ctx_len=16
    dim_att=64
    dim_ffn=224
    tiny_att_dim=-1
    tiny_att_layer=-1
    pre_ffn=0
    my_pos_emb=0
    head_size_a=64
    head_size_divisor=8
    dropout=0.0
    grad_cp=1
    head_qk=0
    my_testing='x070'
    train_type=''
    layerwise_lr=1
    my_pile_stage=0
    weight_decay=0.0
    lr_init=6e-4
    betas=(0.9, 0.99)
    adam_eps=1e-8

class TrainerStub:
    def __init__(self):
        self.strategy = object()
        self.is_global_zero = False

model = RWKV(Args())
stub = TrainerStub()
model.trainer = stub
opt = model.configure_optimizers()
assert _HAS_DEEPSPEED is False
assert FusedAdam is None
assert type(opt).__name__ == "AdamW"

idx = torch.randint(0, Args.vocab_size, (2, 16))
targets = torch.randint(0, Args.vocab_size, (2, 16))
loss = F.cross_entropy(model(idx).reshape(-1, Args.vocab_size), targets.reshape(-1))
loss.backward()
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
