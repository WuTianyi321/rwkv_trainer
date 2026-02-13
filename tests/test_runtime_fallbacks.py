import os
import json
import subprocess
import sys
import tempfile
import pickle
import gc
from pathlib import Path

import numpy as np


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


def test_process_jsonl_magic_prime_uses_sequence_length():
    """JSONL path should compute magic_prime with the actual ctx length, not hardcoded 1024."""
    sys.path.insert(0, str(ROOT / "src"))
    from data_utils.converter import DataPipeline, JsonlToBinIdxConverter

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        jsonl_path = td_path / "data.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for _ in range(200):
                arr = np.random.randint(0, 100, size=16)
                f.write(json.dumps({"text": " ".join(map(str, arr.tolist()))}) + "\n")

        pipeline = DataPipeline()
        result = pipeline.process_jsonl(
            jsonl_path=jsonl_path,
            output_dir=td_path,
            name="train",
            n_epochs=1,
            sequence_length=16,
        )

        converter = JsonlToBinIdxConverter(pipeline.tokenizer)
        expected = converter.compute_magic_prime(result["binidx_prefix"], 16)
        assert result["magic_prime"] == expected


def test_mmapindexed_dataset_pickle_roundtrip_on_windows_spawn_path():
    """MMapIndexedDataset should be pickle-restorable (used by DataLoader workers on Windows)."""
    sys.path.insert(0, str(ROOT / "src"))
    from data_utils.converter import DataPipeline
    from data_utils.binidx import MMapIndexedDataset

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        data = np.random.randint(0, 50, size=(8, 16), dtype=np.int64)
        result = DataPipeline().process_numpy(
            data=data,
            output_dir=td_path,
            name="train",
            sequence_length=16,
            n_epochs=1,
        )
        ds = MMapIndexedDataset(str(result["binidx_prefix"]))
        blob = pickle.dumps(ds)
        restored = pickle.loads(blob)
        assert len(restored) == len(ds)
        sample = restored[0]
        assert sample is not None
        restored.close()
        ds.close()
        del sample
        del restored
        del ds
        gc.collect()
