"""
Utilities for loading and inspecting RWKV model checkpoints
Supports auto-detection of model architecture from .pth files
"""

import re
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DetectedModelConfig:
    """Model configuration detected from checkpoint"""
    n_layer: int
    n_embd: int
    vocab_size: int
    model_type: str = "x060"  # 默认假设
    ctx_len: int = 1024       # 无法从权重检测，使用默认值
    head_size_a: int = 64     # 默认
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_layer': self.n_layer,
            'n_embd': self.n_embd,
            'vocab_size': self.vocab_size,
            'model_type': self.model_type,
            'ctx_len': self.ctx_len,
            'head_size_a': self.head_size_a,
        }


def inspect_rwkv_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Inspect RWKV checkpoint and extract model architecture
    
    Args:
        checkpoint_path: Path to .pth file
        
    Returns:
        Dictionary with model info
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint (map to CPU to avoid GPU memory usage)
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    # Get all keys
    keys = list(state_dict.keys())
    
    # Analyze structure
    info = {
        'checkpoint_path': str(checkpoint_path),
        'file_size_mb': checkpoint_path.stat().st_size / (1024 * 1024),
        'num_parameters': sum(p.numel() for p in state_dict.values()),
        'keys_sample': keys[:10],
    }
    
    # Detect architecture
    try:
        config = detect_architecture(state_dict)
        info['detected_config'] = config.to_dict()
    except Exception as e:
        info['detection_error'] = str(e)
    
    return info


def detect_architecture(state_dict: Dict[str, torch.Tensor]) -> DetectedModelConfig:
    """
    Detect model architecture from state dict
    
    Detection strategy:
    1. n_layer: Count unique block indices (blocks.0, blocks.1, ...)
    2. n_embd: Get embedding dimension from emb.weight or blocks.0.ln1.weight
    3. vocab_size: Get from emb.weight shape[0] or head.weight shape[0]
    4. model_type: Infer from key patterns (x052, x060, x070 have different keys)
    """
    
    keys = list(state_dict.keys())
    
    # 1. Detect n_layer by counting blocks
    block_indices = set()
    for key in keys:
        match = re.match(r'blocks\.(\d+)\.', key)
        if match:
            block_indices.add(int(match.group(1)))
    
    if block_indices:
        n_layer = max(block_indices) + 1
    else:
        raise ValueError("Cannot detect n_layer: no 'blocks.N.' keys found")
    
    # 2. Detect n_embd
    n_embd = None
    
    # Try emb.weight first
    if 'emb.weight' in state_dict:
        n_embd = state_dict['emb.weight'].shape[1]
    # Try first block's ln0 or ln1
    elif 'blocks.0.ln0.weight' in state_dict:
        n_embd = state_dict['blocks.0.ln0.weight'].shape[0]
    elif 'blocks.0.ln1.weight' in state_dict:
        n_embd = state_dict['blocks.0.ln1.weight'].shape[0]
    
    if n_embd is None:
        raise ValueError("Cannot detect n_embd: no emb.weight or block LN weights found")
    
    # 3. Detect vocab_size
    vocab_size = None
    
    # Try emb.weight
    if 'emb.weight' in state_dict:
        vocab_size = state_dict['emb.weight'].shape[0]
    # Try head.weight
    elif 'head.weight' in state_dict:
        vocab_size = state_dict['head.weight'].shape[0]
    
    if vocab_size is None:
        raise ValueError("Cannot detect vocab_size: no emb.weight or head.weight found")
    
    # 4. Infer model_type from key patterns
    model_type = infer_model_type(keys)
    
    # 5. Try to detect head_size_a (optional)
    head_size_a = 64  # default
    for key in keys:
        if 'time_first' in key or 'time_decay' in key:
            # In RWKV, these have shape [n_embd] or [H, N]
            tensor = state_dict[key]
            if len(tensor.shape) >= 2:
                # If 2D, second dim might be head_size
                pass
            break
    
    return DetectedModelConfig(
        n_layer=n_layer,
        n_embd=n_embd,
        vocab_size=vocab_size,
        model_type=model_type,
        head_size_a=head_size_a
    )


def infer_model_type(keys: list) -> str:
    """
    Infer RWKV model type from key patterns
    
    x052 (RWKV-5): time_first, time_decay
    x060 (RWKV-6): time_faaaa, time_decay
    x070 (RWKV-7): Different key patterns
    """
    key_set = set(keys)
    
    # Check for x070 patterns (RWKV-7)
    # RWKV-7 has different attention mechanism
    if any('lora_A' in k or 'lora_B' in k for k in keys):
        # Might be a LoRA model, check further
        pass
    
    # Check for x060 specific keys
    # RWKV-6 uses time_faaaa instead of time_first
    has_time_faaaa = any('time_faaaa' in k for k in keys)
    has_time_first = any('time_first' in k for k in keys)
    
    if has_time_faaaa:
        return "x060"
    elif has_time_first:
        # Could be x052 or x060 with compatibility
        # Check for other x060 specific patterns
        if any(' receptance' in k or '.receptance.' in k for k in keys):
            return "x060"
        return "x052"
    
    # Default to x060 (most common)
    return "x060"


def load_checkpoint_for_training(
    checkpoint_path: str,
    override_config: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, torch.Tensor], DetectedModelConfig]:
    """
    Load checkpoint and return state dict + configuration
    
    Args:
        checkpoint_path: Path to .pth file
        override_config: User-specified config to override detected values
        
    Returns:
        Tuple of (state_dict, config)
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    detected = detect_architecture(state_dict)
    
    # Apply overrides
    if override_config:
        for key, value in override_config.items():
            if hasattr(detected, key):
                setattr(detected, key, value)
                print(f"  Override: {key} = {value}")
    
    return state_dict, detected


def validate_checkpoint_compatibility(
    checkpoint_config: DetectedModelConfig,
    pipeline_config: Dict[str, Any]
) -> Tuple[bool, list]:
    """
    Validate if checkpoint is compatible with pipeline configuration
    
    Returns:
        Tuple of (is_compatible, warnings)
    """
    warnings = []
    
    # Check vocab_size (must match exactly)
    if checkpoint_config.vocab_size != pipeline_config.get('vocab_size'):
        warnings.append(
            f"Vocab size mismatch: "
            f"checkpoint={checkpoint_config.vocab_size}, "
            f"pipeline={pipeline_config.get('vocab_size')}"
        )
        return False, warnings
    
    # Check n_embd (should match, but could potentially be adapted)
    if checkpoint_config.n_embd != pipeline_config.get('n_embd'):
        warnings.append(
            f"Embedding dim mismatch: "
            f"checkpoint={checkpoint_config.n_embd}, "
            f"pipeline={pipeline_config.get('n_embd')}"
        )
    
    # Check n_layer (should match)
    if checkpoint_config.n_layer != pipeline_config.get('n_layer'):
        warnings.append(
            f"Layer count mismatch: "
            f"checkpoint={checkpoint_config.n_layer}, "
            f"pipeline={pipeline_config.get('n_layer')}"
        )
        return False, warnings
    
    return len(warnings) == 0, warnings
