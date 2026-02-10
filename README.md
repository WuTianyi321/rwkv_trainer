# RWKV Trainer

A packaged, general-purpose RWKV training framework for sequence data.

## Features

- **Universal Data Support**: Works with any integer sequence data (not just angles)
- **Flexible Tokenization**: Custom vocabularies, custom tokenizers
- **Multiple Input Formats**: Numpy arrays, JSONL files
- **External Checkpoint Resume**: Auto-detect architecture from any RWKV-LM checkpoint
- **One-line interface**: Simple Python API for complete training pipeline
- **Self-contained**: All dependencies packaged, no external RWKV-LM repository needed
- **Work directory based**: All outputs organized in a single working directory

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/WuTianyi321/rwkv_trainer.git
cd rwkv_trainer
uv venv
uv pip install -e ".[dev]"
```

## Quick Start

### Example 1: Integer Sequence Data (0-999)

```python
from rwkv_trainer import RWKVTrainingPipeline, ModelConfig, IntegerTokenizer
import numpy as np

# Create custom tokenizer for values 0-999
tokenizer = IntegerTokenizer(max_value=999)

# Create data
data = np.random.randint(0, 1000, size=(1000, 1024))

# Train
pipeline = RWKVTrainingPipeline(
    work_dir="./experiment",
    model_config=ModelConfig(n_layer=3, n_embd=128, vocab_size=1001),
    tokenizer=tokenizer
)
pipeline.train(data, num_epochs=100)
```

### Example 2: From JSONL File

```python
from rwkv_trainer import RWKVTrainingPipeline

pipeline = RWKVTrainingPipeline(work_dir="./experiment")

# Prepare from existing JSONL (each line: {"text": "1 2 3 ..."})
pipeline.prepare_data_from_jsonl("my_data.jsonl")

# Train
pipeline.train(num_epochs=100)
```

### Example 3: Custom Vocabulary

```python
from rwkv_trainer import (
    RWKVTrainingPipeline, 
    GenericTokenizer,
    create_vocab_file_from_tokens
)

# Create custom vocab
tokens = ['hello', 'world', 'foo', 'bar', ' ']
create_vocab_file_from_tokens(tokens, "custom_vocab.txt")

# Use custom tokenizer
tokenizer = GenericTokenizer("custom_vocab.txt")
pipeline = RWKVTrainingPipeline(
    work_dir="./experiment",
    tokenizer=tokenizer
)
```

### Example 4: Resume from External Checkpoint

```python
from rwkv_trainer import RWKVTrainingPipeline

pipeline = RWKVTrainingPipeline(work_dir="./experiment")

# Prepare your data
pipeline.prepare_data_from_jsonl("your_data.jsonl")

# Resume from any RWKV-LM checkpoint (auto-detect architecture)
pipeline.train_from_checkpoint(
    checkpoint_path="/path/to/pretrained_model.pth",
    num_epochs=50  # Fine-tune for 50 more epochs
)
```

---

## 📊 Complete Data Flow

Here is the complete pipeline from input data to trained model:

### Input → Processing → Output

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT OPTIONS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Option 1: Numpy Array                                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ data = np.random.randint(0, 1000, size=(1000, 1024))                  │ │
│  │ Shape: (n_sequences, sequence_length)                                  │ │
│  │ Values: Integers in range [0, vocab_size-1]                           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                               │
│  pipeline.prepare_data(data) → NumpyToJsonlConverter                        │
│                              ↓                                               │
│  Generates: data/train.jsonl  (each line: {"text": "1 2 3 ..."})            │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  Option 2: JSONL File                                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ File: my_data.jsonl                                                    │ │
│  │ Format: One JSON per line                                              │ │
│  │   {"text": "value1 value2 value3 ..."}                                 │ │
│  │   {"text": "10 20 30 40 ..."}                                          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                               │
│  pipeline.prepare_data_from_jsonl("my_data.jsonl")                          │
│                              ↓                                               │
│  Copies to: data/train.jsonl                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TOKENIZATION & CONVERSION                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  JsonlToBinIdxConverter                                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 1. Load JSONL lines                                                    │ │
│  │ 2. Shuffle and duplicate (n_epochs times)                             │ │
│  │ 3. Tokenize each line using configured tokenizer                      │ │
│  │    - IntegerTokenizer: "10 20 30" → [11, 21, 31] (+1 offset)         │ │
│  │    - GenericTokenizer: Uses TRIE for subword tokenization            │ │
│  │ 4. Append end_of_doc token (0)                                        │ │
│  │ 5. Write to binary format                                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                               │
│  Generates:                                                                  │
│  ├── data/train.bin   (raw token IDs, uint16, memory-mapped)               │
│  ├── data/train.idx   (index for random access)                            │
│  └── data/vocab.txt   (tokenizer vocabulary)                               │
│                                                                              │
│  Computes:                                                                   │
│  ├── total_tokens: Total number of tokens in dataset                       │
│  └── magic_prime: For RWKV's sampling strategy                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODEL INITIALIZATION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Stage 1: Initialize Weights (CPU)                                           │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ python train.py --my_pile_stage 1 ...                                  │ │
│  │                                                                        │ │
│  │ Creates: out/rwkv-init.pth                                             │ │
│  │                                                                        │ │
│  │ Weight initialization:                                                 │ │
│  │ - emb.weight: uniform_(-1e-4, 1e-4)                                   │ │
│  │ - head.weight: orthogonal_ initialization                             │ │
│  │ - ln_x.weight: layer-wise scaling                                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Stage 3: Train (GPU)                                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ python train.py --my_pile_stage 3 ...                                  │ │
│  │                                                                        │ │
│  │ Data Loading:                                                          │ │
│  │ - Memory-mapped access to train.bin                                   │ │
│  │ - Random sampling using magic_prime strategy                          │ │
│  │ - Batch size: micro_bsz (default: 16)                                 │ │
│  │                                                                        │ │
│  │ Training Loop:                                                         │ │
│  │ - Optimizer: Adam with DeepSpeed ZeRO-1/2                             │ │
│  │ - Learning rate: lr_init → lr_final (cosine schedule)                 │ │
│  │ - Gradient checkpointing: Save VRAM, slower speed                     │ │
│  │                                                                        │ │
│  │ Output Checkpoints:                                                    │ │
│  │ - out/rwkv-init.pth   (initial weights)                               │ │
│  │ - out/rwkv-0.pth      (after epoch 0)                                 │ │
│  │ - out/rwkv-1.pth      (after epoch 1)                                 │ │
│  │ - out/rwkv-*.pth      (every epoch_save epochs)                       │ │
│  │ - out/rwkv-final.pth  (when training completes)                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Work Directory Structure (After Training)

```
work_dir/
├── configs/
│   └── config.json              # Saved configuration
│
├── data/
│   ├── train.jsonl              # Text data (JSON lines)
│   ├── train.bin                # Binary token data (memory-mapped)
│   ├── train.idx                # Index for random access
│   └── vocab.txt                # Tokenizer vocabulary
│
└── out/
    ├── rwkv-init.pth            # Initial model weights
    ├── rwkv-0.pth               # Checkpoint epoch 0
    ├── rwkv-1.pth               # Checkpoint epoch 1
    ├── ...
    ├── rwkv-final.pth           # Final model
    └── train_log.txt            # Training log
```

---

## Tokenizers

### IntegerTokenizer

For integer sequence data (0 to max_value):

```python
from rwkv_trainer import IntegerTokenizer

# Values 0-999 mapped to tokens 1-1000, token 0 = end_of_doc
tokenizer = IntegerTokenizer(max_value=999)
tokens = tokenizer.encode_sequence([0, 100, 500, 999])  # [1, 101, 501, 1000]
values = tokenizer.decode_sequence(tokens)               # [0, 100, 500, 999]
```

### GenericTokenizer

For custom vocabularies:

```python
from rwkv_trainer import GenericTokenizer, create_vocab_file_from_tokens

# Create vocab file
tokens = ['hello', 'world', ' ', '!']
create_vocab_file_from_tokens(tokens, "vocab.txt")

# Load tokenizer
tokenizer = GenericTokenizer("vocab.txt")
tokens = tokenizer.encode("hello world!")  # [1, 2, 3, 4]
```

### AngleTokenizer (Specialized)

For angle data 0-359 degrees (backward compatibility):

```python
from rwkv_trainer import AngleTokenizer

tokenizer = AngleTokenizer()
tokens = tokenizer.encode_angle_sequence([0, 45, 90])  # [1, 46, 91]
```

---

## Configuration

### ModelConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | "x060" | RWKV version: "x052", "x060", "x070" |
| `n_layer` | 3 | Number of transformer layers |
| `n_embd` | 128 | Embedding dimension |
| `ctx_len` | 1024 | Context length |
| `vocab_size` | 361 | Vocabulary size (auto-set from tokenizer) |

### TrainingConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr_init` | 6e-4 | Initial learning rate |
| `lr_final` | 6e-5 | Final learning rate |
| `micro_bsz` | 16 | Batch size per GPU |
| `grad_cp` | 1 | Gradient checkpointing |
| `precision` | "bf16" | "bf16", "fp16", or "fp32" |

---

## Advanced Usage

### Resume from External Checkpoint

You can resume training from any RWKV-LM checkpoint (any path, any filename). The pipeline will **auto-detect** model architecture from the checkpoint.

#### Example 1: Auto-detect Everything

```python
from rwkv_trainer import RWKVTrainingPipeline

pipeline = RWKVTrainingPipeline(work_dir="./my_experiment")

# Prepare your data
pipeline.prepare_data_from_jsonl("your_data.jsonl")

# Resume from external checkpoint (auto-detect n_layer, n_embd, vocab_size, model_type)
pipeline.train_from_checkpoint(
    checkpoint_path="/path/to/any/rwkv-model.pth",
    num_epochs=100
)
```

**Auto-detected parameters:**
- `n_layer`: Count transformer blocks
- `n_embd`: From embedding weight shape
- `vocab_size`: From embedding or head weight shape
- `model_type`: Infer from key patterns (x052/x060/x070)

#### Example 2: Override Specific Parameters

```python
# Auto-detect but override ctx_len (cannot be detected from weights)
pipeline.train_from_checkpoint(
    checkpoint_path="/path/to/model.pth",
    num_epochs=100,
    override_config={
        'ctx_len': 2048,      # Override context length
        'lr_init': 1e-4,      # Override learning rate
    }
)
```

#### Example 3: Inspect Checkpoint Before Training

```python
# Check what the pipeline will detect
info = pipeline.inspect_checkpoint("/path/to/model.pth")

print(f"File size: {info['file_size_mb']:.1f} MB")
print(f"Parameters: {info['num_parameters']:,}")
print(f"Detected config: {info['detected_config']}")
# Output: {'n_layer': 12, 'n_embd': 768, 'vocab_size': 65536, 'model_type': 'x060'}
```

#### Important Notes

1. **Vocab size must match**: Your data/tokenizer vocab size must match the checkpoint's vocab size
2. **Checkpoint copied**: External checkpoint is copied to `work_dir/out/rwkv-init.pth`
3. **Auto-save config**: Detected/overridden config is saved to `work_dir/configs/config.json`

### Continue from Pipeline's Own Checkpoint

```python
# Continue training from work_dir's latest checkpoint
pipeline = RWKVTrainingPipeline(work_dir="./existing_experiment")
pipeline.train(num_epochs=200, continue_training=True)
```

### Step-by-Step Pipeline

```python
# Initialize
pipeline = RWKVTrainingPipeline(work_dir="./experiment")

# Step 1: Prepare data
pipeline.prepare_data(data, "train")

# Step 2: Initialize model (Stage 1)
pipeline.initialize_model()

# Step 3: Train (Stage 3)
pipeline.train(num_epochs=100)
```

---

## Testing

```bash
# Run all tests
uv run ./run_tests.sh

# Or individually
uv run python tests/test_tokenizer_simple.py
uv run python tests/test_data_converter_simple.py
uv run python tests/test_pipeline_simple.py
```

---

## License

Same as RWKV-LM
