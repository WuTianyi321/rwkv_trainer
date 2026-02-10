# RWKV Trainer for Vicsek Model

A packaged RWKV training framework for collective motion (Vicsek model) angle sequence data.

## Features

- **One-line interface**: Simple Python API for complete training pipeline
- **Self-contained**: All dependencies packaged, no external RWKV-LM repository needed
- **Data pipeline**: Automatic conversion from numpy arrays → JSONL → binary indexed format
- **Configurable**: Easy configuration of model architecture and training hyperparameters
- **Work directory based**: All outputs organized in a single working directory

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from trainer.pipeline import RWKVTrainingPipeline, ModelConfig, TrainingConfig
import numpy as np

# Create or load your angle data (0-359 degrees)
data = np.random.randint(0, 360, size=(1000, 1024))

# Initialize pipeline
pipeline = RWKVTrainingPipeline(
    work_dir="./my_experiment",
    model_config=ModelConfig(n_layer=3, n_embd=128),
    training_config=TrainingConfig(lr_init=6e-4, lr_final=6e-5)
)

# Train (automatically prepares data, initializes model, and trains)
pipeline.train(data, num_epochs=100)
```

## Project Structure

```
rwkv_trainer/
├── src/
│   ├── data_utils/          # Data conversion and tokenization
│   │   ├── tokenizer.py     # Angle tokenizer (0-359 degrees)
│   │   ├── converter.py     # numpy → jsonl → bin/idx
│   │   └── binidx.py        # Memory-mapped dataset
│   ├── model/               # RWKV model implementation
│   │   └── rwkv_model.py    # RWKV architecture
│   ├── trainer/             # Training logic
│   │   ├── pipeline.py      # Main training pipeline
│   │   ├── trainer_module.py # Training callbacks
│   │   └── dataset.py       # Dataset loader
│   └── cuda/                # CUDA kernels for RWKV
├── tests/                   # Unit tests
├── examples/                # Example scripts
├── train.py                 # Entry point script
└── README.md

# After training, your work_dir will contain:
my_experiment/
├── data/                    # Processed data
│   ├── train.jsonl
│   ├── train.bin
│   ├── train.idx
│   └── vocab.txt
├── out/                     # Model checkpoints
│   ├── rwkv-init.pth
│   ├── rwkv-0.pth
│   ├── rwkv-1.pth
│   └── ...
└── configs/                 # Configuration files
    └── config.json
```

## Configuration

### ModelConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | "x060" | RWKV version: "x052", "x060", "x070" |
| `n_layer` | 3 | Number of transformer layers |
| `n_embd` | 128 | Embedding dimension |
| `ctx_len` | 1024 | Context length |
| `vocab_size` | 361 | Vocabulary size (0-359 angles + end_of_doc) |

### TrainingConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr_init` | 6e-4 | Initial learning rate |
| `lr_final` | 6e-5 | Final learning rate |
| `micro_bsz` | 16 | Batch size per GPU |
| `grad_cp` | 1 | Gradient checkpointing (0 or 1) |
| `epoch_save` | 1 | Save checkpoint every N epochs |
| `precision` | "bf16" | Precision: "bf16", "fp16", "fp32" |

## Data Format

Input data should be numpy arrays of integers in range [0, 359], representing angles in degrees:

```python
# Shape: (n_sequences, sequence_length)
data = np.array([
    [0, 45, 90, ..., 315],
    [10, 55, 100, ..., 320],
    ...
])
```

## Testing

Run tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_tokenizer.py -v
pytest tests/test_data_converter.py -v
pytest tests/test_pipeline.py -v

# Run integration tests
pytest tests/test_integration.py -v
```

## Advanced Usage

### Step-by-step pipeline

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

### Continue training

```python
pipeline = RWKVTrainingPipeline(work_dir="./existing_experiment")
# Don't prepare data again, just continue training
pipeline.train(num_epochs=200, continue_training=True)
```

### Load from numpy file

```python
pipeline = RWKVTrainingPipeline(work_dir="./experiment")
pipeline.prepare_data_from_file("path/to/data.npy", "train")
pipeline.train(num_epochs=100)
```

## License

Same as RWKV-LM
