#!/usr/bin/env python3
"""
Simple example of using RWKVTrainingPipeline

This example shows how to:
1. Create synthetic angle data
2. Prepare data for training
3. Initialize model
4. Run training
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from trainer.pipeline import (
    RWKVTrainingPipeline, 
    ModelConfig, 
    TrainingConfig, 
    DataConfig
)


def generate_synthetic_data(n_sequences=1000, seq_length=100):
    """
    Generate synthetic angle data for testing
    
    This simulates Vicsek model angle data
    """
    print(f"Generating synthetic data: {n_sequences} sequences of length {seq_length}")
    
    # Generate random angles
    data = np.random.randint(0, 360, size=(n_sequences, seq_length))
    
    return data


def main():
    # Configuration
    WORK_DIR = "./test_experiment"
    
    # Model configuration (small model for testing)
    model_config = ModelConfig(
        model_type="x060",      # RWKV-6.0
        n_layer=3,              # 3 layers
        n_embd=128,             # 128 embedding dimensions
        ctx_len=1024,           # context length
        vocab_size=361          # 0-359 angles + end_of_doc
    )
    
    # Training configuration
    training_config = TrainingConfig(
        lr_init=6e-4,           # initial learning rate
        lr_final=6e-5,          # final learning rate
        micro_bsz=16,           # batch size per GPU
        grad_cp=1,              # gradient checkpointing (save memory)
        epoch_save=1            # save every epoch
    )
    
    # Data configuration
    data_config = DataConfig(
        sequence_length=1024,   # split sequences to this length
        n_epochs_duplication=3  # duplicate data 3 times with shuffling
    )
    
    # Initialize pipeline
    print("### Initializing pipeline...")
    pipeline = RWKVTrainingPipeline(
        work_dir=WORK_DIR,
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        wandb_project=None  # Set to project name to enable wandb logging
    )
    
    # Generate or load data
    print("### Preparing data...")
    data = generate_synthetic_data(n_sequences=1000, seq_length=1024)
    
    # Run complete training pipeline
    # This will:
    # 1. Convert numpy -> jsonl -> bin/idx
    # 2. Initialize model (Stage 1)
    # 3. Run training (Stage 3)
    print("### Starting training...")
    pipeline.train(
        data=data,
        num_epochs=10
    )
    
    print("### Training complete!")
    print(f"Checkpoints saved in: {pipeline.output_dir}")
    
    # List checkpoints
    checkpoints = pipeline.list_checkpoints()
    print(f"\nCheckpoints:")
    for ckpt in checkpoints:
        print(f"  - {ckpt}")


if __name__ == "__main__":
    main()
