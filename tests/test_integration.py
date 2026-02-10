"""
Integration tests for the full pipeline
These tests use small models and small data to verify the entire workflow
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
import tempfile
import shutil

from trainer.pipeline import (
    RWKVTrainingPipeline, 
    ModelConfig, 
    TrainingConfig, 
    DataConfig
)


@pytest.fixture
def temp_work_dir():
    """Create a temporary working directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_end_to_end_data_preparation(temp_work_dir):
    """
    Test the complete data preparation pipeline:
    numpy -> jsonl -> bin/idx
    """
    # Create synthetic angle data
    np.random.seed(42)
    n_sequences = 100
    seq_length = 50
    data = np.random.randint(0, 360, size=(n_sequences, seq_length))
    
    # Initialize pipeline with small config
    pipeline = RWKVTrainingPipeline(
        work_dir=temp_work_dir,
        model_config=ModelConfig(
            model_type="x060",
            n_layer=2,
            n_embd=64,
            ctx_len=seq_length
        ),
        data_config=DataConfig(
            sequence_length=seq_length,
            n_epochs_duplication=2
        )
    )
    
    # Prepare data
    result = pipeline.prepare_data(data, "train")
    
    # Verify results
    assert result['num_sequences'] == n_sequences
    assert result['total_tokens'] > 0
    assert result['magic_prime'] is not None
    
    # Verify files exist
    assert (Path(temp_work_dir) / "data" / "train.jsonl").exists()
    assert (Path(temp_work_dir) / "data" / "train.bin").exists()
    assert (Path(temp_work_dir) / "data" / "train.idx").exists()
    
    print(f"Data prepared successfully!")
    print(f"  Total tokens: {result['total_tokens']}")
    print(f"  Magic prime: {result['magic_prime']}")


def test_pipeline_configuration_consistency(temp_work_dir):
    """
    Test that configuration is consistent throughout the pipeline
    """
    config = {
        'model_type': 'x060',
        'n_layer': 3,
        'n_embd': 128,
        'ctx_len': 64,
        'lr_init': 5e-4,
        'lr_final': 5e-5,
    }
    
    pipeline = RWKVTrainingPipeline(
        work_dir=temp_work_dir,
        model_config=ModelConfig(
            model_type=config['model_type'],
            n_layer=config['n_layer'],
            n_embd=config['n_embd'],
            ctx_len=config['ctx_len']
        ),
        training_config=TrainingConfig(
            lr_init=config['lr_init'],
            lr_final=config['lr_final']
        )
    )
    
    # Verify configuration is stored correctly
    assert pipeline.model_config.n_layer == config['n_layer']
    assert pipeline.model_config.n_embd == config['n_embd']
    assert pipeline.training_config.lr_init == config['lr_init']
    
    # Load saved config and verify
    import json
    config_path = Path(temp_work_dir) / "configs" / "config.json"
    with open(config_path, 'r') as f:
        saved_config = json.load(f)
    
    assert saved_config['model']['n_layer'] == config['n_layer']
    assert saved_config['model']['n_embd'] == config['n_embd']
    assert saved_config['training']['lr_init'] == config['lr_init']


def test_data_pipeline_idempotent(temp_work_dir):
    """
    Test that running data preparation multiple times produces consistent results
    """
    data = np.random.randint(0, 360, size=(50, 30))
    
    pipeline = RWKVTrainingPipeline(
        work_dir=temp_work_dir,
        data_config=DataConfig(sequence_length=30, n_epochs_duplication=1)
    )
    
    # Run twice with same data
    result1 = pipeline.prepare_data(data, "train")
    
    # Create new pipeline in same directory
    pipeline2 = RWKVTrainingPipeline(
        work_dir=temp_work_dir,
        data_config=DataConfig(sequence_length=30, n_epochs_duplication=1)
    )
    result2 = pipeline2.prepare_data(data, "train")
    
    # Results should be the same (except for potential shuffle differences)
    assert result1['num_sequences'] == result2['num_sequences']
    assert result1['total_tokens'] == result2['total_tokens']


def test_different_model_types(temp_work_dir):
    """
    Test that pipeline works with different model types
    """
    data = np.random.randint(0, 360, size=(20, 20))
    
    for model_type in ["x052", "x060"]:
        work_subdir = Path(temp_work_dir) / model_type
        
        pipeline = RWKVTrainingPipeline(
            work_dir=work_subdir,
            model_config=ModelConfig(
                model_type=model_type,
                n_layer=2,
                n_embd=64
            ),
            data_config=DataConfig(sequence_length=20)
        )
        
        result = pipeline.prepare_data(data, "train")
        
        assert result['magic_prime'] is not None
        print(f"Model type {model_type}: magic_prime = {result['magic_prime']}")


def test_large_data_handling(temp_work_dir):
    """
    Test handling of larger datasets
    """
    # Create larger dataset
    n_sequences = 1000
    seq_length = 100
    data = np.random.randint(0, 360, size=(n_sequences, seq_length))
    
    pipeline = RWKVTrainingPipeline(
        work_dir=temp_work_dir,
        data_config=DataConfig(
            sequence_length=seq_length,
            n_epochs_duplication=1
        )
    )
    
    result = pipeline.prepare_data(data, "train")
    
    assert result['num_sequences'] == n_sequences
    assert result['total_tokens'] == n_sequences * seq_length
    
    print(f"Large data test passed!")
    print(f"  Sequences: {result['num_sequences']}")
    print(f"  Total tokens: {result['total_tokens']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
