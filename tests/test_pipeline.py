"""
Tests for RWKVTrainingPipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
import json

from trainer.pipeline import (
    RWKVTrainingPipeline, 
    ModelConfig, 
    TrainingConfig, 
    DataConfig
)


def test_model_config():
    """Test ModelConfig dataclass"""
    config = ModelConfig(
        model_type="x060",
        n_layer=3,
        n_embd=128,
        ctx_len=512
    )
    
    assert config.model_type == "x060"
    assert config.n_layer == 3
    assert config.n_embd == 128
    assert config.ctx_len == 512
    assert config.vocab_size == 361


def test_training_config():
    """Test TrainingConfig dataclass"""
    config = TrainingConfig(
        lr_init=6e-4,
        lr_final=6e-5,
        micro_bsz=8
    )
    
    assert config.lr_init == 6e-4
    assert config.lr_final == 6e-5
    assert config.micro_bsz == 8


def test_pipeline_initialization(tmp_path):
    """Test pipeline initialization"""
    work_dir = tmp_path / "test_work"
    
    pipeline = RWKVTrainingPipeline(
        work_dir=work_dir,
        model_config=ModelConfig(n_layer=2, n_embd=64),
        training_config=TrainingConfig(lr_init=1e-3),
        wandb_project=None
    )
    
    # Check directories created
    assert (work_dir / "data").exists()
    assert (work_dir / "out").exists()
    assert (work_dir / "configs").exists()
    
    # Check config saved
    assert (work_dir / "configs" / "config.json").exists()


def test_pipeline_prepare_data(tmp_path):
    """Test data preparation in pipeline"""
    work_dir = tmp_path / "test_work"
    
    pipeline = RWKVTrainingPipeline(
        work_dir=work_dir,
        data_config=DataConfig(sequence_length=10)
    )
    
    # Create test data
    data = np.random.randint(0, 360, size=(20, 10))
    
    result = pipeline.prepare_data(data, "train")
    
    # Check outputs
    assert 'jsonl_path' in result
    assert 'binidx_prefix' in result
    assert 'magic_prime' in result
    assert 'total_tokens' in result
    
    assert result['jsonl_path'].exists()
    assert Path(str(result['binidx_prefix']) + ".bin").exists()
    
    assert pipeline.data_prepared == True


def test_pipeline_prepare_data_from_file(tmp_path):
    """Test loading numpy file and preparing data"""
    work_dir = tmp_path / "test_work"
    
    # Save test numpy file
    data = np.random.randint(0, 360, size=(30, 15))
    npy_path = tmp_path / "test_data.npy"
    np.save(npy_path, data)
    
    pipeline = RWKVTrainingPipeline(
        work_dir=work_dir,
        data_config=DataConfig(sequence_length=15)
    )
    
    result = pipeline.prepare_data_from_file(npy_path, "train")
    
    assert result['num_sequences'] == 30


def test_detect_jsonl_format_text(tmp_path):
    """Text JSONL should be detected as text (no skipped first line)."""
    work_dir = tmp_path / "test_work"
    jsonl_path = tmp_path / "text_data.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "hello world"}) + "\n")
        f.write(json.dumps({"text": "1 2 3"}) + "\n")

    pipeline = RWKVTrainingPipeline(work_dir=work_dir)
    assert pipeline._detect_jsonl_format(jsonl_path) == "text"


def test_detect_jsonl_format_integer(tmp_path):
    """Integer JSONL should still be detected as integer."""
    work_dir = tmp_path / "test_work"
    jsonl_path = tmp_path / "int_data.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "1 2 3 4"}) + "\n")
        f.write(json.dumps({"text": "5 6 7 8"}) + "\n")

    pipeline = RWKVTrainingPipeline(work_dir=work_dir)
    assert pipeline._detect_jsonl_format(jsonl_path) == "integer"


def test_pipeline_save_config(tmp_path):
    """Test that pipeline saves configuration"""
    work_dir = tmp_path / "test_work"
    
    pipeline = RWKVTrainingPipeline(
        work_dir=work_dir,
        model_config=ModelConfig(n_layer=3, n_embd=128),
        training_config=TrainingConfig(lr_init=6e-4, lr_final=6e-5),
        wandb_project="test_project"
    )
    
    config_path = work_dir / "configs" / "config.json"
    assert config_path.exists()
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    assert config['model']['n_layer'] == 3
    assert config['model']['n_embd'] == 128
    assert config['training']['lr_init'] == 6e-4
    assert config['wandb_project'] == "test_project"


def test_pipeline_get_data_info(tmp_path):
    """Test getting data info after preparation"""
    work_dir = tmp_path / "test_work"
    
    pipeline = RWKVTrainingPipeline(
        work_dir=work_dir,
        data_config=DataConfig(sequence_length=10)
    )
    
    # Prepare data
    data = np.random.randint(0, 360, size=(50, 10))
    pipeline.prepare_data(data, "train")
    
    # Check info is stored
    assert pipeline.my_exit_tokens is not None
    assert pipeline.my_exit_tokens > 0
    assert pipeline.magic_prime is not None


def test_pipeline_uses_current_python_for_subprocess(tmp_path):
    """Pipeline should launch training subprocess with current interpreter."""
    pipeline = RWKVTrainingPipeline(work_dir=tmp_path / "test_work")
    pipeline.data_prefix = "dummy"
    pipeline.magic_prime = 2
    pipeline.my_exit_tokens = 2048
    cmd = pipeline._build_train_command(stage=1, random_seed=-1)
    assert cmd[0] == sys.executable


def test_pipeline_magic_prime_fallback_for_small_data(tmp_path):
    """Small-data path should get a safe fallback magic_prime."""
    pipeline = RWKVTrainingPipeline(work_dir=tmp_path / "test_work")
    pipeline.model_config.ctx_len = 1024
    pipeline.my_exit_tokens = 1025
    pipeline.magic_prime = None
    pipeline._normalize_data_stats()
    assert pipeline.magic_prime == 2


def test_pipeline_rejects_too_small_dataset(tmp_path):
    """Dataset with <= ctx_len tokens cannot create training pairs."""
    pipeline = RWKVTrainingPipeline(work_dir=tmp_path / "test_work")
    pipeline.model_config.ctx_len = 1024
    pipeline.my_exit_tokens = 1024
    pipeline.data_prefix = "dummy"
    pipeline.magic_prime = 2
    with pytest.raises(RuntimeError, match="Dataset too small"):
        pipeline._build_train_command(stage=1, random_seed=-1)


@pytest.mark.skip(reason="Requires actual model training, too slow for unit tests")
def test_pipeline_initialize_model():
    """Test model initialization (requires PyTorch)"""
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
