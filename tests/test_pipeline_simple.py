#!/usr/bin/env python3
"""
Simple tests for RWKVTrainingPipeline (no pytest required)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tempfile
import os
import json
import numpy as np
import shutil

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
    
    assert config.model_type == "x060", "model_type mismatch"
    assert config.n_layer == 3, "n_layer mismatch"
    assert config.n_embd == 128, "n_embd mismatch"
    assert config.ctx_len == 512, "ctx_len mismatch"
    assert config.vocab_size == 361, "vocab_size should be 361"
    print("✓ test_model_config passed")


def test_training_config():
    """Test TrainingConfig dataclass"""
    config = TrainingConfig(
        lr_init=6e-4,
        lr_final=6e-5,
        micro_bsz=8
    )
    
    assert config.lr_init == 6e-4, "lr_init mismatch"
    assert config.lr_final == 6e-5, "lr_final mismatch"
    assert config.micro_bsz == 8, "micro_bsz mismatch"
    print("✓ test_training_config passed")


def test_pipeline_initialization():
    """Test pipeline initialization"""
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = os.path.join(tmpdir, "test_work")
        
        pipeline = RWKVTrainingPipeline(
            work_dir=work_dir,
            model_config=ModelConfig(n_layer=2, n_embd=64),
            training_config=TrainingConfig(lr_init=1e-3),
            wandb_project=None
        )
        
        # Check directories created
        assert os.path.exists(os.path.join(work_dir, "data")), "data dir not created"
        assert os.path.exists(os.path.join(work_dir, "out")), "out dir not created"
        assert os.path.exists(os.path.join(work_dir, "configs")), "configs dir not created"
        
        # Check config saved
        assert os.path.exists(os.path.join(work_dir, "configs", "config.json")), "config not saved"
    
    print("✓ test_pipeline_initialization passed")


def test_pipeline_prepare_data():
    """Test data preparation in pipeline"""
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = os.path.join(tmpdir, "test_work")
        
        pipeline = RWKVTrainingPipeline(
            work_dir=work_dir,
            data_config=DataConfig(sequence_length=10)
        )
        
        # Create test data
        data = np.random.randint(0, 360, size=(20, 10))
        
        result = pipeline.prepare_data(data, "train")
        
        # Check outputs
        assert 'jsonl_path' in result, "Missing jsonl_path"
        assert 'binidx_prefix' in result, "Missing binidx_prefix"
        assert 'magic_prime' in result, "Missing magic_prime"
        assert 'total_tokens' in result, "Missing total_tokens"
        
        assert os.path.exists(result['jsonl_path']), "jsonl not created"
        assert os.path.exists(str(result['binidx_prefix']) + ".bin"), "bin not created"
        
        assert pipeline.data_prepared == True, "data_prepared should be True"
        
        print(f"  Prepared {result['num_sequences']} sequences, {result['total_tokens']} tokens")
    
    print("✓ test_pipeline_prepare_data passed")


def test_pipeline_prepare_data_from_file():
    """Test loading numpy file and preparing data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = os.path.join(tmpdir, "test_work")
        
        # Save test numpy file
        data = np.random.randint(0, 360, size=(30, 15))
        npy_path = os.path.join(tmpdir, "test_data.npy")
        np.save(npy_path, data)
        
        pipeline = RWKVTrainingPipeline(
            work_dir=work_dir,
            data_config=DataConfig(sequence_length=15)
        )
        
        result = pipeline.prepare_data_from_file(npy_path, "train")
        
        # With n_epochs_duplication=3 (default), we get 30 * 3 = 90 sequences
        assert result['num_sequences'] == 90, f"Expected 90 sequences (30 * 3), got {result['num_sequences']}"
    
    print("✓ test_pipeline_prepare_data_from_file passed")


def test_pipeline_save_config():
    """Test that pipeline saves configuration"""
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = os.path.join(tmpdir, "test_work")
        
        pipeline = RWKVTrainingPipeline(
            work_dir=work_dir,
            model_config=ModelConfig(n_layer=3, n_embd=128),
            training_config=TrainingConfig(lr_init=6e-4, lr_final=6e-5),
            wandb_project="test_project"
        )
        
        config_path = os.path.join(work_dir, "configs", "config.json")
        assert os.path.exists(config_path), "config.json not created"
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        assert config['model']['n_layer'] == 3, "n_layer not saved correctly"
        assert config['model']['n_embd'] == 128, "n_embd not saved correctly"
        assert config['training']['lr_init'] == 6e-4, "lr_init not saved correctly"
        assert config['wandb_project'] == "test_project", "wandb_project not saved correctly"
    
    print("✓ test_pipeline_save_config passed")


def test_pipeline_get_data_info():
    """Test getting data info after preparation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = os.path.join(tmpdir, "test_work")
        
        pipeline = RWKVTrainingPipeline(
            work_dir=work_dir,
            data_config=DataConfig(sequence_length=10)
        )
        
        # Prepare data
        data = np.random.randint(0, 360, size=(50, 10))
        pipeline.prepare_data(data, "train")
        
        # Check info is stored
        assert pipeline.my_exit_tokens is not None, "my_exit_tokens should not be None"
        assert pipeline.my_exit_tokens > 0, "my_exit_tokens should be positive"
        assert pipeline.magic_prime is not None, "magic_prime should not be None"
    
    print("✓ test_pipeline_get_data_info passed")


def run_all_tests():
    """Run all tests"""
    print("Running RWKVTrainingPipeline tests...\n")
    
    tests = [
        test_model_config,
        test_training_config,
        test_pipeline_initialization,
        test_pipeline_prepare_data,
        test_pipeline_prepare_data_from_file,
        test_pipeline_save_config,
        test_pipeline_get_data_info,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
