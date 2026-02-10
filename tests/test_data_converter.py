"""
Tests for data converters
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
import json

from data_utils.converter import (
    NumpyToJsonlConverter, 
    JsonlToBinIdxConverter,
    DataPipeline
)
from data_utils.tokenizer import AngleTokenizer


def test_numpy_to_jsonl_converter(tmp_path):
    """Test converting numpy to JSONL"""
    converter = NumpyToJsonlConverter()
    
    # Create test data
    data = np.array([[0, 45, 90], [135, 180, 225], [270, 315, 359]])
    output_path = tmp_path / "test.jsonl"
    
    result = converter.convert(data, output_path)
    
    assert result.exists()
    
    # Verify content
    with open(result, 'r') as f:
        lines = f.readlines()
    
    assert len(lines) == 3
    for i, line in enumerate(lines):
        data_dict = json.loads(line)
        assert "text" in data_dict
        angles = [int(x) for x in data_dict["text"].split()]
        assert angles == data[i].tolist()


def test_numpy_to_jsonl_1d(tmp_path):
    """Test converting 1D numpy array"""
    converter = NumpyToJsonlConverter()
    
    # Create 1D test data
    data = np.arange(20)
    output_path = tmp_path / "test.jsonl"
    
    result = converter.convert(data, output_path, sequence_length=5)
    
    assert result.exists()
    
    # Verify content - should have 4 sequences (20/5)
    with open(result, 'r') as f:
        lines = f.readlines()
    
    assert len(lines) == 4


def test_jsonl_to_binidx_converter(tmp_path):
    """Test converting JSONL to bin/idx"""
    tokenizer = AngleTokenizer()
    converter = JsonlToBinIdxConverter(tokenizer)
    
    # Create test JSONL
    jsonl_path = tmp_path / "test.jsonl"
    with open(jsonl_path, 'w') as f:
        for i in range(10):
            angles = list(range(i * 10, (i + 1) * 10))
            f.write(json.dumps({"text": " ".join(map(str, angles))}) + '\n')
    
    output_prefix = tmp_path / "test_data"
    result = converter.convert(jsonl_path, output_prefix, n_epochs=1, shuffle=False)
    
    # Check files were created
    assert Path(str(result) + ".bin").exists()
    assert Path(str(result) + ".idx").exists()
    
    # Check info
    info = converter.get_data_info(result)
    assert 'num_items' in info
    assert 'total_tokens' in info
    assert info['num_items'] == 10


def test_compute_magic_prime(tmp_path):
    """Test magic prime computation"""
    tokenizer = AngleTokenizer()
    converter = JsonlToBinIdxConverter(tokenizer)
    
    # Create test data with enough tokens
    jsonl_path = tmp_path / "test.jsonl"
    with open(jsonl_path, 'w') as f:
        for i in range(100):
            angles = list(range(100))  # 100 angles per sequence
            f.write(json.dumps({"text": " ".join(map(str, angles))}) + '\n')
    
    output_prefix = tmp_path / "test_data"
    converter.convert(jsonl_path, output_prefix, n_epochs=1, shuffle=False)
    
    magic_prime = converter.compute_magic_prime(output_prefix, ctx_len=50)
    
    assert magic_prime is not None
    assert magic_prime > 0
    # Check it's of form 3n+2
    assert magic_prime % 3 == 2


def test_data_pipeline(tmp_path):
    """Test full data pipeline"""
    pipeline = DataPipeline()
    
    # Create test data
    data = np.random.randint(0, 360, size=(50, 20))
    
    result = pipeline.process(
        data=data,
        output_dir=tmp_path,
        name="test",
        sequence_length=20,
        n_epochs=2
    )
    
    # Check outputs exist
    assert result['jsonl_path'].exists()
    assert Path(str(result['binidx_prefix']) + ".bin").exists()
    assert Path(str(result['binidx_prefix']) + ".idx").exists()
    
    # Check info
    assert result['num_sequences'] == 50
    assert result['total_tokens'] > 0
    assert result['magic_prime'] is not None


def test_is_prime():
    """Test prime checking function"""
    converter = JsonlToBinIdxConverter()
    
    assert converter._is_prime(2) == True
    assert converter._is_prime(3) == True
    assert converter._is_prime(4) == False
    assert converter._is_prime(17) == True
    assert converter._is_prime(18) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
