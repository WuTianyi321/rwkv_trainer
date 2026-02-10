#!/usr/bin/env python3
"""
Simple tests for data converters (no pytest required)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tempfile
import os
import json
import numpy as np

from data_utils.converter import (
    NumpyToJsonlConverter, 
    JsonlToBinIdxConverter,
    DataPipeline
)
from data_utils.tokenizer import AngleTokenizer


def test_numpy_to_jsonl_converter():
    """Test converting numpy to JSONL"""
    converter = NumpyToJsonlConverter()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        data = np.array([[0, 45, 90], [135, 180, 225], [270, 315, 359]])
        output_path = os.path.join(tmpdir, "test.jsonl")
        
        result = converter.convert(data, output_path)
        
        assert os.path.exists(result), "Output file not created"
        
        # Verify content
        with open(result, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}"
        for i, line in enumerate(lines):
            data_dict = json.loads(line)
            assert "text" in data_dict, "Missing 'text' field"
            angles = [int(x) for x in data_dict["text"].split()]
            assert angles == data[i].tolist(), f"Line {i} mismatch"
    
    print("✓ test_numpy_to_jsonl_converter passed")


def test_numpy_to_jsonl_1d():
    """Test converting 1D numpy array"""
    converter = NumpyToJsonlConverter()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 1D test data
        data = np.arange(20)
        output_path = os.path.join(tmpdir, "test.jsonl")
        
        result = converter.convert(data, output_path, sequence_length=5)
        
        assert os.path.exists(result), "Output file not created"
        
        # Verify content - should have 4 sequences (20/5)
        with open(result, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 4, f"Expected 4 lines, got {len(lines)}"
    
    print("✓ test_numpy_to_jsonl_1d passed")


def test_jsonl_to_binidx_converter():
    """Test converting JSONL to bin/idx"""
    tokenizer = AngleTokenizer()
    converter = JsonlToBinIdxConverter(tokenizer)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test JSONL
        jsonl_path = os.path.join(tmpdir, "test.jsonl")
        with open(jsonl_path, 'w') as f:
            for i in range(10):
                angles = list(range(i * 10, (i + 1) * 10))
                f.write(json.dumps({"text": " ".join(map(str, angles))}) + '\n')
        
        output_prefix = os.path.join(tmpdir, "test_data")
        result = converter.convert(jsonl_path, output_prefix, n_epochs=1, shuffle=False)
        
        # Check files were created
        assert os.path.exists(str(result) + ".bin"), "Bin file not created"
        assert os.path.exists(str(result) + ".idx"), "Idx file not created"
        
        # Check info
        info = converter.get_data_info(result)
        assert 'num_items' in info, "Missing num_items"
        assert 'total_tokens' in info, "Missing total_tokens"
        assert info['num_items'] == 10, f"Expected 10 items, got {info['num_items']}"
    
    print("✓ test_jsonl_to_binidx_converter passed")


def test_compute_magic_prime():
    """Test magic prime computation"""
    tokenizer = AngleTokenizer()
    converter = JsonlToBinIdxConverter(tokenizer)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data with enough tokens
        jsonl_path = os.path.join(tmpdir, "test.jsonl")
        with open(jsonl_path, 'w') as f:
            for i in range(100):
                angles = list(range(100))  # 100 angles per sequence
                f.write(json.dumps({"text": " ".join(map(str, angles))}) + '\n')
        
        output_prefix = os.path.join(tmpdir, "test_data")
        converter.convert(jsonl_path, output_prefix, n_epochs=1, shuffle=False)
        
        magic_prime = converter.compute_magic_prime(output_prefix, ctx_len=50)
        
        assert magic_prime is not None, "Magic prime should not be None"
        assert magic_prime > 0, "Magic prime should be positive"
        # Check it's of form 3n+2
        assert magic_prime % 3 == 2, f"Magic prime {magic_prime} not of form 3n+2"
        
        print(f"  Magic prime computed: {magic_prime}")
    
    print("✓ test_compute_magic_prime passed")


def test_data_pipeline():
    """Test full data pipeline"""
    pipeline = DataPipeline()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        data = np.random.randint(0, 360, size=(50, 20))
        
        result = pipeline.process(
            data=data,
            output_dir=tmpdir,
            name="test",
            sequence_length=20,
            n_epochs=2
        )
        
        # Check outputs exist
        assert os.path.exists(result['jsonl_path']), "JSONL not created"
        assert os.path.exists(str(result['binidx_prefix']) + ".bin"), "Bin not created"
        assert os.path.exists(str(result['binidx_prefix']) + ".idx"), "Idx not created"
        
        # Check info (50 sequences * 2 epochs = 100 items)
        assert result['num_sequences'] == 100, f"Expected 100 sequences (50 * 2 epochs), got {result['num_sequences']}"
        assert result['total_tokens'] > 0, "Total tokens should be positive"
        assert result['magic_prime'] is not None, "Magic prime should not be None"
        
        print(f"  Total tokens: {result['total_tokens']}")
        print(f"  Magic prime: {result['magic_prime']}")
    
    print("✓ test_data_pipeline passed")


def test_is_prime():
    """Test prime checking function"""
    converter = JsonlToBinIdxConverter()
    
    assert converter._is_prime(2) == True, "2 should be prime"
    assert converter._is_prime(3) == True, "3 should be prime"
    assert converter._is_prime(4) == False, "4 should not be prime"
    assert converter._is_prime(17) == True, "17 should be prime"
    assert converter._is_prime(18) == False, "18 should not be prime"
    
    print("✓ test_is_prime passed")


def run_all_tests():
    """Run all tests"""
    print("Running data converter tests...\n")
    
    tests = [
        test_numpy_to_jsonl_converter,
        test_numpy_to_jsonl_1d,
        test_jsonl_to_binidx_converter,
        test_compute_magic_prime,
        test_data_pipeline,
        test_is_prime,
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
