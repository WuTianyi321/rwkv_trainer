#!/usr/bin/env python3
"""
Simple tests for AngleTokenizer (no pytest required)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_utils.tokenizer import AngleTokenizer


def test_default_vocab_size():
    """Test that default vocab size is 361"""
    tokenizer = AngleTokenizer()
    assert tokenizer.vocab_size == 361, f"Expected 361, got {tokenizer.vocab_size}"
    print("✓ test_default_vocab_size passed")


def test_encode_decode_single_angle():
    """Test encoding and decoding single angle"""
    tokenizer = AngleTokenizer()
    
    for angle in [0, 100, 200, 359]:
        tokens = tokenizer.encode_angle_sequence([angle])
        assert len(tokens) == 1, f"Expected 1 token, got {len(tokens)}"
        assert tokens[0] == angle + 1, f"Expected {angle + 1}, got {tokens[0]}"
        
        decoded = tokenizer.decode_angle_sequence(tokens)
        assert decoded == [angle], f"Expected [{angle}], got {decoded}"
    
    print("✓ test_encode_decode_single_angle passed")


def test_encode_decode_sequence():
    """Test encoding and decoding sequence of angles"""
    tokenizer = AngleTokenizer()
    
    angles = [0, 45, 90, 135, 180, 225, 270, 315, 359]
    tokens = tokenizer.encode_angle_sequence(angles)
    
    assert len(tokens) == len(angles), f"Expected {len(angles)} tokens, got {len(tokens)}"
    assert all(1 <= t <= 360 for t in tokens), "Tokens out of range"
    
    decoded = tokenizer.decode_angle_sequence(tokens)
    assert decoded == angles, f"Expected {angles}, got {decoded}"
    print("✓ test_encode_decode_sequence passed")


def test_out_of_range_angle():
    """Test that out-of-range angles raise error"""
    tokenizer = AngleTokenizer()
    
    try:
        tokenizer.encode_angle_sequence([-1])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    try:
        tokenizer.encode_angle_sequence([360])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    print("✓ test_out_of_range_angle passed")


def test_save_load_vocab():
    """Test saving and loading vocabulary"""
    import tempfile
    import os
    
    tokenizer = AngleTokenizer()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = os.path.join(tmpdir, "vocab.txt")
        
        # Save vocab
        tokenizer.save_vocab(vocab_path)
        assert os.path.exists(vocab_path), "Vocab file not created"
        
        # Load vocab
        tokenizer2 = AngleTokenizer(vocab_path)
        
        # Test that both tokenizers work the same
        angles = [0, 123, 359]
        tokens1 = tokenizer.encode_angle_sequence(angles)
        tokens2 = tokenizer2.encode_angle_sequence(angles)
        assert tokens1 == tokens2, f"Expected {tokens1}, got {tokens2}"
    
    print("✓ test_save_load_vocab passed")


def test_string_encode_single():
    """Test string-based encoding/decoding for single numbers"""
    tokenizer = AngleTokenizer()
    
    # Test single numbers (without spaces)
    for num in ["0", "45", "123", "359"]:
        tokens = tokenizer.encode(num)
        assert isinstance(tokens, list), "Expected list"
        assert len(tokens) > 0, "Expected non-empty list"
        
        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, str), "Expected string"
        assert decoded == num, f"Expected '{num}', got '{decoded}'"
    
    print("✓ test_string_encode_single passed")


def run_all_tests():
    """Run all tests"""
    print("Running AngleTokenizer tests...\n")
    
    tests = [
        test_default_vocab_size,
        test_encode_decode_single_angle,
        test_encode_decode_sequence,
        test_out_of_range_angle,
        test_save_load_vocab,
        test_string_encode_single,
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
