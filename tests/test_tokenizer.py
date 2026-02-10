"""
Tests for AngleTokenizer
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from data_utils.tokenizer import AngleTokenizer


def test_default_vocab_size():
    """Test that default vocab size is 361"""
    tokenizer = AngleTokenizer()
    assert tokenizer.vocab_size == 361


def test_encode_decode_single_angle():
    """Test encoding and decoding single angle"""
    tokenizer = AngleTokenizer()
    
    # Test a few angles
    for angle in [0, 100, 200, 359]:
        tokens = tokenizer.encode_angle_sequence([angle])
        assert len(tokens) == 1
        assert tokens[0] == angle + 1  # +1 because 0 is reserved
        
        decoded = tokenizer.decode_angle_sequence(tokens)
        assert decoded == [angle]


def test_encode_decode_sequence():
    """Test encoding and decoding sequence of angles"""
    tokenizer = AngleTokenizer()
    
    angles = [0, 45, 90, 135, 180, 225, 270, 315, 359]
    tokens = tokenizer.encode_angle_sequence(angles)
    
    assert len(tokens) == len(angles)
    assert all(1 <= t <= 360 for t in tokens)
    
    decoded = tokenizer.decode_angle_sequence(tokens)
    assert decoded == angles


def test_out_of_range_angle():
    """Test that out-of-range angles raise error"""
    tokenizer = AngleTokenizer()
    
    with pytest.raises(ValueError):
        tokenizer.encode_angle_sequence([-1])
    
    with pytest.raises(ValueError):
        tokenizer.encode_angle_sequence([360])


def test_invalid_token():
    """Test that invalid tokens raise error"""
    tokenizer = AngleTokenizer()
    
    with pytest.raises(ValueError):
        tokenizer.decode_angle_sequence([400])


def test_save_load_vocab(tmp_path):
    """Test saving and loading vocabulary"""
    tokenizer = AngleTokenizer()
    vocab_path = tmp_path / "vocab.txt"
    
    # Save vocab
    tokenizer.save_vocab(str(vocab_path))
    assert vocab_path.exists()
    
    # Load vocab
    tokenizer2 = AngleTokenizer(str(vocab_path))
    
    # Test that both tokenizers work the same
    angles = [0, 123, 359]
    tokens1 = tokenizer.encode_angle_sequence(angles)
    tokens2 = tokenizer2.encode_angle_sequence(angles)
    assert tokens1 == tokens2


def test_string_encode_decode():
    """Test string-based encoding/decoding"""
    tokenizer = AngleTokenizer()
    
    text = "0 45 90 180 359"
    tokens = tokenizer.encode(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    
    decoded = tokenizer.decode(tokens)
    assert isinstance(decoded, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
