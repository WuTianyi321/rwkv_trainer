"""
TRIE-based tokenizer for general sequence data
Adapted from RWKV-LM

Supports:
- Custom vocabulary from file
- Integer sequence tokenization
- Generic text tokenization
"""

import json
import os
from typing import List, Union, Optional, Dict, Any
from abc import ABC, abstractmethod


class TRIE:
    """TRIE data structure for efficient tokenization"""
    __slots__ = tuple("ch,to,values,front".split(","))
    
    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for _ in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while fr is not None:
            if fr.ch is not None:
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>" % (ret[::-1], self.values)
    
    def add(self, key: bytes, idx: int = 0, val=None):
        if idx == len(key):
            if val is None:
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if self.to[ch] is None:
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx+1, val=val)
    
    def find_longest(self, key: bytes, idx: int = 0):
        u: TRIE = self
        ch: int = key[idx]
        ret = None
        
        while u.to[ch] is not None:
            u = u.to[ch]
            idx += 1
            if u.values:
                ret = idx, u, u.values
            if idx == len(key):
                break
            ch = key[idx]
        return ret


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers"""
    
    @abstractmethod
    def encode(self, src: str) -> List[int]:
        """Encode string to token IDs"""
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to string"""
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return vocabulary size"""
        pass


class GenericTokenizer(BaseTokenizer):
    """
    Generic tokenizer that loads vocabulary from file
    
    Vocab file format (RWKV-LM style, token IDs start from 1):
        <token_id> <token_string_or_bytes_repr> <length>
    
    Example:
        1 'hello' 5
        2 'world' 5
        3 '\n' 1
    
    Note:
        - Token 0 is RESERVED internally for end_of_document
        - Vocab file should start from 1 (following RWKV convention)
    """
    
    def __init__(self, vocab_file: str):
        """
        Initialize tokenizer from vocabulary file
        
        Args:
            vocab_file: Path to vocabulary file
        """
        self.idx2token = {}
        self.token2idx = {}
        self.vocab_file = vocab_file
        
        self._load_from_file(vocab_file)
        
        # Add internal end_of_doc token (0)
        self.idx2token[0] = b'\x00'
        self.token2idx[b'\x00'] = 0
        
        # Build TRIE for encoding
        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))
    
    def _load_from_file(self, vocab_file: str):
        """Load vocabulary from file"""
        with open(vocab_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            idx = int(line[:line.index(' ')])
            x = eval(line[line.index(' '):line.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(line[line.rindex(' '):])
            
            self.idx2token[idx] = x
            self.token2idx[x] = idx
    
    def encodeBytes(self, src: bytes) -> List[int]:
        """Encode bytes to token IDs using TRIE"""
        idx = 0
        tokens = []
        while idx < len(src):
            _idx = idx
            result = self.root.find_longest(src, idx)
            if result is None:
                raise ValueError(f"Cannot encode byte at position {idx}: unknown token starting with {src[idx:]}")
            idx, _, values = result
            assert idx != _idx, "Tokenizer stuck at position"
            _, token = next(iter(values))
            tokens.append(token)
        return tokens
    
    def decodeBytes(self, tokens: List[int]) -> bytes:
        """Decode token IDs to bytes"""
        return b''.join(self.idx2token[i] for i in tokens if i in self.idx2token)
    
    def encode(self, src: str) -> List[int]:
        """Encode string to token IDs"""
        return self.encodeBytes(src.encode("utf-8"))
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to string"""
        try:
            return self.decodeBytes(tokens).decode('utf-8')
        except UnicodeDecodeError:
            return '\ufffd'  # replacement character for bad utf-8
    
    @property
    def vocab_size(self) -> int:
        return max(self.idx2token.keys()) + 1 if self.idx2token else 0
    
    def printTokens(self, tokens: List[int]):
        """Print tokens for debugging"""
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
        print()


class AngleTokenizer(BaseTokenizer):
    """
    Tokenizer for angle data (0-359 degrees)
    
    This is a simple integer tokenizer where:
    - Each angle value 0-359 maps to a unique token ID
    - Token 0 is reserved for end_of_document
    - Total vocab size = 361
    """
    
    END_OF_DOC_TOKEN = 0
    
    def __init__(self, vocab_file: str = None):
        """
        Initialize tokenizer
        
        Args:
            vocab_file: Path to vocab file (optional). If None, uses default 0-359 mapping.
        """
        self.idx2token = {}
        self.token2idx = {}
        
        if vocab_file and os.path.exists(vocab_file):
            # Load from file
            self._load_from_file(vocab_file)
        else:
            # Create default mapping for angles 0-359
            self._create_default_vocab()
        
        # Build TRIE for encoding
        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))
    
    def _create_default_vocab(self):
        """Create default vocabulary for angles 0-359"""
        for i in range(360):
            # Each angle is represented as its string form
            token_str = str(i)
            token_bytes = token_str.encode('utf-8')
            self.idx2token[i + 1] = token_bytes  # +1 because 0 is reserved
            self.token2idx[token_bytes] = i + 1
        
        # Token 0 is end_of_doc (represented as empty or special marker)
        self.idx2token[0] = b'\x00'
        self.token2idx[b'\x00'] = 0
    
    def _load_from_file(self, vocab_file: str):
        """Load vocabulary from file"""
        with open(vocab_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for line in lines:
            idx = int(line[:line.index(' ')])
            x = eval(line[line.index(' '):line.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(line[line.rindex(' '):])
            
            self.idx2token[idx] = x
            self.token2idx[x] = idx
    
    def encodeBytes(self, src: bytes) -> List[int]:
        """Encode bytes to token IDs using TRIE"""
        idx = 0
        tokens = []
        while idx < len(src):
            _idx = idx
            result = self.root.find_longest(src, idx)
            if result is None:
                raise ValueError(f"Cannot encode byte at position {idx}")
            idx, _, values = result
            assert idx != _idx, "Tokenizer stuck at position"
            _, token = next(iter(values))
            tokens.append(token)
        return tokens
    
    def decodeBytes(self, tokens: List[int]) -> bytes:
        """Decode token IDs to bytes"""
        return b''.join(self.idx2token[i] for i in tokens if i in self.idx2token)
    
    def encode(self, src: str) -> List[int]:
        """Encode string to token IDs"""
        return self.encodeBytes(src.encode("utf-8"))
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to string"""
        try:
            return self.decodeBytes(tokens).decode('utf-8')
        except UnicodeDecodeError:
            return '\ufffd'  # replacement character for bad utf-8
    
    def encode_angle_sequence(self, angles: List[int]) -> List[int]:
        """
        Encode a sequence of angle values
        
        Args:
            angles: List of integers 0-359
            
        Returns:
            List of token IDs
        """
        tokens = []
        for angle in angles:
            if not (0 <= angle <= 359):
                raise ValueError(f"Angle must be in [0, 359], got {angle}")
            tokens.append(angle + 1)  # +1 because 0 is reserved
        return tokens
    
    def decode_angle_sequence(self, tokens: List[int]) -> List[int]:
        """
        Decode token IDs to angle values
        
        Args:
            tokens: List of token IDs
            
        Returns:
            List of integers 0-359
        """
        angles = []
        for token in tokens:
            if token == 0:
                continue  # Skip end_of_doc tokens
            if not (1 <= token <= 360):
                raise ValueError(f"Invalid token ID {token}")
            angles.append(token - 1)  # -1 to get back to 0-359
        return angles
    
    def save_vocab(self, filepath: str):
        """Save vocabulary to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for idx in sorted(self.idx2token.keys()):
                token_bytes = self.idx2token[idx]
                # Represent as string if possible, otherwise as bytes
                try:
                    token_str = token_bytes.decode('utf-8')
                    token_repr = repr(token_str)
                except:
                    token_repr = repr(token_bytes)
                f.write(f"{idx} {token_repr} {len(token_bytes)}\n")
    
    @property
    def vocab_size(self) -> int:
        return 361
    
    def printTokens(self, tokens: List[int]):
        """Print tokens for debugging"""
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
        print()


class IntegerTokenizer(BaseTokenizer):
    """
    Generic integer sequence tokenizer
    
    Maps integers [0, max_value] to token IDs [1, max_value+1]
    Token 0 is reserved for end_of_document
    
    Usage:
        tokenizer = IntegerTokenizer(max_value=1000)
        tokens = tokenizer.encode_sequence([0, 100, 500, 999])
    """
    
    END_OF_DOC_TOKEN = 0
    
    def __init__(self, max_value: int = 359, vocab_file: str = None):
        """
        Initialize integer tokenizer
        
        Args:
            max_value: Maximum integer value (inclusive)
            vocab_file: Optional vocab file to load
        """
        self.max_value = max_value
        self.idx2token = {}
        self.token2idx = {}
        
        if vocab_file and os.path.exists(vocab_file):
            self._load_from_file(vocab_file)
        else:
            self._create_vocab()
        
        # Build TRIE for encoding (for string-based encoding)
        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))
    
    def _create_vocab(self):
        """Create vocabulary for integers 0 to max_value"""
        for i in range(self.max_value + 1):
            token_str = str(i)
            token_bytes = token_str.encode('utf-8')
            self.idx2token[i + 1] = token_bytes  # +1 because 0 is reserved
            self.token2idx[token_bytes] = i + 1
        
        # Token 0 is end_of_doc
        self.idx2token[0] = b'\x00'
        self.token2idx[b'\x00'] = 0
    
    def _load_from_file(self, vocab_file: str):
        """Load vocabulary from file"""
        with open(vocab_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for line in lines:
            idx = int(line[:line.index(' ')])
            x = eval(line[line.index(' '):line.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            self.idx2token[idx] = x
            self.token2idx[x] = idx
        
        # Update max_value
        self.max_value = max(self.idx2token.keys()) - 1
    
    def encode_sequence(self, values: List[int]) -> List[int]:
        """
        Encode a sequence of integer values
        
        Args:
            values: List of integers in range [0, max_value]
            
        Returns:
            List of token IDs
        """
        tokens = []
        for val in values:
            if not (0 <= val <= self.max_value):
                raise ValueError(f"Value must be in [0, {self.max_value}], got {val}")
            tokens.append(val + 1)  # +1 because 0 is reserved
        return tokens
    
    def decode_sequence(self, tokens: List[int]) -> List[int]:
        """
        Decode token IDs to integer values
        
        Args:
            tokens: List of token IDs
            
        Returns:
            List of integers
        """
        values = []
        for token in tokens:
            if token == 0:
                continue  # Skip end_of_doc tokens
            if not (1 <= token <= self.max_value + 1):
                raise ValueError(f"Invalid token ID {token}")
            values.append(token - 1)  # -1 to get back to original value
        return values
    
    def encode(self, src: str) -> List[int]:
        """Encode string to token IDs"""
        values = [int(x) for x in src.split()]
        return self.encode_sequence(values)
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to string"""
        values = self.decode_sequence(tokens)
        return ' '.join(str(v) for v in values)
    
    def save_vocab(self, filepath: str):
        """Save vocabulary to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for idx in sorted(self.idx2token.keys()):
                token_bytes = self.idx2token[idx]
                try:
                    token_str = token_bytes.decode('utf-8')
                    token_repr = repr(token_str)
                except:
                    token_repr = repr(token_bytes)
                f.write(f"{idx} {token_repr} {len(token_bytes)}\n")
    
    @property
    def vocab_size(self) -> int:
        return self.max_value + 2  # 0 to max_value + end_of_doc


def create_vocab_file_from_tokens(tokens: List[Union[str, bytes]], filepath: str):
    """
    Create a vocabulary file from a list of tokens
    
    Args:
        tokens: List of token strings or bytes
        filepath: Output vocabulary file path
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        # Token 0 is reserved for end_of_doc
        f.write(f"0 b'\\x00' 1\n")
        
        for idx, token in enumerate(tokens, start=1):
            if isinstance(token, str):
                token_bytes = token.encode('utf-8')
                token_repr = repr(token)
            else:
                token_bytes = token
                token_repr = repr(token)
            f.write(f"{idx} {token_repr} {len(token_bytes)}\n")


def create_default_vocab_file(filepath: str, max_value: int = 359):
    """Create a default vocabulary file for integer data"""
    tokenizer = IntegerTokenizer(max_value=max_value)
    tokenizer.save_vocab(filepath)
    return filepath
