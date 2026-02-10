"""
TRIE-based tokenizer for angle data (0-359 degrees)
Adapted from RWKV-LM
"""

import json
import os
from typing import List, Union


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


class AngleTokenizer:
    """
    Tokenizer for angle data (0-359 degrees)
    
    This is a simple character-level tokenizer where:
    - Each angle value 0-359 maps to a unique token ID
    - Token 0 is reserved for end_of_document
    - Total vocab size = 361
    """
    
    VOCAB_SIZE = 361  # 0-359 angles + 1 for end_of_doc
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
        return self.VOCAB_SIZE
    
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


def create_default_vocab_file(filepath: str):
    """Create a default vocabulary file for angle data"""
    tokenizer = AngleTokenizer()
    tokenizer.save_vocab(filepath)
    return filepath
