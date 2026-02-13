"""
Data conversion utilities: numpy -> jsonl -> bin/idx

Supports:
- Custom tokenizers
- Generic numpy arrays
- Pre-built JSONL files
"""

import json
import os
import random
import struct
from pathlib import Path
from typing import List, Union, Optional, Callable
import numpy as np

from .tokenizer import BaseTokenizer, IntegerTokenizer
from .binidx import MMapIndexedDatasetBuilder


class NumpyToJsonlConverter:
    """
    Convert numpy arrays to JSONL format
    
    Supports:
    - Integer arrays (0-max_value) -> space-separated string
    - Custom transform function
    """
    
    def __init__(self, transform: Optional[Callable] = None):
        """
        Initialize converter
        
        Args:
            transform: Optional function to transform array elements to strings
                      If None, uses str() for each element
        """
        self.transform = transform or (lambda x: str(int(x)))
    
    def convert(self, 
                data: np.ndarray, 
                output_path: Union[str, Path],
                sequence_length: Optional[int] = None) -> Path:
        """
        Convert numpy array to JSONL
        
        Args:
            data: numpy array of shape (n_sequences, seq_len) or (total_len,)
            output_path: output JSONL file path
            sequence_length: if data is 1D, split into sequences of this length
            
        Returns:
            Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle 1D array
        if data.ndim == 1:
            if sequence_length is None:
                # Single sequence
                sequences = [data]
            else:
                # Split into chunks
                n_full = len(data) // sequence_length
                sequences = [
                    data[i*sequence_length:(i+1)*sequence_length]
                    for i in range(n_full)
                ]
        else:
            sequences = data
        
        # Write to JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for seq in sequences:
                # Convert values to space-separated string
                value_str = ' '.join(self.transform(x) for x in seq)
                json_line = json.dumps({"text": value_str})
                f.write(json_line + '\n')
        
        return output_path
    
    def convert_file(self,
                     input_path: Union[str, Path],
                     output_path: Union[str, Path],
                     sequence_length: int = 1024) -> Path:
        """
        Load numpy file and convert to JSONL
        
        Args:
            input_path: path to .npy file
            output_path: output JSONL file path
            sequence_length: split sequences to this length
            
        Returns:
            Path to output file
        """
        data = np.load(input_path)
        return self.convert(data, output_path, sequence_length)


class JsonlToBinIdxConverter:
    """
    Convert JSONL to binary indexed format (.bin + .idx)
    Used for efficient memory-mapped training data loading
    
    Supports custom tokenizers for different data types
    """
    
    def __init__(self, tokenizer: Optional[BaseTokenizer] = None):
        """
        Initialize converter
        
        Args:
            tokenizer: Tokenizer to use. If None, uses IntegerTokenizer
        """
        self.tokenizer = tokenizer or IntegerTokenizer(max_value=359)
    
    def convert(self,
                jsonl_path: Union[str, Path],
                output_prefix: Union[str, Path],
                n_epochs: int = 3,
                shuffle: bool = True,
                random_seed: Optional[int] = None) -> Path:
        """
        Convert JSONL to bin/idx format
        
        Args:
            jsonl_path: input JSONL file
            output_prefix: output file prefix (without extension)
            n_epochs: number of times to duplicate and shuffle data
            shuffle: whether to shuffle data
            random_seed: optional random seed for reproducible shuffling
            
        Returns:
            Path to output prefix
        """
        jsonl_path = Path(jsonl_path)
        output_prefix = Path(output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        
        # Set random seed if provided (for reproducibility)
        if random_seed is not None:
            random.seed(random_seed)
            print(f"### Using fixed random seed: {random_seed}")
        
        # Load JSONL
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"### Loaded {len(lines)} sequences from {jsonl_path}")
        
        # Create temp file with shuffled duplicates
        temp_file = output_prefix.parent / f"{output_prefix.name}_temp.jsonl"
        with open(temp_file, 'w', encoding='utf-8') as f:
            for epoch in range(n_epochs):
                if shuffle:
                    random.shuffle(lines)
                for line in lines:
                    f.write(line + '\n')
        
        # Build bin/idx
        print("### Building bin/idx...")
        builder = MMapIndexedDatasetBuilder(str(output_prefix) + ".bin")
        
        with open(temp_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                text = data["text"]
                
                # Tokenize using the configured tokenizer
                tokens = self.tokenizer.encode(text)
                tokens.append(0)  # end_of_doc token
                
                builder.add_item(np.array(tokens, dtype=np.uint16))
                builder.end_document()
                
                if i % 500 == 0:
                    print(f"Processed {i} sequences...", end='\r')
        
        builder.finalize(str(output_prefix) + ".idx")
        print(f"\n### Done! Created {output_prefix}.bin and {output_prefix}.idx")
        
        # Clean up temp file
        os.remove(temp_file)
        
        return output_prefix
    
    def get_data_info(self, prefix: Union[str, Path]) -> dict:
        """
        Get information about a bin/idx dataset
        
        Args:
            prefix: path prefix to .bin/.idx files
            
        Returns:
            Dictionary with dataset information
        """
        from .binidx import MMapIndexedDataset
        
        dataset = MMapIndexedDataset(str(prefix))
        data_size = len(dataset._bin_buffer) // dataset._index._dtype_size
        
        return {
            'num_items': len(dataset),
            'total_tokens': data_size,
            'dtype': str(dataset._index.dtype),
        }
    
    def compute_magic_prime(self, 
                           data_prefix: Union[str, Path], 
                           ctx_len: int) -> Optional[int]:
        """
        Compute magic_prime for given context length
        
        magic_prime is the largest prime of form 3n+2 that is smaller than
        datalen/ctxlen - 1. This is used for the sampling strategy in RWKV.
        
        Args:
            data_prefix: path prefix to .bin/.idx files
            ctx_len: context length
            
        Returns:
            The magic prime number, or None if data is too small
        """
        info = self.get_data_info(data_prefix)
        data_size = info['total_tokens']
        
        if data_size < ctx_len * 3:
            return None
        
        n_chunk = int(data_size // ctx_len) - 1
        
        # Find largest 3n+2 prime <= n_chunk
        for i in range(n_chunk, 0, -1):
            if i % 3 == 2 and self._is_prime(i):
                return i
        
        return None
    
    @staticmethod
    def _is_prime(n: int) -> bool:
        """Check if n is prime"""
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True


class DataPipeline:
    """
    Complete data processing pipeline: numpy -> jsonl -> bin/idx
    
    Flexible pipeline supporting:
    - Custom tokenizers
    - Different data types
    - Pre-built JSONL files
    """
    
    def __init__(self, tokenizer: Optional[BaseTokenizer] = None):
        self.tokenizer = tokenizer or IntegerTokenizer(max_value=359)
        self.numpy_converter = NumpyToJsonlConverter()
        self.jsonl_converter = JsonlToBinIdxConverter(self.tokenizer)

    def process(self,
                data: np.ndarray,
                output_dir: Union[str, Path],
                name: str = "data",
                sequence_length: int = 1024,
                n_epochs: int = 3) -> dict:
        """
        Backward-compatible alias for processing numpy data.
        """
        return self.process_numpy(
            data=data,
            output_dir=output_dir,
            name=name,
            sequence_length=sequence_length,
            n_epochs=n_epochs,
        )
    
    def process_numpy(self,
                     data: np.ndarray,
                     output_dir: Union[str, Path],
                     name: str = "data",
                     sequence_length: int = 1024,
                     n_epochs: int = 3) -> dict:
        """
        Process numpy data through full pipeline
        
        Args:
            data: numpy array of integer sequences
            output_dir: output directory
            name: base name for output files
            sequence_length: sequence length for splitting
            n_epochs: number of epochs for data duplication
            
        Returns:
            Dictionary with paths and information
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: numpy -> jsonl
        jsonl_path = output_dir / f"{name}.jsonl"
        self.numpy_converter.convert(data, jsonl_path, sequence_length)
        
        # Step 2: jsonl -> bin/idx
        binidx_prefix = output_dir / name
        self.jsonl_converter.convert(jsonl_path, binidx_prefix, n_epochs)
        
        # Get info
        info = self.jsonl_converter.get_data_info(binidx_prefix)
        magic_prime = self.jsonl_converter.compute_magic_prime(binidx_prefix, sequence_length)
        
        # Save vocab
        vocab_path = output_dir / "vocab.txt"
        self.tokenizer.save_vocab(vocab_path)
        
        return {
            'jsonl_path': jsonl_path,
            'binidx_prefix': binidx_prefix,
            'vocab_path': vocab_path,
            'num_sequences': info['num_items'],
            'total_tokens': info['total_tokens'],
            'magic_prime': magic_prime,
        }
    
    def process_jsonl(self,
                     jsonl_path: Union[str, Path],
                     output_dir: Union[str, Path],
                     name: str = "data",
                     n_epochs: int = 3) -> dict:
        """
        Process existing JSONL file through bin/idx conversion
        
        Args:
            jsonl_path: path to existing JSONL file
            output_dir: output directory
            name: base name for output files
            n_epochs: number of epochs for data duplication
            
        Returns:
            Dictionary with paths and information
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy JSONL to output dir
        import shutil
        jsonl_dest = output_dir / f"{name}.jsonl"
        shutil.copy(jsonl_path, jsonl_dest)
        
        # Convert to bin/idx
        binidx_prefix = output_dir / name
        self.jsonl_converter.convert(jsonl_dest, binidx_prefix, n_epochs)
        
        # Get info (use ctx_len=1024 as default)
        info = self.jsonl_converter.get_data_info(binidx_prefix)
        magic_prime = self.jsonl_converter.compute_magic_prime(binidx_prefix, 1024)
        
        # Save vocab
        vocab_path = output_dir / "vocab.txt"
        self.tokenizer.save_vocab(vocab_path)
        
        return {
            'jsonl_path': jsonl_dest,
            'binidx_prefix': binidx_prefix,
            'vocab_path': vocab_path,
            'num_sequences': info['num_items'],
            'total_tokens': info['total_tokens'],
            'magic_prime': magic_prime,
        }
