#!/usr/bin/env python3
"""
Strict comparison test between original make_data.py and our implementation.
Both use fixed random seed=42 and same vocabulary.
"""

import sys
import os
import subprocess
import numpy as np

sys.path.insert(0, 'src')

# Test data
TEST_JSONL = 'test_strict.jsonl'
RANDOM_SEED = 42

# Create test data
with open(TEST_JSONL, 'w') as f:
    f.write('{"text":"0,45,90,"}\n')
    f.write('{"text":"135,180,225,"}\n')
    f.write('{"text":"270,315,359,"}\n')
    f.write('{"text":"10,20,30,"}\n')
    f.write('{"text":"100,200,300,"}\n')

print("="*70)
print("STRICT COMPARISON TEST")
print("="*70)
print(f"Test file: {TEST_JSONL}")
print(f"Random seed: {RANDOM_SEED}")
print()

# 1. Run original make_data.py with fixed seed
print("1. Running ORIGINAL make_data.py with fixed seed...")
result = subprocess.run([
    'python3', '/mnt/ssd2t/home/wty/lib/RWKV-LM/RWKV-v5/make_data_fixed_seed.py',
    TEST_JSONL, '1', '/mnt/ssd2t/home/wty/lib/RWKV-LM/RWKV-v5/test_result', '1024'
], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr)
    exit(1)

# 2. Run our implementation with fixed seed
print("\n2. Running OUR implementation with fixed seed...")
from data_utils.converter import JsonlToBinIdxConverter
from data_utils.tokenizer import GenericTokenizer
from data_utils.binidx import MMapIndexedDataset

tokenizer = GenericTokenizer('/mnt/ssd2t/home/wty/research/vicsek_near_critical/tokenizer/vocab.txt')
converter = JsonlToBinIdxConverter(tokenizer)

result = converter.convert(
    TEST_JSONL,
    'my_result',
    n_epochs=1,
    shuffle=True,
    random_seed=RANDOM_SEED
)

# 3. Compare results
print("\n3. COMPARING RESULTS...")

# Read original
orig_dataset = MMapIndexedDataset('/mnt/ssd2t/home/wty/lib/RWKV-LM/RWKV-v5/test_result')
orig_tokens = []
for i in range(len(orig_dataset)):
    orig_tokens.extend(orig_dataset[i].tolist())

# Read ours
my_dataset = MMapIndexedDataset('my_result')
my_tokens = []
for i in range(len(my_dataset)):
    my_tokens.extend(my_dataset[i].tolist())

print(f"\nOriginal tokens ({len(orig_tokens)} total): {orig_tokens}")
print(f"My tokens       ({len(my_tokens)} total): {my_tokens}")

# Check if identical
if orig_tokens == my_tokens:
    print("\n" + "="*70)
    print("✅ SUCCESS: Results are IDENTICAL!")
    print("="*70)
else:
    print("\n" + "="*70)
    print("❌ MISMATCH: Results differ!")
    print("="*70)
    
    # Find differences
    min_len = min(len(orig_tokens), len(my_tokens))
    for i in range(min_len):
        if orig_tokens[i] != my_tokens[i]:
            print(f"First difference at position {i}:")
            print(f"  Original: {orig_tokens[i]}")
            print(f"  Mine:     {my_tokens[i]}")
            break
    
    if len(orig_tokens) != len(my_tokens):
        print(f"Length mismatch: {len(orig_tokens)} vs {len(my_tokens)}")

# 4. Also compare raw binary files
print("\n4. COMPARING RAW BINARY FILES...")
with open('/mnt/ssd2t/home/wty/lib/RWKV-LM/RWKV-v5/test_result.bin', 'rb') as f:
    orig_bin = f.read()
with open('my_result.bin', 'rb') as f:
    my_bin = f.read()

if orig_bin == my_bin:
    print("✅ Binary files are IDENTICAL!")
else:
    print("❌ Binary files differ!")
    print(f"  Original size: {len(orig_bin)} bytes")
    print(f"  My size:       {len(my_bin)} bytes")
    # Show hex diff
    import difflib
    orig_hex = orig_bin.hex()
    my_hex = my_bin.hex()
    print(f"  Original hex: {orig_hex[:80]}...")
    print(f"  My hex:       {my_hex[:80]}...")

# Cleanup
os.remove(TEST_JSONL)
print("\nTest complete!")
