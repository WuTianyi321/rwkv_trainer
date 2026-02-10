#!/bin/bash
# Run all tests for rwkv_trainer

echo "==============================================="
echo "Running RWKV Trainer Tests"
echo "==============================================="
echo ""

cd "$(dirname "$0")"

echo "1. Testing Tokenizer..."
python tests/test_tokenizer_simple.py
echo ""

echo "2. Testing Data Converter..."
python tests/test_data_converter_simple.py
echo ""

echo "3. Testing Training Pipeline..."
python tests/test_pipeline_simple.py
echo ""

echo "==============================================="
echo "All tests completed!"
echo "==============================================="
