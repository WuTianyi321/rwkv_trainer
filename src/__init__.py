"""
RWKV Trainer - A packaged RWKV training framework

Supports:
- Custom sequence data (numpy, JSONL)
- Custom tokenizers and vocabularies
- Flexible model configurations
"""

__version__ = "0.1.0"

from .trainer.pipeline import RWKVTrainingPipeline, ModelConfig, TrainingConfig, DataConfig
from .data_utils.converter import NumpyToJsonlConverter, JsonlToBinIdxConverter, DataPipeline
from .data_utils.tokenizer import (
    BaseTokenizer,
    GenericTokenizer,
    AngleTokenizer,
    IntegerTokenizer,
    create_vocab_file_from_tokens,
    create_default_vocab_file
)

__all__ = [
    'RWKVTrainingPipeline',
    'ModelConfig',
    'TrainingConfig',
    'DataConfig',
    'NumpyToJsonlConverter', 
    'JsonlToBinIdxConverter',
    'DataPipeline',
    'BaseTokenizer',
    'GenericTokenizer',
    'AngleTokenizer',
    'IntegerTokenizer',
    'create_vocab_file_from_tokens',
    'create_default_vocab_file',
]
