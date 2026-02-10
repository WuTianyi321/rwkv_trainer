"""
RWKV Trainer - A packaged RWKV training framework for Vicsek model data
"""

__version__ = "0.1.0"

from .trainer.pipeline import RWKVTrainingPipeline
from .data_utils.converter import NumpyToJsonlConverter, JsonlToBinIdxConverter
from .data_utils.tokenizer import AngleTokenizer

__all__ = [
    'RWKVTrainingPipeline',
    'NumpyToJsonlConverter', 
    'JsonlToBinIdxConverter',
    'AngleTokenizer',
]
