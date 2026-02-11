"""
RWKV Training Pipeline - Unified interface for training

Supports:
- Custom data inputs (numpy, JSONL)
- Custom tokenizers and vocabularies
- Flexible model configurations
- Resume from external RWKV-LM checkpoints
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
import subprocess

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_utils.converter import DataPipeline
from data_utils.tokenizer import BaseTokenizer, IntegerTokenizer, GenericTokenizer


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    model_type: str = "x060"  # x052, x060, x070
    n_layer: int = 3
    n_embd: int = 128
    ctx_len: int = 1024
    vocab_size: int = 361  # Will be updated based on tokenizer
    head_size_a: int = 64


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    lr_init: float = 6e-4
    lr_final: float = 6e-5
    warmup_steps: int = 10
    beta1: float = 0.9
    beta2: float = 0.99
    adam_eps: float = 1e-18
    weight_decay: float = 0.001
    micro_bsz: int = 16
    grad_cp: int = 1  # gradient checkpointing
    epoch_save: int = 1
    precision: str = "bf16"  # bf16, fp16, fp32
    strategy: str = "deepspeed_stage_1"


@dataclass
class DataConfig:
    """Data configuration"""
    sequence_length: int = 1024
    n_epochs_duplication: int = 3


class RWKVTrainingPipeline:
    """
    Complete training pipeline for RWKV on custom sequence data
    
    Usage Examples:
        # 1. From numpy array (integer sequences)
        pipeline = RWKVTrainingPipeline(
            work_dir="./experiment",
            model_config=ModelConfig(n_layer=3, n_embd=128),
            training_config=TrainingConfig(lr_init=6e-4),
            tokenizer=IntegerTokenizer(max_value=999)  # Custom vocab size
        )
        data = np.random.randint(0, 1000, size=(1000, 1024))
        pipeline.train(data, num_epochs=100)
        
        # 2. From existing JSONL file
        pipeline.prepare_data_from_jsonl("data.jsonl")
        pipeline.train(num_epochs=100)
        
        # 3. Resume from external checkpoint
        pipeline.train_from_checkpoint(
            checkpoint_path="/path/to/rwkv-10.pth",
            num_epochs=100,
            auto_detect=True  # Auto-detect model architecture
        )
    """
    
    def __init__(self,
                 work_dir: Union[str, Path],
                 model_config: Optional[ModelConfig] = None,
                 training_config: Optional[TrainingConfig] = None,
                 data_config: Optional[DataConfig] = None,
                 tokenizer: Optional[BaseTokenizer] = None,
                 wandb_project: Optional[str] = None):
        """
        Initialize training pipeline
        
        Args:
            work_dir: working directory for outputs (checkpoints, logs, etc.)
            model_config: model architecture configuration
            training_config: training hyperparameters
            data_config: data processing configuration
            tokenizer: custom tokenizer (default: IntegerTokenizer(max_value=359))
            wandb_project: wandb project name (None to disable)
        """
        self.work_dir = Path(work_dir).resolve()
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        self.data_config = data_config or DataConfig()
        self.wandb_project = wandb_project
        
        # Initialize tokenizer
        self.tokenizer = tokenizer or IntegerTokenizer(max_value=359)
        
        # Update vocab_size in model_config based on tokenizer
        self.model_config.vocab_size = self.tokenizer.vocab_size
        
        # Create directories
        self.data_dir = self.work_dir / "data"
        self.output_dir = self.work_dir / "out"
        self.config_dir = self.work_dir / "configs"
        
        for d in [self.data_dir, self.output_dir, self.config_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Initialize data pipeline
        self.data_pipeline = DataPipeline(self.tokenizer)
        
        # State
        self.data_prepared = False
        self.model_initialized = False
        self.magic_prime = None
        self.my_exit_tokens = None
        self.data_prefix = None
        
        # Save config
        self._save_config()
    
    def _save_config(self):
        """Save configuration to file"""
        config = {
            'model': asdict(self.model_config),
            'training': asdict(self.training_config),
            'data': asdict(self.data_config),
            'wandb_project': self.wandb_project,
            'vocab_size': self.tokenizer.vocab_size,
        }
        with open(self.config_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def prepare_data(self, 
                     data: np.ndarray,
                     data_name: str = "train") -> Dict[str, Any]:
        """
        Prepare numpy data for training (numpy -> jsonl -> bin/idx)
        
        Args:
            data: numpy array of integer sequences (shape: (n_seq, seq_len) or (total_len,))
            data_name: name for this dataset
            
        Returns:
            Dictionary with data information
        """
        print(f"### Preparing data: {data_name}")
        print(f"    Input shape: {data.shape}")
        print(f"    Value range: [{data.min()}, {data.max()}]")
        print(f"    Vocab size: {self.tokenizer.vocab_size}")
        
        # Validate data range
        if data.min() < 0 or data.max() >= self.tokenizer.vocab_size - 1:
            print(f"    WARNING: Data values [{data.min()}, {data.max()}] may exceed tokenizer range")
        
        # Process data
        result = self.data_pipeline.process_numpy(
            data=data,
            output_dir=self.data_dir,
            name=data_name,
            sequence_length=self.data_config.sequence_length,
            n_epochs=self.data_config.n_epochs_duplication
        )
        
        self.data_prefix = result['binidx_prefix']
        self.magic_prime = result['magic_prime']
        self.my_exit_tokens = result['total_tokens']
        self.data_prepared = True
        
        print(f"### Data preparation complete:")
        print(f"    Total tokens: {self.my_exit_tokens}")
        print(f"    Magic prime: {self.magic_prime}")
        print(f"    Data prefix: {self.data_prefix}")
        print(f"    Vocab file: {result['vocab_path']}")
        
        return result
    
    def prepare_data_from_file(self,
                               file_path: Union[str, Path],
                               data_name: str = "train") -> Dict[str, Any]:
        """
        Load numpy file and prepare data
        
        Args:
            file_path: path to .npy file
            data_name: name for this dataset
            
        Returns:
            Dictionary with data information
        """
        data = np.load(file_path)
        return self.prepare_data(data, data_name)
    
    def _detect_jsonl_format(self, jsonl_path: Path) -> str:
        """
        Detect JSONL format by trying to parse first few lines
        
        Returns:
            'integer': if all values are integers (space-separated)
            'text': if contains non-integer tokens
            'unknown': if cannot determine
        """
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                lines = [f.readline().strip() for _ in range(5) if f.readline()]
            
            if not lines:
                return 'unknown'
            
            all_integers = True
            for line in lines:
                if not line:
                    continue
                data = json.loads(line)
                text = data.get("text", "")
                tokens = text.split()
                
                for token in tokens:
                    try:
                        int(token)
                    except ValueError:
                        all_integers = False
                        break
                if not all_integers:
                    break
            
            return 'integer' if all_integers else 'text'
        except Exception:
            return 'unknown'
    
    def prepare_data_from_jsonl(self,
                                jsonl_path: Union[str, Path],
                                data_name: str = "train",
                                n_epochs: int = None,
                                vocab_file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare existing JSONL file for training with intelligent format detection
        
        This method will:
        1. Detect if JSONL contains integer sequences or text
        2. Validate that current tokenizer can handle the format
        3. Load custom vocabulary if provided (for text data)
        4. Provide helpful error messages if tokenizer mismatch
        
        Args:
            jsonl_path: path to existing JSONL file
            data_name: name for this dataset
            n_epochs: number of epochs for data duplication (default: from data_config)
            vocab_file_path: optional path to vocabulary file for text data
                            (if provided, will create GenericTokenizer automatically)
            
        Returns:
            Dictionary with data information
            
        Raises:
            RuntimeError: If tokenizer cannot handle the JSONL format
        """
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        
        print(f"### Preparing data from JSONL: {jsonl_path}")
        
        # If vocab file provided, create GenericTokenizer automatically
        if vocab_file_path:
            print(f"    Loading vocabulary from: {vocab_file_path}")
            from data_utils.tokenizer import GenericTokenizer
            self.tokenizer = GenericTokenizer(vocab_file_path)
            self.data_pipeline = DataPipeline(self.tokenizer)
            self.model_config.vocab_size = self.tokenizer.vocab_size
            self._save_config()
            print(f"    Vocab size: {self.tokenizer.vocab_size}")
        
        # Detect format
        detected_format = self._detect_jsonl_format(jsonl_path)
        print(f"    Detected format: {detected_format}")
        
        # Check if tokenizer can handle the format
        if detected_format == 'text':
            # Check if current tokenizer is IntegerTokenizer (which only handles integers)
            from data_utils.tokenizer import IntegerTokenizer
            if isinstance(self.tokenizer, IntegerTokenizer):
                print(f"\n" + "="*70)
                print("ERROR: JSONL contains text data, but using IntegerTokenizer!")
                print("="*70)
                print(f"\nYour JSONL file appears to contain text tokens (e.g., 'hello world'),")
                print(f"but you are using IntegerTokenizer which only handles integers.")
                print(f"\nTo fix this, you have two options:")
                print(f"\n1. Pass vocabulary file directly to prepare_data_from_jsonl():")
                print(f"   pipeline.prepare_data_from_jsonl(")
                print(f"       'data.jsonl',")
                print(f"       vocab_file_path='your_vocab.txt'")
                print(f"   )")
                print(f"\n2. Or initialize pipeline with GenericTokenizer:")
                print(f"   from rwkv_trainer import GenericTokenizer")
                print(f"   tokenizer = GenericTokenizer('your_vocab.txt')")
                print(f"   pipeline = RWKVTrainingPipeline(..., tokenizer=tokenizer)")
                print(f"\n3. If your data should be integers, check your JSONL format.")
                print(f"   Expected: {{'text': '1 2 3 4 5'}}")
                print(f"   Got:      {{'text': 'hello world'}}")
                print(f"\nSee examples/vocab_example.txt for vocabulary format.")
                print("="*70 + "\n")
                raise RuntimeError("Tokenizer mismatch: text data with IntegerTokenizer")
        
        # Try to process and catch tokenization errors
        n_epochs = n_epochs or self.data_config.n_epochs_duplication
        
        try:
            result = self.data_pipeline.process_jsonl(
                jsonl_path=jsonl_path,
                output_dir=self.data_dir,
                name=data_name,
                n_epochs=n_epochs
            )
        except ValueError as e:
            if "Cannot encode" in str(e):
                print(f"\n" + "="*70)
                print("ERROR: Tokenization failed!")
                print("="*70)
                print(f"\nThe tokenizer could not encode some tokens in your JSONL file.")
                print(f"Error: {e}")
                print(f"\nThis usually means:")
                print(f"1. Your vocabulary doesn't cover all tokens in the data")
                print(f"2. You're using the wrong tokenizer for your data type")
                print(f"\nSolutions:")
                print(f"- Pass vocab_file_path to prepare_data_from_jsonl()")
                print(f"- Or check that your vocab file contains all necessary tokens")
                print(f"\nSee examples/vocab_example.txt for vocabulary format.")
                print("="*70 + "\n")
                raise RuntimeError("Tokenization failed. See error message above.")
            raise
        
        self.data_prefix = result['binidx_prefix']
        self.magic_prime = result['magic_prime']
        self.my_exit_tokens = result['total_tokens']
        self.data_prepared = True
        
        print(f"### Data preparation complete:")
        print(f"    Total tokens: {self.my_exit_tokens}")
        print(f"    Magic prime: {self.magic_prime}")
        print(f"    Data prefix: {self.data_prefix}")
        
        return result
    
    def use_existing_data(self,
                          binidx_prefix: Union[str, Path],
                          magic_prime: int,
                          exit_tokens: int) -> None:
        """
        Use existing bin/idx data for training
        
        This allows you to train on pre-generated bin/idx files,
        useful for comparison tests or when data is prepared separately.
        
        Args:
            binidx_prefix: Path prefix to .bin and .idx files (without extension)
            magic_prime: Magic prime for data sampling
            exit_tokens: Total number of tokens in dataset
            
        Example:
            pipeline.use_existing_data(
                binidx_prefix="/path/to/train",
                magic_prime=1733,
                exit_tokens=111024
            )
            pipeline.train(num_epochs=10)
        """
        binidx_prefix = Path(binidx_prefix)
        bin_path = binidx_prefix.with_suffix('.bin')
        idx_path = binidx_prefix.with_suffix('.idx')
        
        if not bin_path.exists():
            raise FileNotFoundError(f"Bin file not found: {bin_path}")
        if not idx_path.exists():
            raise FileNotFoundError(f"Idx file not found: {idx_path}")
        
        self.data_prefix = str(binidx_prefix)
        self.magic_prime = magic_prime
        self.my_exit_tokens = exit_tokens
        self.data_prepared = True
        
        print(f"### Using existing data:")
        print(f"    Data prefix: {self.data_prefix}")
        print(f"    Magic prime: {self.magic_prime}")
        print(f"    Exit tokens: {self.my_exit_tokens}")
    
    def _get_train_script_path(self) -> Path:
        """Get path to train.py script"""
        package_dir = Path(__file__).parent.parent
        return package_dir / "train.py"
    
    def initialize_model(self, force: bool = False, random_seed: int = -1) -> Path:
        """
        Initialize model weights (Stage 1)
        
        Args:
            force: if True, reinitialize even if init file exists
            random_seed: random seed for model initialization (-1 for random)
            
        Returns:
            Path to initialized model
        """
        if not self.data_prepared:
            raise RuntimeError("Data must be prepared before model initialization")
        
        init_path = self.output_dir / "rwkv-init.pth"
        
        if init_path.exists() and not force:
            print(f"### Init model already exists: {init_path}")
            self.model_initialized = True
            return init_path
        
        print(f"### Initializing model (Stage 1)...")
        print(f"    Output: {self.output_dir}")
        print(f"    Vocab size: {self.model_config.vocab_size}")
        print(f"    Random seed: {random_seed}")
        
        # Build command
        cmd = self._build_train_command(stage=1, random_seed=random_seed)
        
        # Run initialization
        self._run_command(cmd, description="Model initialization")
        
        self.model_initialized = True
        print(f"### Model initialized: {init_path}")
        
        return init_path
    
    def train(self,
              data: Optional[np.ndarray] = None,
              num_epochs: int = 100,
              continue_training: bool = False,
              random_seed: int = -1) -> None:
        """
        Run complete training pipeline
        
        Args:
            data: numpy array of sequences (if None, use prepared data)
            num_epochs: number of training epochs
            continue_training: if True, continue from latest checkpoint in work_dir
            random_seed: random seed for training (-1 for random)
        """
        # Step 1: Prepare data if provided
        if data is not None:
            self.prepare_data(data)
        elif not self.data_prepared:
            raise RuntimeError("Either provide data or call prepare_data() first")
        
        # Step 2: Initialize model
        if not self.model_initialized and not continue_training:
            self.initialize_model(random_seed=random_seed)
        
        # Step 3: Train (Stage 3)
        print(f"### Starting training (Stage 3)...")
        print(f"    Epochs: {num_epochs}")
        print(f"    Output: {self.output_dir}")
        print(f"    Random seed: {random_seed}")
        
        cmd = self._build_train_command(stage=3, num_epochs=num_epochs, random_seed=random_seed, 
                                       continue_training=continue_training)
        
        self._run_command(cmd, description="Training")
        
        print(f"### Training complete!")
    
    def train_from_checkpoint(self,
                              checkpoint_path: Union[str, Path],
                              num_epochs: int = 100,
                              auto_detect: bool = True,
                              override_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Train from an external RWKV-LM checkpoint
        
        This method allows you to:
        1. Load a checkpoint from anywhere (auto-detect architecture)
        2. Override specific parameters if needed
        3. Continue training with your data
        
        Args:
            checkpoint_path: Path to external .pth file (RWKV-LM format)
            num_epochs: Number of epochs to train
            auto_detect: If True, automatically detect model architecture from checkpoint
            override_config: Dict of parameters to override (e.g., {'ctx_len': 2048})
            
        Raises:
            RuntimeError: If data not prepared or checkpoint incompatible
            
        Example:
            # Auto-detect everything
            pipeline.train_from_checkpoint("/path/to/rwkv-10.pth", num_epochs=100)
            
            # Override ctx_len
            pipeline.train_from_checkpoint(
                "/path/to/rwkv-10.pth",
                num_epochs=100,
                override_config={'ctx_len': 2048}
            )
        """
        if not self.data_prepared:
            raise RuntimeError("Data must be prepared before training")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"### Loading external checkpoint: {checkpoint_path}")
        
        # Import here to avoid circular imports
        from .model_utils import (
            inspect_rwkv_checkpoint,
            load_checkpoint_for_training,
            validate_checkpoint_compatibility
        )
        
        # Inspect checkpoint
        info = inspect_rwkv_checkpoint(str(checkpoint_path))
        print(f"### Checkpoint info:")
        print(f"    File size: {info['file_size_mb']:.1f} MB")
        print(f"    Parameters: {info['num_parameters']:,}")
        
        if auto_detect and 'detected_config' in info:
            detected = info['detected_config']
            print(f"### Auto-detected architecture:")
            print(f"    n_layer: {detected['n_layer']}")
            print(f"    n_embd: {detected['n_embd']}")
            print(f"    vocab_size: {detected['vocab_size']}")
            print(f"    model_type: {detected['model_type']}")
            
            # Update model config with detected values
            original_config = asdict(self.model_config)
            
            self.model_config.n_layer = detected['n_layer']
            self.model_config.n_embd = detected['n_embd']
            self.model_config.vocab_size = detected['vocab_size']
            self.model_config.model_type = detected['model_type']
            
            # Apply user overrides
            if override_config:
                print(f"### Applying user overrides:")
                for key, value in override_config.items():
                    if hasattr(self.model_config, key):
                        print(f"    {key}: {getattr(self.model_config, key)} -> {value}")
                        setattr(self.model_config, key, value)
                    else:
                        print(f"    Warning: Unknown config key '{key}'")
            
            # Validate compatibility
            if self.model_config.vocab_size != self.tokenizer.vocab_size:
                print(f"### ERROR: Vocab size mismatch!")
                print(f"    Checkpoint: {self.model_config.vocab_size}")
                print(f"    Your data/tokenizer: {self.tokenizer.vocab_size}")
                print(f"### You need to:")
                print(f"    1. Create a tokenizer with matching vocab_size, or")
                print(f"    2. Train with data that matches the checkpoint's vocab")
                raise RuntimeError("Vocab size mismatch")
            
            # Save updated config
            self._save_config()
            print(f"### Model config updated and saved")
        
        # Copy checkpoint to output directory for training
        target_path = self.output_dir / "rwkv-init.pth"
        print(f"### Copying checkpoint to: {target_path}")
        shutil.copy(checkpoint_path, target_path)
        
        # Set flag to indicate model is initialized
        self.model_initialized = True
        
        # Build training command with explicit checkpoint path
        print(f"### Starting training from checkpoint...")
        cmd = self._build_train_command(
            stage=3, 
            num_epochs=num_epochs,
            continue_training=False  # We manually copied the checkpoint
        )
        
        # Replace the load_model argument with our checkpoint path
        for i, arg in enumerate(cmd):
            if arg == "--load_model":
                cmd[i+1] = str(target_path)
                break
        
        self._run_command(cmd, description="Training from external checkpoint")
        print(f"### Training complete!")
    
    def inspect_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Inspect an external RWKV checkpoint without training
        
        Args:
            checkpoint_path: Path to .pth file
            
        Returns:
            Dictionary with checkpoint information
        """
        from .model_utils import inspect_rwkv_checkpoint
        return inspect_rwkv_checkpoint(str(checkpoint_path))
    
    def _build_train_command(self, 
                            stage: int,
                            num_epochs: int = 100,
                            continue_training: bool = False,
                            random_seed: int = -1) -> list:
        """Build training command"""
        m = self.model_config
        t = self.training_config
        
        script_path = self._get_train_script_path()
        
        cmd = [
            "python", str(script_path),
            "--wandb", self.wandb_project or "",
            "--proj_dir", str(self.output_dir),
            "--data_file", str(self.data_prefix),
            "--data_type", "binidx",
            "--vocab_size", str(m.vocab_size),
            "--my_testing", m.model_type,
            "--ctx_len", str(m.ctx_len),
            "--n_layer", str(m.n_layer),
            "--n_embd", str(m.n_embd),
            "--head_size_a", str(m.head_size_a),
            "--my_pile_stage", str(stage),
            "--epoch_count", str(num_epochs if stage == 3 else 1),
            "--epoch_save", str(t.epoch_save),
            "--micro_bsz", str(t.micro_bsz if stage == 3 else 1),
            "--lr_init", str(t.lr_init if stage == 3 else 1e-5),
            "--lr_final", str(t.lr_final if stage == 3 else 1e-5),
            "--warmup_steps", str(t.warmup_steps),
            "--beta1", str(t.beta1),
            "--beta2", str(t.beta2),
            "--adam_eps", str(t.adam_eps),
            "--weight_decay", str(t.weight_decay if stage == 3 else 0),
            "--grad_cp", str(t.grad_cp),
            "--precision", t.precision,
            "--strategy", t.strategy,
            "--random_seed", str(random_seed),
        ]
        
        if stage == 1:
            # Stage 1: initialization on CPU
            cmd.extend([
                "--epoch_begin", "0",
                "--my_exit_tokens", str(self.my_exit_tokens),
                "--magic_prime", str(self.magic_prime),
                "--accelerator", "cpu",
                "--devices", "1",
            ])
        else:
            # Stage 3: training on GPU
            cmd.extend([
                "--load_model", "0" if not continue_training else "",
                "--my_exit_tokens", str(self.my_exit_tokens),
                "--magic_prime", str(self.magic_prime),
                "--accelerator", "gpu",
                "--devices", "1",
                "--enable_progress_bar", "True",
            ])
        
        return cmd
    
    def _run_command(self, cmd: list, description: str = ""):
        """Run command with error handling"""
        print(f"### Running: {' '.join(cmd[:5])} ...")
        
        # Set environment variables for CUDA kernels
        env = os.environ.copy()
        env["RWKV_MY_TESTING"] = self.model_config.model_type
        env["RWKV_CTXLEN"] = str(self.model_config.ctx_len)
        env["RWKV_HEAD_SIZE_A"] = str(self.model_config.head_size_a)
        env["RWKV_JIT_ON"] = "1"
        
        try:
            result = subprocess.run(
                cmd, 
                check=True,
                env=env,
                cwd=str(self.work_dir)
            )
            return result
        except subprocess.CalledProcessError as e:
            print(f"### Error during {description}: {e}")
            raise
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint"""
        checkpoints = sorted(self.output_dir.glob("rwkv-*.pth"))
        if not checkpoints:
            return None
        return checkpoints[-1]
    
    def list_checkpoints(self) -> list:
        """List all checkpoints"""
        return sorted(self.output_dir.glob("rwkv-*.pth"))
