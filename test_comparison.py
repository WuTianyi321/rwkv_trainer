#!/usr/bin/env python3
"""
RWKV-LM vs rwkv_trainer 训练结果对比测试
确保两者使用相同的数据和配置，比较 loss 曲线
"""

import numpy as np
import sys
import os
import json
import shutil

# 设置固定随机种子确保可重复性
np.random.seed(42)

# 添加 src 到路径
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from trainer.pipeline import RWKVTrainingPipeline, ModelConfig, TrainingConfig, DataConfig

# 测试配置（与原始 RWKV-LM 保持一致）
CONFIG = {
    'n_layer': 2,
    'n_embd': 64,
    'ctx_len': 64,
    'vocab_size': 361,  # 0-360，其中 0 是 end_of_doc
    'model_type': 'x052',
    'lr_init': 6e-4,
    'lr_final': 1e-5,
    'micro_bsz': 4,
    'warmup_steps': 10,
    'precision': 'bf16',
    'strategy': 'deepspeed_stage_1',
    'grad_cp': 1,
}

def generate_fixed_data(n_samples=1000, seq_len=64):
    """生成固定的测试数据（使用固定种子）"""
    np.random.seed(42)
    # 生成 0-359 的角度数据
    data = np.random.randint(0, 360, size=(n_samples, seq_len))
    return data

def save_data_for_original_rwkv(data, output_dir):
    """保存数据为原始 RWKV-LM 可用的格式"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为 JSONL
    jsonl_path = os.path.join(output_dir, 'test_data.jsonl')
    with open(jsonl_path, 'w') as f:
        for seq in data:
            # 格式: {"text": "0 45 90 ..."}
            text = ' '.join(map(str, seq))
            f.write(json.dumps({"text": text}) + '\n')
    
    return jsonl_path

def test_rwkv_trainer(data, work_dir):
    """使用 rwkv_trainer 训练"""
    print("\n" + "="*70)
    print("Testing rwkv_trainer")
    print("="*70)
    
    # 清理目录
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    
    pipeline = RWKVTrainingPipeline(
        work_dir=work_dir,
        model_config=ModelConfig(
            model_type=CONFIG['model_type'],
            n_layer=CONFIG['n_layer'],
            n_embd=CONFIG['n_embd'],
            ctx_len=CONFIG['ctx_len'],
            vocab_size=CONFIG['vocab_size'],
        ),
        training_config=TrainingConfig(
            lr_init=CONFIG['lr_init'],
            lr_final=CONFIG['lr_final'],
            micro_bsz=CONFIG['micro_bsz'],
            warmup_steps=CONFIG['warmup_steps'],
            precision=CONFIG['precision'],
            strategy=CONFIG['strategy'],
            grad_cp=CONFIG['grad_cp'],
            epoch_save=1,
        ),
        data_config=DataConfig(
            sequence_length=CONFIG['ctx_len'],
            n_epochs_duplication=1,
        )
    )
    
    # 训练 1 个 epoch，记录 loss
    print(f"Data shape: {data.shape}")
    pipeline.train(data, num_epochs=1)
    
    # 读取训练日志
    log_path = os.path.join(work_dir, 'out', 'train_log.txt')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            print("\nTrainer log:")
            print(f.read()[:2000])
    
    return work_dir

def create_original_rwkv_script(data_dir, work_dir):
    """创建原始 RWKV-LM 的训练脚本"""
    script_path = os.path.join(work_dir, 'run_original.sh')
    
    # 计算 magic_prime (datalen/ctx_len - 1 以下的最大 3n+2 质数)
    # 1000 个序列，每个 64 tokens = 64000 tokens
    # 64000 / 64 - 1 = 999
    # 质数 997 = 3*332 + 1 (不符合)
    # 质数 991 = 3*330 + 1 (不符合)
    # 质数 983 = 3*327 + 2 (符合！)
    magic_prime = 983
    
    # 先创建初始化脚本
    init_script = f'''#!/bin/bash
MODEL_TYPE="{CONFIG['model_type']}"
N_LAYER="{CONFIG['n_layer']}"
N_EMBD="{CONFIG['n_embd']}"
CTX_LEN="{CONFIG['ctx_len']}"
PROJ_DIR="{work_dir}/out"
MY_EXIT_TOKENS="64000"
MAGIC_PRIME="{magic_prime}"
DATA_FILE="{data_dir}/test_data"

echo "Step 1: Initialize model..."
python /home/wty/lib/RWKV-LM/RWKV-v5/train.py --wandb "" --proj_dir $PROJ_DIR \\
 --data_file $DATA_FILE --data_type "binidx" --vocab_size {CONFIG['vocab_size']} --my_testing $MODEL_TYPE \\
 --ctx_len $CTX_LEN --my_pile_stage 1 --epoch_count 1 --epoch_begin 0 \\
 --epoch_save 1 --weight_decay 0 --head_size_a 64 \\
 --num_nodes 1 --micro_bsz 1 --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 --my_exit_tokens $MY_EXIT_TOKENS --magic_prime $MAGIC_PRIME \\
 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --my_pile_edecay 0 \\
 --accelerator cpu --devices 1 --precision {CONFIG['precision']} --strategy {CONFIG['strategy']} --grad_cp {CONFIG['grad_cp']}

echo "Step 2: Training..."
python /home/wty/lib/RWKV-LM/RWKV-v5/train.py --load_model "0" --wandb "" --proj_dir $PROJ_DIR --my_testing $MODEL_TYPE \\
 --ctx_len $CTX_LEN --my_pile_stage 3 --epoch_count 1 --epoch_begin 0 \\
 --data_file $DATA_FILE --my_exit_tokens $MY_EXIT_TOKENS --magic_prime $MAGIC_PRIME \\
 --num_nodes 1 --micro_bsz {CONFIG['micro_bsz']} --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 \\
 --lr_init {CONFIG['lr_init']} --lr_final {CONFIG['lr_final']} --warmup_steps {CONFIG['warmup_steps']} --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 --my_pile_edecay 0 --data_type "binidx" --vocab_size {CONFIG['vocab_size']} \\
 --weight_decay 0.001 --epoch_save 1 --head_size_a 64 \\
 --accelerator gpu --devices 1 --precision {CONFIG['precision']} --strategy {CONFIG['strategy']} --grad_cp {CONFIG['grad_cp']} --enable_progress_bar True
'''
    
    with open(script_path, 'w') as f:
        f.write(init_script)
    os.chmod(script_path, 0o755)
    
    return script_path

def prepare_data_for_original(data_dir):
    """为原始 RWKV-LM 准备 bin/idx 数据 - 使用 numpy 直接转换"""
    # 使用 numpy 数据直接创建 bin/idx
    from data_utils.converter import DataPipeline
    from data_utils.tokenizer import AngleTokenizer
    import numpy as np
    
    # 重新加载数据
    np.random.seed(42)
    data = np.random.randint(0, 360, size=(1000, 64))
    
    tokenizer = AngleTokenizer()
    pipeline = DataPipeline(tokenizer)
    
    # 使用 process_numpy 方法
    result = pipeline.process_numpy(
        data=data,
        output_dir=data_dir,
        name="test_data",
        sequence_length=64,
        n_epochs=1
    )
    
    return str(result['binidx_prefix'])

def run_comparison():
    """运行对比测试"""
    print("="*70)
    print("RWKV-LM vs rwkv_trainer Training Comparison")
    print("="*70)
    
    # 1. 生成固定数据
    data = generate_fixed_data(n_samples=1000, seq_len=64)
    print(f"\nGenerated test data: {data.shape}")
    
    # 2. 保存数据
    data_dir = "/mnt/ssd2t/home/wty/research/vicsek_near_critical/rwkv_trainer/comparison_data"
    os.makedirs(data_dir, exist_ok=True)
    
    jsonl_path = save_data_for_original_rwkv(data, data_dir)
    print(f"Data saved to: {jsonl_path}")
    
    # 3. 准备原始 RWKV-LM 的数据格式 (bin/idx)
    print("\nPreparing data for original RWKV-LM...")
    bin_prefix = prepare_data_for_original(data_dir)
    print(f"bin/idx created: {bin_prefix}")
    
    # 4. 测试 rwkv_trainer
    trainer_dir = "/mnt/ssd2t/home/wty/research/vicsek_near_critical/rwkv_trainer/comparison_trainer"
    test_rwkv_trainer(data, trainer_dir)
    
    # 5. 创建原始 RWKV-LM 的脚本
    original_dir = "/mnt/ssd2t/home/wty/research/vicsek_near_critical/rwkv_trainer/comparison_original"
    os.makedirs(original_dir, exist_ok=True)
    
    script_path = create_original_rwkv_script(data_dir, original_dir)
    print(f"\n{'='*70}")
    print("Original RWKV-LM script created:")
    print(f"  {script_path}")
    print(f"\nTo run original RWKV-LM comparison:")
    print(f"  cd {os.path.dirname(script_path)}")
    print(f"  source {script_path}")
    print("="*70)
    
    # 6. 打印总结
    print("\n" + "="*70)
    print("Comparison Setup Complete")
    print("="*70)
    print(f"\nrwkv_trainer output: {trainer_dir}/out/")
    print(f"Original RWKV-LM output: {original_dir}/out/")
    print(f"\nBoth use the same:")
    print(f"  - Data: {data_dir}/")
    print(f"  - Model: {CONFIG['n_layer']}L-{CONFIG['n_embd']}D-{CONFIG['model_type']}")
    print(f"  - Config: lr={CONFIG['lr_init']}-{CONFIG['lr_final']}, bsz={CONFIG['micro_bsz']}")
    print("\nCompare the loss values in both training logs!")

if __name__ == "__main__":
    run_comparison()
