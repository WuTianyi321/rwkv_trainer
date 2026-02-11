#!/bin/bash
# RWKV-LM vs rwkv_trainer 训练对比测试
# 确保两者使用相同的数据和配置

set -e

# 环境配置 - 使用 micromamba envs/rwkv
MICROMAMBA_ENV="/mnt/ssd2t/home/wty/micromamba/envs/rwkv"
PYTHON="$MICROMAMBA_ENV/bin/python"

DATA_DIR="/mnt/ssd2t/home/wty/research/vicsek_near_critical/rwkv_trainer/comparison_data"
ORIGINAL_DIR="/mnt/ssd2t/home/wty/research/vicsek_near_critical/rwkv_trainer/comparison_original"
TRAINER_DIR="/mnt/ssd2t/home/wty/research/vicsek_near_critical/rwkv_trainer/comparison_trainer"
RWKV_LM_PATH="/mnt/ssd2t/home/wty/lib/RWKV-LM/RWKV-v5"

echo "Using Python: $PYTHON"
echo "Python version: $($PYTHON --version)"

# 配置
N_LAYER=2
N_EMBD=64
CTX_LEN=64
VOCAB_SIZE=361
MODEL_TYPE="x052"
LR_INIT="6e-4"
LR_FINAL="1e-5"
MICRO_BSZ=4
WARMUP_STEPS=10
PRECISION="bf16"
STRATEGY="deepspeed_stage_1"
GRAD_CP=1

# 计算 magic_prime (111024 tokens / 64 - 1, 最大 3n+2 质数 = 1733)
MAGIC_PRIME=1733
MY_EXIT_TOKENS=111024

echo "=========================================="
echo "RWKV-LM vs rwkv_trainer Comparison Test"
echo "=========================================="

# 1. 创建数据目录
mkdir -p $DATA_DIR
mkdir -p $ORIGINAL_DIR/out
mkdir -p $TRAINER_DIR

# 2. 生成测试数据 (使用 Python)
echo ""
echo "Step 1: Generating test data..."
$PYTHON << 'EOF'
import numpy as np
import json
import os

np.random.seed(42)
data_dir = "/mnt/ssd2t/home/wty/research/vicsek_near_critical/rwkv_trainer/comparison_data"

# 生成 1000 个序列，每个 64 个角度
data = np.random.randint(0, 360, size=(1000, 64))

# 保存为 JSONL (格式: {"text": "0 45 90 ..."})
with open(f"{data_dir}/test_data.jsonl", 'w') as f:
    for seq in data:
        text = ' '.join(map(str, seq))
        f.write(json.dumps({"text": text}) + '\n')

print(f"Generated {len(data)} sequences")
print(f"Saved to {data_dir}/test_data.jsonl")
EOF

# 3. 使用 RWKV-LM 的 make_data.py 生成 bin/idx
echo ""
echo "Step 2: Preparing bin/idx data using RWKV-LM make_data.py..."
cd $RWKV_LM_PATH
$PYTHON make_data.py $DATA_DIR/test_data.jsonl 1 64

# 复制生成的文件到数据目录
cp $RWKV_LM_PATH/test_data.bin $DATA_DIR/
cp $RWKV_LM_PATH/test_data.idx $DATA_DIR/

echo ""
echo "Data prepared:"
ls -lh $DATA_DIR/

# 4. 使用原始 RWKV-LM 训练
echo ""
echo "=========================================="
echo "Step 3: Training with ORIGINAL RWKV-LM"
echo "=========================================="

# Stage 1: 初始化模型
echo "Stage 1: Initialize model..."
$PYTHON $RWKV_LM_PATH/train.py --wandb "" --proj_dir $ORIGINAL_DIR/out \
 --data_file $DATA_DIR/test_data --data_type "binidx" --vocab_size $VOCAB_SIZE --my_testing $MODEL_TYPE \
 --ctx_len $CTX_LEN --my_pile_stage 1 --epoch_count 1 --epoch_begin 0 \
 --epoch_save 1 --weight_decay 0 --head_size_a 64 \
 --num_nodes 1 --micro_bsz 1 --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 \
 --my_exit_tokens $MY_EXIT_TOKENS --magic_prime $MAGIC_PRIME \
 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --my_pile_edecay 0 \
 --accelerator cpu --devices 1 --precision $PRECISION --strategy $STRATEGY --grad_cp $GRAD_CP

# Stage 3: 训练
echo ""
echo "Stage 3: Training..."
$PYTHON $RWKV_LM_PATH/train.py --load_model "0" --wandb "" --proj_dir $ORIGINAL_DIR/out --my_testing $MODEL_TYPE \
 --ctx_len $CTX_LEN --my_pile_stage 3 --epoch_count 1 --epoch_begin 0 \
 --data_file $DATA_DIR/test_data --my_exit_tokens $MY_EXIT_TOKENS --magic_prime $MAGIC_PRIME \
 --num_nodes 1 --micro_bsz $MICRO_BSZ --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 \
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps $WARMUP_STEPS \
 --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 --my_pile_edecay 0 \
 --data_type "binidx" --vocab_size $VOCAB_SIZE --weight_decay 0.001 --epoch_save 1 --head_size_a 64 \
 --accelerator gpu --devices 1 --precision $PRECISION --strategy $STRATEGY --grad_cp $GRAD_CP --enable_progress_bar True \
 2>&1 | tee $ORIGINAL_DIR/training_log.txt

echo ""
echo "Original RWKV-LM training complete!"
echo "Output: $ORIGINAL_DIR/out/"
ls -lh $ORIGINAL_DIR/out/

# 5. 使用 rwkv_trainer 训练
echo ""
echo "=========================================="
echo "Step 4: Training with rwkv_trainer"
echo "=========================================="

cd /mnt/ssd2t/home/wty/research/vicsek_near_critical/rwkv_trainer

$PYTHON << 'EOF'
import numpy as np
import sys
import os
import shutil

# 设置路径 - 使用 rwkv_trainer 的 src
trainer_path = "/mnt/ssd2t/home/wty/research/vicsek_near_critical/rwkv_trainer"
src_path = os.path.join(trainer_path, 'src')
sys.path.insert(0, src_path)
sys.path.insert(0, trainer_path)

from trainer.pipeline import RWKVTrainingPipeline, ModelConfig, TrainingConfig, DataConfig

# 固定随机种子
np.random.seed(42)
data = np.random.randint(0, 360, size=(1000, 64))

trainer_dir = "/mnt/ssd2t/home/wty/research/vicsek_near_critical/rwkv_trainer/comparison_trainer"

# 清理目录
if os.path.exists(trainer_dir):
    shutil.rmtree(trainer_dir)

pipeline = RWKVTrainingPipeline(
    work_dir=trainer_dir,
    model_config=ModelConfig(
        model_type="x052",
        n_layer=2,
        n_embd=64,
        ctx_len=64,
        vocab_size=361,
    ),
    training_config=TrainingConfig(
        lr_init=6e-4,
        lr_final=1e-5,
        micro_bsz=4,
        warmup_steps=10,
        precision="bf16",
        strategy="deepspeed_stage_1",
        grad_cp=1,
        epoch_save=1,
    ),
    data_config=DataConfig(
        sequence_length=64,
        n_epochs_duplication=1,
    )
)

print(f"Data shape: {data.shape}")
pipeline.train(data, num_epochs=1)
print("rwkv_trainer training complete!")
EOF

echo ""
echo "rwkv_trainer training complete!"
echo "Output: $TRAINER_DIR/out/"
ls -lh $TRAINER_DIR/out/

# 6. 对比结果
echo ""
echo "=========================================="
echo "Comparison Summary"
echo "=========================================="
echo ""
echo "Original RWKV-LM:"
echo "  Log: $ORIGINAL_DIR/training_log.txt"
echo "  Model: $ORIGINAL_DIR/out/"
echo ""
echo "rwkv_trainer:"
echo "  Log: $TRAINER_DIR/out/train_log.txt"
echo "  Model: $TRAINER_DIR/out/"
echo ""
echo "Compare the loss values from both logs!"
