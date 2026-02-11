#!/bin/bash
# RWKV-LM vs rwkv_trainer 公平对比测试
# 使用完全相同的 bin/idx 数据文件

set -e

# 环境配置 - 使用 micromamba envs/rwkv
MICROMAMBA_ENV="/mnt/ssd2t/home/wty/micromamba/envs/rwkv"
PYTHON="$MICROMAMBA_ENV/bin/python"

DATA_DIR="/mnt/ssd2t/home/wty/research/vicsek_near_critical/rwkv_trainer/comparison_data"
ORIGINAL_DIR="/mnt/ssd2t/home/wty/research/vicsek_near_critical/rwkv_trainer/comparison_original"
TRAINER_DIR="/mnt/ssd2t/home/wty/research/vicsek_near_critical/rwkv_trainer/comparison_trainer"
RWKV_LM_PATH="/mnt/ssd2t/home/wty/lib/RWKV-LM/RWKV-v5"
RWKV_TRAINER_PATH="/mnt/ssd2t/home/wty/research/vicsek_near_critical/rwkv_trainer"

echo "Using Python: $PYTHON"
echo "Python version: $($PYTHON --version)"

# 统一配置
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
EPOCH_SAVE=1
NUM_EPOCHS=1

# 创建目录
mkdir -p $DATA_DIR
mkdir -p $ORIGINAL_DIR/out
mkdir -p $TRAINER_DIR

echo ""
echo "=========================================="
echo "Step 1: Generate test data using rwkv_trainer"
echo "=========================================="

cd $RWKV_TRAINER_PATH

$PYTHON << 'EOF'
import numpy as np
import sys
import os

# 设置路径
trainer_path = "/mnt/ssd2t/home/wty/research/vicsek_near_critical/rwkv_trainer"
sys.path.insert(0, trainer_path)
sys.path.insert(0, os.path.join(trainer_path, 'src'))

from data_utils.converter import DataPipeline
from data_utils.tokenizer import AngleTokenizer

# 固定随机种子生成数据
np.random.seed(42)
data = np.random.randint(0, 360, size=(1000, 64))

# 使用 rwkv_trainer 的 pipeline 准备数据
data_dir = "/mnt/ssd2t/home/wty/research/vicsek_near_critical/rwkv_trainer/comparison_data"
tokenizer = AngleTokenizer()
pipeline = DataPipeline(tokenizer)

result = pipeline.process_numpy(
    data=data,
    output_dir=data_dir,
    name="train",
    sequence_length=64,
    n_epochs=1
)

print(f"Data prepared:")
print(f"  JSONL: {result['jsonl_path']}")
print(f"  bin/idx prefix: {result['binidx_prefix']}")
print(f"  Total tokens: {result['total_tokens']}")
print(f"  Magic prime: {result['magic_prime']}")

# 保存 magic_prime 供脚本使用
with open(f"{data_dir}/magic_prime.txt", 'w') as f:
    f.write(str(result['magic_prime']))
EOF

# 读取 magic_prime
MAGIC_PRIME=$(cat $DATA_DIR/magic_prime.txt)
MY_EXIT_TOKENS=$(python3 -c "import numpy as np; data = np.random.randint(0, 360, size=(1000, 64)); print(len(data.flatten()))")

echo ""
echo "Data info:"
echo "  Magic prime: $MAGIC_PRIME"
echo "  Exit tokens: $MY_EXIT_TOKENS"
ls -lh $DATA_DIR/

echo ""
echo "=========================================="
echo "Step 2: Training with ORIGINAL RWKV-LM"
echo "=========================================="

# 必须在 RWKV-LM 目录下运行
cd $RWKV_LM_PATH

# Stage 1: 初始化模型
echo "Stage 1: Initialize model..."
$PYTHON train.py --wandb "" --proj_dir $ORIGINAL_DIR/out \
 --data_file $DATA_DIR/train --data_type "binidx" --vocab_size $VOCAB_SIZE --my_testing $MODEL_TYPE \
 --ctx_len $CTX_LEN --my_pile_stage 1 --epoch_count 1 --epoch_begin 0 \
 --epoch_save 1 --weight_decay 0 --head_size_a 64 \
 --num_nodes 1 --micro_bsz 1 --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 \
 --my_exit_tokens $MY_EXIT_TOKENS --magic_prime $MAGIC_PRIME \
 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --my_pile_edecay 0 \
 --accelerator cpu --devices 1 --precision $PRECISION --strategy $STRATEGY --grad_cp $GRAD_CP

# Stage 3: 训练
echo ""
echo "Stage 3: Training..."
$PYTHON train.py --load_model "0" --wandb "" --proj_dir $ORIGINAL_DIR/out --my_testing $MODEL_TYPE \
 --ctx_len $CTX_LEN --my_pile_stage 3 --epoch_count $NUM_EPOCHS --epoch_begin 0 \
 --data_file $DATA_DIR/train --my_exit_tokens $MY_EXIT_TOKENS --magic_prime $MAGIC_PRIME \
 --num_nodes 1 --micro_bsz $MICRO_BSZ --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 \
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps $WARMUP_STEPS \
 --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 --my_pile_edecay 0 \
 --data_type "binidx" --vocab_size $VOCAB_SIZE --weight_decay 0.001 --epoch_save $EPOCH_SAVE --head_size_a 64 \
 --accelerator gpu --devices 1 --precision $PRECISION --strategy $STRATEGY --grad_cp $GRAD_CP --enable_progress_bar True \
 2>&1 | tee $ORIGINAL_DIR/training_log.txt

echo ""
echo "Original RWKV-LM training complete!"
echo "Output: $ORIGINAL_DIR/out/"
ls -lh $ORIGINAL_DIR/out/

echo ""
echo "=========================================="
echo "Step 3: Training with rwkv_trainer"
echo "=========================================="

cd $RWKV_TRAINER_PATH

$PYTHON << EOF
import numpy as np
import sys
import os
import shutil

# 设置路径
trainer_path = "$RWKV_TRAINER_PATH"
sys.path.insert(0, trainer_path)
sys.path.insert(0, os.path.join(trainer_path, 'src'))

from trainer.pipeline import RWKVTrainingPipeline, ModelConfig, TrainingConfig, DataConfig

# 清理目录
trainer_dir = "$TRAINER_DIR"
if os.path.exists(trainer_dir):
    shutil.rmtree(trainer_dir)

# 使用相同的数据目录（已经准备好的 bin/idx）
data_dir = "$DATA_DIR"

# 创建 pipeline，指定已有数据
pipeline = RWKVTrainingPipeline(
    work_dir=trainer_dir,
    model_config=ModelConfig(
        model_type="$MODEL_TYPE",
        n_layer=$N_LAYER,
        n_embd=$N_EMBD,
        ctx_len=$CTX_LEN,
        vocab_size=$VOCAB_SIZE,
    ),
    training_config=TrainingConfig(
        lr_init=$LR_INIT,
        lr_final=$LR_FINAL,
        micro_bsz=$MICRO_BSZ,
        warmup_steps=$WARMUP_STEPS,
        precision="$PRECISION",
        strategy="$STRATEGY",
        grad_cp=$GRAD_CP,
        epoch_save=$EPOCH_SAVE,
    ),
    data_config=DataConfig(
        sequence_length=$CTX_LEN,
        n_epochs_duplication=1,
    )
)

# 使用已有的 bin/idx 数据
pipeline.use_existing_data(
    binidx_prefix=os.path.join(data_dir, 'train'),
    magic_prime=$MAGIC_PRIME,
    exit_tokens=$MY_EXIT_TOKENS
)

# 训练
pipeline.train(num_epochs=$NUM_EPOCHS)
print("rwkv_trainer training complete!")
EOF

echo ""
echo "rwkv_trainer training complete!"
echo "Output: $TRAINER_DIR/out/"
ls -lh $TRAINER_DIR/out/

echo ""
echo "=========================================="
echo "Step 4: Compare Results"
echo "=========================================="

echo ""
echo "Original RWKV-LM final checkpoint:"
ls -lh $ORIGINAL_DIR/out/rwkv-*.pth 2>/dev/null || echo "No checkpoint found"

echo ""
echo "rwkv_trainer final checkpoint:"
ls -lh $TRAINER_DIR/out/rwkv-*.pth 2>/dev/null || echo "No checkpoint found"

echo ""
echo "Compare loss values:"
echo "  RWKV-LM log: $ORIGINAL_DIR/training_log.txt"
echo "  rwkv_trainer log: $TRAINER_DIR/out/train_log.txt"

echo ""
echo "=========================================="
echo "Fair Comparison Test Complete!"
echo "=========================================="
