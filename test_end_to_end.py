#!/usr/bin/env python3
"""
端到端训练测试脚本
使用随机生成的角度数据训练一个微型 RWKV 模型
"""

import numpy as np
import sys
import os

# 添加 src 到路径（本地测试时）
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)
print(f"Using source from: {src_path}")

from trainer.pipeline import RWKVTrainingPipeline, ModelConfig, TrainingConfig, DataConfig

def generate_test_data(n_samples=1000, seq_len=128):
    """生成测试用的角度序列数据（模拟 Vicsek 模型）"""
    print(f"Generating test data: {n_samples} samples, seq_len={seq_len}")
    
    # 生成随机角度 0-359
    data = np.random.randint(0, 360, size=(n_samples, seq_len))
    return data

def test_full_training():
    """测试完整训练流程"""
    print("=" * 60)
    print("RWKV Trainer End-to-End Test")
    print("=" * 60)
    
    # 1. 准备测试数据
    data = generate_test_data(n_samples=500, seq_len=64)
    print(f"Data shape: {data.shape}, dtype: {data.dtype}")
    print(f"Data range: [{data.min()}, {data.max()}]")
    
    # 2. 创建 pipeline，使用超小模型配置
    print("\nCreating training pipeline...")
    work_dir = "./test_experiment"
    
    # 清理旧实验目录
    import shutil
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
        print(f"Cleaned old experiment dir: {work_dir}")
    
    pipeline = RWKVTrainingPipeline(
        work_dir=work_dir,
        model_config=ModelConfig(
            model_type="x052",  # RWKV-5
            n_layer=2,          # 超小模型：2层
            n_embd=64,          # 超小模型：64维
            ctx_len=64,         # 上下文长度
            vocab_size=360      # 角度 0-359
        ),
        training_config=TrainingConfig(
            lr_init=6e-4,
            lr_final=1e-5,
            micro_bsz=4,        # 小 batch
            epoch_save=1,       # 每 epoch 保存
        ),
        data_config=DataConfig(
            sequence_length=64,
            n_epochs_duplication=1
        )
    )
    print("Pipeline created successfully!")
    
    # 3. 训练模型
    print("\nStarting training...")
    print("-" * 60)
    
    try:
        pipeline.train(
            data=data,
            num_epochs=2       # 只跑 2 个 epoch（测试用）
        )
        print("-" * 60)
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 验证输出文件
    print("\nVerifying output files...")
    expected_files = [
        "out/rwkv-init.pth",
        "out/rwkv-0.pth",
        "out/rwkv-1.pth",
    ]
    
    all_exist = True
    for f in expected_files:
        path = os.path.join(work_dir, f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✅ {f} ({size:,} bytes)")
        else:
            print(f"  ❌ {f} NOT FOUND")
            all_exist = False
    
    # 5. 验证数据文件
    data_files = ["data/train.bin", "data/train.idx"]
    print("\nData files:")
    for f in data_files:
        path = os.path.join(work_dir, f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✅ {f} ({size:,} bytes)")
        else:
            print(f"  ❌ {f} NOT FOUND")
            all_exist = False
    
    print("\n" + "=" * 60)
    if all_exist:
        print("🎉 All tests PASSED!")
        print(f"Experiment saved to: {os.path.abspath(work_dir)}")
    else:
        print("⚠️ Some files missing")
    print("=" * 60)
    
    return all_exist

if __name__ == "__main__":
    success = test_full_training()
    sys.exit(0 if success else 1)
