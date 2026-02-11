# RWKV Trainer - 项目记忆

## 项目目标
为 Vicsek 模型的角度数据训练 RWKV 语言模型，实现与 RWKV-LM 官方实现的 100% 兼容。

## 关键里程碑

### 2025-02-11: 严格对比测试通过 ✅
使用固定随机种子 (seed=42) 进行严格对比测试，验证与 RWKV-LM 的完全兼容性：

| 测试项目 | 结果 |
|---------|------|
| 数据管道 (bin/idx 生成) | ✅ 100% 一致 |
| 模型初始化权重 | ✅ 完全相同 |
| 训练 1 epoch 后权重 | ✅ 完全相同 |

**结论**: rwkv_trainer 与 RWKV-LM 在功能上完全等价，可互换使用。

### 已解决问题

1. **pydantic/deepspeed 兼容性**
   - 问题: deepspeed 与 pydantic v1 不兼容
   - 解决: 升级 pydantic 到 v2 (`pydantic>=2.0`)

2. **CUDA 路径问题**
   - 问题: JIT 编译 CUDA 内核时路径解析错误
   - 解决: 将相对路径 `cuda/` 改为绝对路径

3. **Tokenizer 编码**
   - 问题: 需要支持空格分隔的角度字符串 (如 "102 348 270")
   - 解决: 在 `encode()` 中检测空格分隔格式并处理

4. **随机种子传递**
   - 问题: 需要确保训练可复现
   - 解决: 在 pipeline 和 train 中添加 `random_seed` 参数

## 关键设计决策

### Token 映射约定
- Token 0: end_of_doc (空序列标记)
- Token 1-360: 角度 0-359
- 编码: `token_id = angle + 1`

### Magic Prime 计算
自动计算为小于 `(data_len / ctx_len - 1)` 的最大 3n+2 素数。

### 依赖版本
- pytorch-lightning==1.9.5 (RWKV-LM 要求)
- deepspeed>=0.9.0
- pydantic>=2.0

## 使用示例

```python
from rwkv_trainer import RWKVTrainingPipeline, AngleTokenizer

# 创建 tokenizer
tokenizer = AngleTokenizer()

# 创建训练管道
pipeline = RWKVTrainingPipeline(
    output_dir="./output",
    model_name="rwkv_vicsek",
    n_layer=6,
    n_embd=512,
    ctx_len=512,
    random_seed=42,  # 确保可复现
)

# 准备数据
sequences = [[0, 45, 90, 135], [180, 225, 270, 315]]
pipeline.prepare_training_data(sequences, tokenizer)

# 运行训练
pipeline.run_training(
    num_epochs=10,
    learning_rate=6e-4,
    batch_size=8,
)
```

## 注意事项

1. 使用 `--random_seed` 参数确保训练结果可复现
2. 首次运行时会 JIT 编译 CUDA 内核，可能需要几分钟
3. 确保数据目录有足够的空间存储 bin/idx 文件
