# MiniGPT

使用 PyTorch 从零实现的 Decoder-only Transformer 学习与工程验证项目。

项目包含两部分：**模型架构**——基础 GPT、RoPE/RMSNorm/SwiGLU、KV Cache 增量推理、MHA/GQA/MQA 统一注意力；
**训练工程**——Token 滑窗数据管线、训练/验证循环、Checkpoint 断点续训、Adapter-only 状态管理、
基于基础 MiniGPT Attention 的 LoRA Q/V 注入，以及同起点对照实验与 pytest 自动化验证。

**范围说明**：LoRA Q/V 注入的目标是 `model/mini_gpt.py` 中基础 MiniGPT 的 `CausalSelfAttention`，
不能据此推断 `cached_*`、`grouped_*`（KV Cache / GQA / MQA）等变体路径已具备同等的 LoRA 训练支持。
详见 [范围与局限](#范围与局限)。

## 当前状态

| 项目 | 值 |
|---|---|
| 分支 | `feat/m03d08-minigpt-v1-release` |
| 代码基线提交 | `f5e16a8` |
| 测试 | `python -m pytest tests/ -q` → **196 passed, 0 failed, 0 skipped**（7.04s）|
| 环境 | Windows 10 / CPU / Python 3.10.9 / PyTorch 2.6.0+cpu |
| 审计产物 | `docs/m03d08_release_audit.md`、`runs/m03d08_release_audit.json` |
| 最终报告 | [`docs/minigpt_v1_final.md`](docs/minigpt_v1_final.md) |

> 旧版 README 中的「`main` 分支 25 passed」是**仅含三阶段模型代码时**的历史数据；
> 当前分支已增加训练工程、LoRA 与实验模块，测试范围不同，不应与 196 混用或互相比较。

---

## 一、模型架构

### 阶段 1：基础 Decoder-only GPT

- `model/mini_gpt.py`：Token / Position Embedding → N × DecoderBlock → LayerNorm → LM Head
- `model/attention.py`：CausalSelfAttention，softmax **前**屏蔽未来位置；`model/decoder_block.py`：Pre-LN + 残差；`model/feed_forward.py`
- `tests/test_model.py`：前向 `[B, T, V]` 形状、反向传播、未来位置 attention 权重 ≈ 0、超长输入抛 `ValueError`

### 阶段 2：KV Cache 增量推理 + 现代 LLM 结构

- `model/cached_gpt.py` / `cached_attention.py` / `cached_block.py`：Prefill 建立各层缓存，Decode 只算新 token；RoPE 位置偏移随缓存长度推进
- `model/rope.py`（作用于 Q/K 的旋转位置编码）、`model/rms_norm.py`（不减均值的均方根归一化）、`model/swiglu.py`（门控 FFN）
- `tests/test_kv_cache.py`：缓存形状、单步 / 多步 decode、分块 prefill 与全序列推理同位置 logits 一致性、矩形 attention mask、越界与非法 token 校验

### 阶段 3：MHA / GQA / MQA 统一实现

- `model/grouped_*.py`：以 `num_q_heads` / `num_kv_heads` 参数化——`H_kv = H_q` 即 MHA，`1 < H_kv < H_q` 即 GQA，`H_kv = 1` 即 MQA（约束 `H_q % H_kv == 0`）
- `tests/test_gqa.py`：三种头配置下 cached vs full logits 一致性、分块 prefill、多步 decode、KV Cache 显存缩减验证、非法头数校验，以及 **MHA 配置与阶段 1 实现的输出对齐**

关键一致性结论：`model.eval()` 下，KV Cache 逐步推理与全序列推理在同位置 logits 的最大绝对误差约 `2e-7`（float32，RoPE offset 正确处理）。

## 二、训练工程

| 路径 | 功能 |
|---|---|
| `training/data.py` | Token 滑动窗口与 DataLoader 构建 |
| `training/loop.py` | next-token Loss、单步训练、token 加权验证、按步训练与历史记录 |
| `training/checkpoint.py` | 模型 / 优化器 / CPU RNG 状态的保存与断点恢复 |
| `training/adapter_io.py` | 冻结基础参数、Adapter-only 快照保存与兼容性校验 |
| `experiments/run_toy.py` | CPU 合成 Token 训练复现入口 |
| `experiments/compare_saved.py` | 两次实验结果逐项对照 |
| `experiments/reporting.py` | 原子写入的 JSON 实验报告 |
| `tools/m03d08_release_audit.py` | 发布证据审计：汇总测试与实验数值（最终收尾新增） |

设计要点：

- Loss 计算只展平 `B,T` 两维，`V` 保持在最后，`ignore_index=-100` 的 token 不参与平均
- 验证 Loss 按**有效 token 数加权**聚合，而非对 batch 做简单平均
- Checkpoint 保存走「同目录临时文件 + 原子替换」，`model_config` 深拷贝以保证快照独立
- 恢复前先校验参数/Buffer 名称、shape、dtype 与优化器参数组结构，**通过后才改模型**

## 三、LoRA 集成

`training/lora_integration.py`：

- `LoRALinear`：包装原始 `nn.Linear`，`y = Wx + (alpha/rank)·B(A(x))`，`A` / `B` 中一者零初始化，保证注入后初始输出与原始层完全一致
- `inject_lora_attention(model, targets=("W_q","W_v"), ...)`：**运行时注入**，遍历 `CausalSelfAttention` 把目标投影替换为 `LoRALinear`，不改动 `model/` 下的模型定义
- 基础权重全程 `requires_grad=False`，只有 `A` / `B` 参与训练；注入前校验 rank、alpha、dropout、targets 的配置类型与取值
- Merge：把 `B @ A * (alpha/rank)` 加回基础权重后还原为普通 `nn.Linear`，供推理使用

注入效果以实测为准（见下节对照实验）：8 个 Adapter 张量、初始输出差异 `0.0`、基础权重未变。

## 四、对照实验

全部实验使用**周期性合成 Token**，目的是验证训练工程与 LoRA 状态管理的正确性，**不代表任何真实语言任务效果**。

### Day 05：训练可复现性

同一配置独立运行两次，逐字段比对：

| 指标 | 值 |
|---|---:|
| Baseline Val Loss | 3.020242 |
| Final Val Loss | 2.686459 |
| 两次运行最大 Loss 差异 | 0.0 |

产物：`runs/day05_a.json`、`runs/day05_b.json`（SHA256 完全一致）。

### Day 07：全参数训练 vs LoRA 训练

两路实验共享**相同初始权重、相同 Batch 序列、相同优化器与学习率**，6 次更新，`rank=2`、`alpha=4.0`、targets 为 `W_q` / `W_v`：

| 指标 | 全参数训练 | LoRA 训练 |
|---|---:|---:|
| 训练前验证 Loss | 2.835182 | 2.835182 |
| 训练后验证 Loss | 2.638760 | 2.830074 |
| 可训练参数数 | 5,104 | 256 |
| 训练更新次数 | 6 | 6 |

验证证据：

| 验证目标 | 证据 |
|---|---|
| 训练工程正常 | Day 05 两次同配置实验最大 Loss 差异 `0` |
| LoRA 正确注入与更新 | 注入后初始输出差异 `0.0`；训练后 `base_unchanged=True`、`adapter_updated=True` |
| LoRA 恢复与合并正确 | Adapter 重载输出差异 `0.0`；Merge 后最大 logits 差异 `4.77e-7` |

可训练参数量减少 **94.98%**（5,104 → 256），**这是参数量对照，不是显存或吞吐测量**。
Adapter 文件大小 4,160 字节。产物：`runs/day07/comparison.json`。

## 五、运行

项目根目录执行：

```bash
python -m pip install -r requirements.txt        # torch>=2.2, pytest>=8

python -m pytest tests/ -q                       # 全量 196 passed
python -m pytest tests/ -q --durations=10 > runs/m03d08_pytest.txt

# 模型演示
python -m demos.demo_cached                      # cache vs full 一致性
python -m demos.demo_gqa                         # MQA 配置下的 KV 显存对比
python -m demos.demo_training_data
python -m demos.demo_training_loop
python -m demos.demo_checkpoint
python -m demos.demo_adapter_io
python -m demos.demo_lora_integration
python -m demos.demo_full_lora_comparison --steps 6

# 可复现实验
python -m experiments.run_toy --seed 42 --steps 12 --output runs/day05_a.json
python -m experiments.run_toy --seed 42 --steps 12 --output runs/day05_b.json
python -m experiments.compare_saved runs/day05_a.json runs/day05_b.json

# 发布证据审计
python -m tools.m03d08_release_audit
```

演示脚本需用 `python -m` 方式运行（`demos/` 为包，直接 `python demos/xxx.py` 会找不到 `model` 模块）。

## 目录结构

```text
MiniGPT/
├─ model/            # 三阶段模型：基础 GPT、cached_*、grouped_* 与 rope/rms_norm/swiglu
├─ training/         # 数据管线、训练循环、Checkpoint、Adapter IO、LoRA 注入
├─ experiments/      # 训练复现入口、结果对照与 JSON 报告
├─ tools/            # 发布证据审计工具
├─ tests/            # 全量 pytest 套件（196 项）
├─ demos/            # 可运行演示脚本
├─ docs/             # 阶段历史输出存档、审计小结、最终报告
├─ runs/             # 实验产物与测试日志（小型 JSON / 文本摘要）
├─ conftest.py       # 使仓库根目录可被测试导入
└─ requirements.txt
```

## 六、验证方式

- **分层测试**：从张量形状/数值正确性 → 训练循环 → 断点恢复 → LoRA 注入与合并，逐层断言
- **确定性对照**：固定 seed 与 Batch 序列，两次独立运行产物完全一致
- **独立重建验证**：恢复 Adapter 时从训练前保存的基础权重副本重新注入再加载 A/B 文件，避免把「复制训练后模型」误判为恢复成功
- **审计脚本**：`tools/m03d08_release_audit.py` 汇总测试日志与实验 JSON，输出 `ready` 判定与 `docs/m03d08_release_audit.md`

## 范围与局限

- **LoRA 覆盖范围有限**：仅对基础 MiniGPT 的 `CausalSelfAttention` 的 `W_q` / `W_v` 实现注入；`cached_*`、`grouped_*(GQA/MQA)` 路径未实现同等的 LoRA 训练支持
- **合成数据**：所有实验基于周期性合成 Token，随机初始化的小模型，**不代表真实语言任务效果**，不构成 SFT 质量结论
- **参数量 ≠ 资源节省**：94.98% 是可训练参数量比例，未测量 GPU 显存峰值、吞吐或端到端延迟
- **训练步数极少**：6 次更新的对照只能验证流程与数值一致性，不足以判断两种策略在一般任务上的优劣
- **Checkpoint 覆盖面**：RNG 恢复主要覆盖 CPU，多设备、多进程与采样器状态未保证完全复现
- **Adapter 依赖基础权重**：`base_model_id` 是字符串标识，不是权重内容的哈希
- **教学规模**：`d_model` 16–128、1–3 层，未做推理性能优化（如缓存 `torch.cat` 的拷贝开销、GQA 分组广播可进一步向量化）

安全与工程注意：

- 不提交 API Token、访问密钥、私人数据或大模型权重
- `runs/` 仅保留小型、可重复生成、不含敏感信息的 JSON / 文本摘要（`*.pt`、`*.bin` 等权重文件已在 `.gitignore` 中排除）
- 真实 GPU 训练、推理服务部署与自然语言 SFT 属于后续独立项目范围

## 后续计划

V1 完成后不再在该仓库扩张功能。下一阶段另建项目，使用 Transformers / PEFT 与真实数据集做 SFT/PEFT 对照与可重复评测，验证自然语言能力。

## 参考

- [nanoGPT](https://github.com/karpathy/nanoGPT)（karpathy）
- RoPE: Su et al., *RoFormer*, 2021；RMSNorm: Zhang & Sennrich, 2019；SwiGLU: Shazeer, 2020；MQA: Shazeer, 2019；GQA: Ainslie et al., 2023
- LoRA: Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021
