# MiniGPT V1 最终工程报告

> 课程编号：month 03 day 06；Git 任务标识：m03d08。
> 代码基线提交：`f5e16a8`；分支：`feat/m03d08-minigpt-v1-release`。
> 本报告中的全部数值均来自该基线上**当日复跑**的测试日志与实验产物，不是历史记忆值。

## 一、项目定位

使用 PyTorch 从零构建 Decoder-only Transformer，并把数据准备、训练验证、状态管理、LoRA 注入与合成数据对照实验连接为一个可测的工程项目。

项目的目标是**学习并证明各技术组件的正确性**，而不是宣称具备真实语言 SFT 性能。所有实验基于随机初始化的小模型与周期性合成 Token，结论仅限于「训练工程可用、LoRA 状态管理正确、同环境可复现」。

## 二、实现范围与证据

| 功能 | 位置 | 验证证据 |
|---|---|---|
| Decoder-only Transformer | `model/mini_gpt.py`、`attention.py`、`decoder_block.py`、`feed_forward.py` | `tests/test_model.py`：前向 `[B,T,V]` 形状、反向传播、未来位置权重≈0、超长输入抛错 |
| RoPE / RMSNorm / SwiGLU | `model/rope.py`、`rms_norm.py`、`swiglu.py` | `tests/test_kv_cache.py`、`tests/test_gqa.py` 中的位置偏移与归一化数值断言；`python -m demos.demo_cached` |
| KV Cache / MHA-GQA-MQA | `model/cached_*.py`、`model/grouped_*.py` | `tests/test_kv_cache.py`：分块 prefill 与全序列同位置 logits 一致性；`tests/test_gqa.py`：三种头配置一致性 + MHA 对齐阶段 1；误差约 `2e-7` |
| 滑窗数据与训练验证 | `training/data.py`、`training/loop.py` | `tests/test_training_data.py`、`tests/test_training_loop.py`：滑窗边界、ignore_index、token 加权验证、按步训练 |
| Checkpoint 恢复 | `training/checkpoint.py` | `tests/test_checkpoint.py`：保存/加载往返、严格校验、RNG 恢复与可关闭、原子写入无残留 `.tmp` |
| Adapter 保存与重载 | `training/adapter_io.py` | `tests/test_adapter_io.py`：冻结参数、Adapter-only 快照、metadata 类型与值的严格比对 |
| LoRA Q/V 实际注入 | `training/lora_integration.py` | `tests/test_lora_integration.py`：注入后输出不变、基础权重冻结、仅 A/B 更新、重复注入与非法配置报错 |
| 同起点全参 / LoRA 对照 | `experiments/compare_full_lora.py` | `runs/day07/comparison.json`、`tests/test_full_lora_comparison.py` |
| LoRA Merge | `merge_lora_for_inference()`（同文件） | Merge 前后 eval 模式 logits 最大差异 `4.77e-7`；原始模型不被修改 |
| 发布证据审计 | `tools/m03d08_release_audit.py` | `docs/m03d08_release_audit.md`、`runs/m03d08_release_audit.json` |

## 三、复现环境与操作

| 项目 | 值 |
|---|---|
| Git 分支 | `feat/m03d08-minigpt-v1-release` |
| 代码基线 SHA | `f5e16a8` |
| Python | 3.10.9 |
| PyTorch | 2.6.0+cpu |
| 设备 / OS | CPU / Windows 10 (10.0.19045) |

```bash
python -m pip install -r requirements.txt

# 全量测试
python -m pytest tests/ -q --durations=10 > runs/m03d08_pytest.txt

# 对照实验（生成 runs/day07/comparison.json）
python -m demos.demo_full_lora_comparison --steps 6

# 发布审计（汇总测试与实验证据）
python -m tools.m03d08_release_audit
```

## 四、实验结果

### 4.1 测试

| 指标 | 值 |
|---|---:|
| 通过 | 204 |
| 失败 | 0 |
| 跳过 | 0 |
| 耗时 | 6.87 秒 |

日志：`runs/m03d08_pytest.txt`（本次共 204 项，含本轮新增的 8 项回归测试；早前一次记录为 19.28s，属机器负载差异，不影响通过数）。

### 4.2 Day 07：全参数训练 vs LoRA 训练

两路实验共享相同初始权重、相同 Batch 序列、相同优化器种类与学习率，6 次更新；`rank=2`、`alpha=4.0`、`lora_dropout=0.0`、`target_modules=("W_q","W_v")`。

| 指标 | 全参数 | LoRA |
|---|---:|---:|
| 训练前验证 Loss | 2.835182 | 2.835182 |
| 训练后验证 Loss | 2.638760 | 2.830074 |
| 可训练参数数 | 5,104 | 256 |
| 模型总参数数 | 5,104 | 5,360 |
| 训练更新次数 | 6 | 6 |

- 可训练参数减少 **94.98%**（5,104 → 256）
- 初始输出差异：`0.0`
- Adapter 重新加载输出差异：`0.0`
- Merge 后最大 logits 差异：`4.76837158203125e-07`
- 基础权重不变：`True`；Adapter 已更新：`True`
- Adapter 文件大小：`4,160` 字节

产物：`runs/day07/comparison.json`。

### 4.3 Day 05：训练可复现性

同一配置独立运行两次：Baseline Val Loss `3.020242` → Final Val Loss `2.686459`，两次运行最大 Loss 差异 `0`，两份 JSON 的 SHA256 完全一致。

产物：`runs/day05_a.json`、`runs/day05_b.json`。

### 4.4 三组核心证据

| 验证目标 | 已有证据 | 结论 |
|---|---|---|
| 训练工程正常 | Day 05 两次同配置实验 Loss 差异为 `0` | 同环境结果可复现 |
| LoRA 正确注入和更新 | Day 06 初始输出差异 `0.0`，基础权重不变 `True` | 注入不改变原模型行为，仅 Adapter 更新 |
| LoRA 恢复与合并正确 | Day 07 恢复差异 `0.0`，Merge 误差约 `4.77e-7` | 状态管理与合并数值正确 |

## 五、正确性与公平性说明

- 两路实验共享初始基础权重、数据集、样本顺序、优化器种类、训练步数与学习率；**模型可训练参数规模本身不同**，这是被比较方法的核心区别，不视为不公平。
- Adapter 重载时，从**训练前保存的基础权重副本**重新注入 LoRA 再加载 A/B 文件，避免把「复制训练后模型」误判为恢复成功。
- Merge 在 `copy.deepcopy` 出的独立副本中完成，且要求 eval 模式、`torch.no_grad()`、dtype/device 一致，不修改原始 Adapter 模型。
- 本实验未评估跨数据集、跨模型、多随机种子的统计显著性，也没有真实语言任务指标。

## 六、限制与不足

- 只有周期性合成数据与随机初始化小模型，**不代表真实语言任务效果**，不存在可支撑真实 SFT 指标的证据。
- LoRA 仅针对基础 MiniGPT `CausalSelfAttention` 的 `W_q` / `W_v` 实现；`cached_*` 与 `grouped_*`（GQA/MQA）路径**未实现**同等 LoRA 训练支持。
- 94.98% 是**可训练参数量**比例，未测量 GPU 显存峰值、吞吐、端到端延迟，不能表述为「显存降低 95%」。
- 6 次更新不足以判断全参与 LoRA 在一般任务上的优劣；本对比只证明两条训练链路都能正确工作。
- 教学模型规模小（`d_model` 16–128、1–3 层），无推理性能优化。
- Checkpoint 的 RNG 恢复主要覆盖 CPU；多设备、多进程与 DataLoader 采样器状态未保证完全复现。

## 七、已知问题与修复记录

| 项 | 内容 | 状态与证据 |
|---|---|---|
| Day 03 | Checkpoint 的 `model_config` 独立深拷贝 | **已修复**：`make_checkpoint()` 中返回 `copy.deepcopy(model_config)`，快照不再与外部 dict 共享引用 |
| Day 03 | RNG 状态预校验 | **已修复**：`load_checkpoint()` 中 `rng_state` 的类型校验位于 `model.load_state_dict(...)` **之前**，非法状态不会先改模型 |
| Day 04 | Adapter metadata 严格比较类型和值，不用近似 `allclose` | **已修复**：`load_adapter_package()` 改为键集合 + `type` 与值的严格比对 |
| Day 04 | 保存/加载后空行与注释整理 | **已修复**：随 `f5e16a8` 一并清理 |
| Day 06 | LoRA 注入前参数配置类型校验 | **已修复**：`inject_lora_attention()` 入口先用 `type(rank) is not int` 校验 `rank` 及 `alpha`/`dropout`，早于任何层收集与维度比较；随后校验 `target_modules` 类型、重复项、未知目标、已注入与 `rank` 越界。`LoRALinear.__init__()` 另有独立校验作为兜底 |
| Day 07 | 恢复时使用 `config.lora_dropout` | **已修复**：`_adapter_config()` 用 `config.lora_dropout` 重建 LoRA 配置，而不是主 dropout |
| Day 07 | 验证输出前切换 eval 模式 | **已修复**：`full_model`、`lora_model`、`reload_base`、`merged_lora` 均在 `eval()` + `torch.no_grad()` 下测量 |
| 收尾 | 审计脚本的参数量校验接受 `256.0` 等浮点值 | **已修复**：`positive_int()` 改为 `type(value) is int`，浮点与 `bool` 一律拒绝，并补 3 项回归测试 |
| 收尾 | 文档中的 LoRA 关键字参数名写成 `targets` | **已修复**：README 与报告统一为真实签名 `target_modules` |
| Git | 是否合并到 main | **未合并**：M03D08 分支仍待评审；`main` 尚未包含 `training/`、`experiments/`、`tools/` |

## 八、后续计划

本项目 V1 到此收尾，不再扩张功能。下一阶段另建独立项目，使用 Transformers / PEFT、真实数据集与可重复评测流程，验证自然语言能力（SFT/PEFT 方向），与本仓库的「架构理解 + 训练工程验证」定位分开。
