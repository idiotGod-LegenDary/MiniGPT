# Week 04 · Day 04 — Adapter-only 管理工程

## 一、项目范围与重要说明

- 本日**只新增四个文件**，可直接拖入已有 MiniGPT 根目录合并。
- **不会包含或覆盖** `training/data.py`、`training/loop.py`、`training/checkpoint.py`，也不修改现有 `model/`。
- `training/adapter_io.py` 共四个 TODO，均附有中文接口契约；测试和演示已提供。
- 原有仓库公开 `main` 暂未集成 LoRA，本次利用最小演示模型测试适配器管理协议。只有接入本人现有 LoRA 代码并完成集成验证后，才可在简历中声明“MiniGPT 已集成 LoRA 训练”。
- 不再实现 LoRA A/B 数学公式，本课重点是 Adapter 参数状态、序列化与测试。

## 二、目录（新增文件）

```text
training/adapter_io.py      # 四个独立 TODO，中文注释
tests/test_adapter_io.py    # 自动测试，无参考答案
demos/demo_adapter_io.py    # 训练、保存与加载演示
docs/week04_day04.md        # 本说明
```

## 三、依赖与执行

在 Windows CMD 的仓库根目录执行：

```bat
python -m pytest tests/test_adapter_io.py -q
python -m demos.demo_adapter_io
python -m pytest tests/ -q
```

本次提供的 `save_adapter_file()` **直接复用 Day 03 的** `training.checkpoint.save_checkpoint()`，因此确保 `training/checkpoint.py` 已完成。无需添加第三方依赖。

## 四、四个 TODO 验收契约

1. `set_adapter_trainable`：输入模型、Adapter 完整参数名序列，冻结其他参数，返回按 `numel` 计的数量报告。非法名称必须先报错、后不改变模型。
2. `make_adapter_package`：只导出 Adapter 参数，配置、权重均为**独立副本**。要求 Adapter 可训练且基础参数冻结。名称排序，版本号为 1。
3. `load_adapter_package`：加载前**完整**验证版本、基础模型 ID、配置、键集合、参数名称、shape 和 dtype；之后在 `torch.no_grad()` 内 `copy_` 到现有参数，不替换 `nn.Parameter`。
4. `compare_parameter_updates`：对训练前的完整参数快照作比较，准确报告基础权重是否改变、哪些 Adapter 参数已改变。

## 五、独立验收

- 解释为什么“冻结”不能只看 `requires_grad`。
- 解释为什么只有 LoRA 参数而没有基础模型权重，仍需要基础模型版本或 ID 信息。
- 解释为什么 `copy_` 不等于替换 Parameter 对象。
- 说明为什么 `max_abs_diff==0` 是当前测试的结果，却不能保证不同 GPU/精度环境下所有实验永远位级一致。
- 能通过 `named_parameters()` 找到真实 LoRA 参数名，并说明与演示中的 `lora_A/lora_B` 是否一致。若现有 LoRA 代码尚未在 MiniGPT，先不修改 `model/`。

## 六、已知限制

- `base_model_id` 是字符串元数据，未对基础权重做哈希校验。
- 当前示例将 Adapter 参数全部视为按名称明确选出的 `nn.Parameter`，不处理 PEFT 多 Adapter、量化和分布式保存。
- `read_adapter_file()` 用 `weights_only=True` 减少不安全反序列化风险，但仍只应读取可信文件。
- 目前未处理合并后的权重回滚/重复 merge；这将在接入已有 LoRA 代码后按其真实接口设计测试。

## 七、完成后 Git 提交

```bat
git add training/adapter_io.py tests/test_adapter_io.py demos/demo_adapter_io.py docs/week04_day04.md
git commit -m "feat: add adapter-only state management and tests"
git status
```
