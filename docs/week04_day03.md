# Week 04 · Day 03 — MiniGPT Checkpoint 与断点续训

## 一、如何合并文件（重要）

**本压缩包仅有以下 4 个新增文件，不包含 `data.py` 和 `loop.py`：**

- `training/checkpoint.py`：本日需要完成的 4 个 TODO。
- `tests/test_checkpoint.py`：本日自动测试。
- `demos/demo_checkpoint.py`：演示程序。
- `docs/week04_day03.md`：本说明文档。

请将这四个文件按相对路径复制到现有 MiniGPT 仓库根目录。Windows 中复制整个 `training/`、`tests/`、`demos/`、`docs/` 子文件夹并合并是可以的；但如果提示**覆盖之前已经完成的同名源文件**，请取消并先检查压缩包内容。

**不要再次解压 Day 01 的 TODO 包覆盖已完成的 `training/data.py`，也不要用 Day 02 的 TODO 包覆盖已完成的 `training/loop.py`。**

依赖现有文件：`training/data.py`、`training/loop.py` 和 `model/mini_gpt.py`。本日不实现学习率调度器，也不保存 DataLoader 迭代器位置。

## 二、本日接口契约

- `make_checkpoint(model, optimizer, step=..., model_config=...)`：构造独立快照，包含精确键集合 `format_version`、`step`、`model_config`、`optimizer_class`、`model_state`、`optimizer_state`、`rng_state`。
- `save_checkpoint(path, checkpoint)`：创建父目录，先写同目录临时文件，再替换目标路径；返回规范化的 `Path`。
- `load_checkpoint(path, model, optimizer, expected_config=..., device='cpu', restore_rng=True)`：使用 `torch.load(weights_only=True, map_location=device)`，先校验元信息、参数键/形状/类型以及优化器参数组结构，再恢复状态；返回已完成的步数。
- `compare_model_parameters(a, b, atol=0, rtol=0)`：比较同名参数，返回 `allclose`、`max_abs_diff`、`different`。

## 三、范围与限制

“完全一致地续训”的前提：模型代码、参数顺序、优化器类型与超参数、后续训练 Batch 顺序和 CPU PyTorch 环境一致，并恢复 CPU 随机数状态。

**本练习不保存** DataLoader/Sampler 的迭代位置、Python/NumPy/CUDA 随机数状态、学习率调度器状态或混合精度 GradScaler 状态。因此不能将此项目声称为适用于所有环境的通用断点续训系统。

**安全提醒**：只加载可信来源的 Checkpoint；`weights_only=True` 能降低部分反序列化风险，但不能保证任意文件绝对安全。

## 四、Windows CMD 运行命令

```bat
python -m pytest tests/test_checkpoint.py -q
python -m demos.demo_checkpoint
python -m pytest tests/ -q
```

## 五、独立验收（先不看答案）

1. 为什么 `copy.deepcopy(model.state_dict())` 比直接保留 `state_dict()` 引用更适合制作快照？
2. 为什么只加载模型参数不能保证 AdamW 的后续训练完全一致？
3. 为什么应当在修改目标模型之前检查配置、参数名和形状？
4. 如果未来使用随机打乱的 DataLoader 和 GPU Dropout，还需要保存哪些状态？
5. 分别运行“连续训练”和“中断后恢复训练”，保持后续 Batch 完全相同，对比所有模型参数。

## 六、后续课程

Day 04 将扩展 Adapter 单独保存、参数键与形状映射、`copy_()`、冻结/Merge/重载对照，以及实验报告整理。
