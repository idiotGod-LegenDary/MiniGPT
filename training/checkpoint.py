"""Week 04 Day 03：Checkpoint 保存、加载与断点续训。

这是增量文件：只需新增 training/checkpoint.py。
不要覆盖已经完成的 training/data.py 或 training/loop.py。
本次范围：CPU、单个模型、单个 AdamW 优化器、不含学习率调度器、
后续训练 Batch 顺序固定。

注意：本练习不保存 DataLoader 迭代位置。即使恢复了模型、优化器
和 CPU 随机数状态，也需要保证恢复后使用的训练 Batch 顺序相同，
才能检验精确的断点续训一致性。
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

import tempfile

FORMAT_VERSION = 1
REQUIRED_KEYS = {
    "format_version", "step", "model_config", "optimizer_class",
    "model_state", "optimizer_state", "rng_state",
}


def _validate_step_and_config(step: int, model_config: dict[str, Any]) -> None:
    """已提供，无需修改：检查训练步数和配置，拒绝布尔步数、非法类型等。"""
    if type(step) is not int or step < 0:
        raise ValueError("step must be a non-negative integer")
    if not isinstance(model_config, dict) or not model_config:
        raise ValueError("model_config must be a nonempty dict")
    if not all(isinstance(k, str) and k for k in model_config):
        raise TypeError("config keys must be nonempty strings")
    if not all(type(v) in (int, float, str, bool) for v in model_config.values()):
        raise TypeError("config values must be flat JSON-compatible primitives")


def make_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    *,
    step: int,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """TODO 1：构造【独立的】内存 Checkpoint 快照，不写磁盘。

    输入：
      model：当前模型；optimizer：当前优化器。
      step：已完成的优化器更新次数，必须为非负整数。
      model_config：非空、扁平的模型配置字典。

    输出：字典，键集合必须恰好等于 REQUIRED_KEYS。
      format_version：FORMAT_VERSION。
      step：已完成更新次数（0 表示尚未更新）。
      model_config：传入配置的独立副本。
      optimizer_class：type(optimizer).__name__。
      model_state：模型 state_dict 的深拷贝。
      optimizer_state：优化器 state_dict 的深拷贝。
      rng_state：torch.get_rng_state().clone()（仅 CPU 随机状态）。

    实现要求：
      1. 调用已提供的 _validate_step_and_config()。
      2. 使用 copy.deepcopy() 创建真正独立的快照，后续训练不能改写快照。
      3. 不修改模型或优化器，也不写任何文件。
    """
    _validate_step_and_config(step,model_config)
    return {'format_version':FORMAT_VERSION,
            'step':step,
            'model_config':copy.deepcopy(model_config),
            'optimizer_class':type(optimizer).__name__,
            'model_state':copy.deepcopy(model.state_dict()),
            'optimizer_state':copy.deepcopy(optimizer.state_dict()),
            'rng_state':torch.get_rng_state().clone()
            }

def save_checkpoint(path: str | Path, checkpoint: dict[str, Any]) -> Path:
    """TODO 2：将 Checkpoint 安全保存到本地文件。

    输入：目标路径（str 或 Path）、Checkpoint 字典。
    输出：规范化后的最终目标路径 Path。

    实现步骤：
      1. 用 Path(path).resolve() 得到目标路径。
      2. 自动创建父目录：mkdir(parents=True, exist_ok=True)。
      3. 在【目标目录】里创建临时文件，可用 tempfile.NamedTemporaryFile
         （delete=False、dir=父目录）。在调用 torch.save 前关闭文件句柄。
      4. 先将 checkpoint 写入临时文件，再用 tmp_path.replace(target) 替换。
      5. 使用 try/finally；即使失败，也要清理残留临时文件。

    不得修改 checkpoint 字典；正常完成后不得留下 .tmp 文件。
    """
    p = Path(path).resolve()
    p.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir = p.parent,
        suffix = '.tmp',
        delete=False
    ) as f:
      tmp_path = Path(f.name)

    try:
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(p)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return p

def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    expected_config: dict[str, Any],
    device: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> int:
    """TODO 3：读取、校验并恢复训练状态，返回已完成的 step。

    输入：
      path：文件路径；model、optimizer：已经构造好的目标对象。
      expected_config：本次模型期望的配置；device：加载映射设备。
      restore_rng：是否恢复保存的 CPU 随机数状态。
    输出：Checkpoint 保存的非负整数 step。

    必须按顺序处理：
      1. 使用 torch.load(path, map_location=device, weights_only=True) 读取。
      2. 检查读取结果是否为 dict，且键集合恰好为 REQUIRED_KEYS。
      3. 校验 format_version、step、model_config、optimizer_class。
      4. 【修改模型之前】比较 model_state 与 model.state_dict()：
         参数/Buffer 名称集合、各 Tensor 的 shape 与 dtype 必须一致。
      5. 检查 optimizer_state 存在 state、param_groups，且参数组数量、
         每组参数 ID 数量与当前优化器一致。
      6. 校验通过后，依次调用 model.load_state_dict(..., strict=True)
         和 optimizer.load_state_dict(...)。
      7. restore_rng=True 时，用 torch.set_rng_state() 恢复 CPU RNG 状态。
      8. 返回保存的 step；不要在这里执行 optimizer.step()。

    不兼容的配置/结构须抛出 ValueError。模型与优化器应已在目标设备上。
    本练习只保证“上述可检查问题在修改模型前被拒绝”，
    不要求实现所有异常情况下的事务性回滚。
    """
    tmp = torch.load(path, map_location=device,weights_only=True)
    if not isinstance(tmp, dict):
        raise TypeError('checkpoint not dict')
    if set(tmp.keys()) != set(REQUIRED_KEYS):
        raise ValueError("checkpoint key error")
    
    if tmp['format_version'] != FORMAT_VERSION:
        raise ValueError("format_version mismatch")
    step = tmp["step"]
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("step must be a non-negative int")

    if tmp["model_config"] != expected_config:
        raise ValueError("model_config mismatch")

    if tmp["optimizer_class"] != type(optimizer).__name__:
        raise ValueError("optimizer_class mismatch")

    cur = model.state_dict()
    saved = tmp["model_state"]
    if set(saved.keys()) != set(cur.keys()):
        raise ValueError("model_state keys mismatch")
    for k in cur:
        if saved[k].shape != cur[k].shape:
            raise ValueError(f"shape mismatch at {k}")
        if saved[k].dtype != cur[k].dtype:
            raise ValueError(f"dtype mismatch at {k}")

    opt_sd = tmp["optimizer_state"]
    if "state" not in opt_sd or "param_groups" not in opt_sd:
        raise ValueError("optimizer_state missing state/param_groups")
    if len(opt_sd["param_groups"]) != len(optimizer.param_groups):
        raise ValueError("param_groups count mismatch")
    for saved_g, cur_g in zip(opt_sd["param_groups"], optimizer.param_groups):
        if len(saved_g["params"]) != len(cur_g["params"]):
            raise ValueError("param group size mismatch")

    if restore_rng:
        rng = tmp["rng_state"]
        if not isinstance(rng, torch.Tensor):
            raise ValueError("rng_state must be a tensor")
        torch.set_rng_state(rng.cpu())

    model.load_state_dict(tmp['model_state'],strict=True)
    optimizer.load_state_dict(tmp['optimizer_state'])

    return tmp['step']


def compare_model_parameters(
    left: nn.Module,
    right: nn.Module,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, Any]:
    """TODO 4：只读比较两个模型的全部【命名参数】，返回差异报告。

    输入：两个 nn.Module；atol/rtol 为非负且有限的容差，不能是 bool。
    输出：字典，键集合必须恰好为：
      {'allclose': bool, 'max_abs_diff': float, 'different': list[str]}
      allclose：所有对应参数是否在容差内一致。
      max_abs_diff：所有参数的最大逐元素绝对差。
      different：不满足容差的参数名，按字典序排序。

    实现要求：
      1. atol/rtol 为负数、非有限数或 bool 时抛出 ValueError。
      2. 用 dict(model.named_parameters()) 获取参数映射，键集合必须一致。
      3. 同名参数 shape 不一致时抛出 ValueError。
      4. detach() 后转到 CPU，并转换为 float64 进行稳定比较。
      5. 使用 torch.allclose(..., atol=atol, rtol=rtol) 判断一致性。
      6. 不修改任何模型参数。相同模型差异为 0.0，different 为空列表。

    本次默认比较浮点参数；此函数不比较 Buffer。
    """
    for name, val in (("atol", atol), ("rtol", rtol)):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"{name} must be a real number, not bool")
        if not math.isfinite(val):
            raise ValueError(f"{name} must be finite")
        if val < 0:
            raise ValueError(f"{name} must be non-negative")
        
    left_params = dict(left.named_parameters())
    right_params = dict(right.named_parameters())
    if set(left_params.keys()) != set(right_params.keys()):
        raise ValueError("parameter name sets differ")

    all_ok = True
    max_abs_diff = 0.0
    different: list[str] = []
    for name, left_t in left_params.items():
        right_t = right_params[name]

        if left_t.shape != right_t.shape:
            raise ValueError(f"shape mismatch at {name}")

        a = left_t.detach().cpu().double()
        b = right_t.detach().cpu().double()

        cur_max = (a - b).abs().max().item()
        if cur_max > max_abs_diff:
            max_abs_diff = cur_max

        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            all_ok = False
            different.append(name)

    return {
        "allclose": all_ok,
        "max_abs_diff": float(max_abs_diff),
        "different": sorted(different),
    }