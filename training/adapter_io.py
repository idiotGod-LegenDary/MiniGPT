"""Week 04 Day 04：Adapter-only 参数管理与一致性验证。

本文件只新增 Adapter 管理逻辑，不重新实现 LoRA 数学公式，
也不修改 MiniGPT 已有的 data.py / loop.py / checkpoint.py。

约定：调用方通过 model.named_parameters() 的完整参数名指定 Adapter。
例如：("projection.lora_A", "projection.lora_B")。
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn

FORMAT_VERSION = 1
PACKAGE_KEYS = {
    "format_version", "base_model_id", "adapter_config",
    "adapter_names", "adapter_state",
}


def _get_names(model: nn.Module, adapter_names: Iterable[str]) -> list[str]:
    """已提供：验证 Adapter 名称集合，并返回按字典序排列的名称。"""
    if isinstance(adapter_names, str):
        raise TypeError("adapter_names 应为名称序列，而不是单个字符串")
    names = list(adapter_names)
    if not names or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("adapter_names 必须是非空的参数名称序列")
    if len(set(names)) != len(names):
        raise ValueError("adapter_names 包含重复名称")
    param_names = set(dict(model.named_parameters()))
    missing = set(names) - param_names
    if missing:
        raise ValueError(f"模型中不存在这些 Adapter 参数：{sorted(missing)}")
    return sorted(names)


def _check_metadata(base_model_id: str, adapter_config: dict[str, Any]) -> None:
    """已提供：限定元数据为可安全序列化的扁平基本类型。"""
    if not isinstance(base_model_id, str) or not base_model_id.strip():
        raise ValueError("base_model_id 必须为非空字符串")
    if not isinstance(adapter_config, dict) or not adapter_config:
        raise ValueError("adapter_config 必须为非空字典")
    for key, value in adapter_config.items():
        if not isinstance(key, str) or not key:
            raise ValueError("adapter_config 的键必须是非空字符串")
        if type(value) not in (int, float, bool, str):
            raise TypeError("adapter_config 只接受 int/float/bool/str")


def snapshot_named_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    """已提供：获取所有命名参数的独立 CPU 快照，用于训练前后对照。"""
    return {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
    }


def save_adapter_file(path: str | Path, package: dict[str, Any]) -> Path:
    """已提供：复用 Day 03 的原子文件写入工具，避免重复造轮子。"""
    from training.checkpoint import save_checkpoint
    return save_checkpoint(path, package)


def read_adapter_file(path: str | Path) -> dict[str, Any]:
    """已提供：以 CPU、weights_only 模式读取本地可信 Adapter 文件。"""
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def set_adapter_trainable(
    model: nn.Module,
    adapter_names: Iterable[str],
) -> dict[str, int | float]:
    """TODO 1：只允许指定 Adapter 参数参与梯度更新。

    输入：model 与其已有参数的完整名称序列。
    输出：{'total': int, 'trainable': int, 'frozen': int, 'ratio': float}。
      total/trainable/frozen 按参数元素数量（numel）统计；ratio 为
      trainable/total，若模型无参数则返回 0.0（本课名称非空，正常不会发生）。
    契约：
      1. 先用 _get_names 验证输入，全部通过之后才允许修改模型；
      2. 对指定参数设置 requires_grad_(True)，其余设置 False；
      3. 返回参数数量统计，不修改参数数值；
      4. 非法名称要抛错，且不能修改原有冻结状态。
    """
    names = _get_names(model,adapter_names=adapter_names)
    total = trainable = 0
    for name,param in model.named_parameters():
        param.requires_grad_(name in names)
        total += param.numel()
        if name in names:
            trainable += param.numel()
    frozen = total - trainable
    ratio = trainable / total  if total else 0.0
    return {"total": total, 
            "trainable": trainable, 
            "frozen": frozen, 
            "ratio": ratio}

def make_adapter_package(
    model: nn.Module,
    *,
    adapter_names: Iterable[str],
    base_model_id: str,
    adapter_config: dict[str, Any],
) -> dict[str, Any]:
    """TODO 2：建立只包含 Adapter 权重的独立内存快照。

    返回字典键集合必须恰好等于 PACKAGE_KEYS：
      format_version：FORMAT_VERSION；
      base_model_id：指定基础模型标识；
      adapter_config：配置字典的深拷贝；
      adapter_names：按名称排序的字符串列表；
      adapter_state：{参数名: 参数的 detach().cpu().clone()}。
    约束：
      1. 使用 _get_names 与 _check_metadata 校验；
      2. 指定 Adapter 参数必须 requires_grad=True，其余参数必须冻结；
      3. 只保存指定 Adapter 参数，不保存基础模型权重；
      4. 创建快照后继续训练，不得改变已保存权重和配置。
    """
    names = _get_names(model, adapter_names)
    _check_metadata(base_model_id, adapter_config)
    for name, param in model.named_parameters():
        expected = name in set(names)
        if param.requires_grad != expected:
            raise ValueError(f"{name} requires_grad mismatch")
    return {
        'format_version':FORMAT_VERSION,
        'base_model_id':base_model_id,
        'adapter_config':copy.deepcopy(adapter_config),
        'adapter_names':sorted(names),
        'adapter_state':{
            name:param.detach().cpu().clone() 
            for name,param in model.named_parameters() if name in names
            }
    }


def load_adapter_package(
    model: nn.Module,
    package: dict[str, Any],
    *,
    expected_base_model_id: str,
    expected_config: dict[str, Any],
) -> None:
    """TODO 3：先完整校验，再将 Adapter 权重复制到现有参数对象。

    必须检查：
      - package 是 dict，顶层键集合严格等于 PACKAGE_KEYS；
      - format_version 正确；基础模型标识、配置与预期严格一致；
      - adapter_names 是无重复的非空名称列表，且与 adapter_state 键集合相同；
      - 这些参数名全部存在于 model；每个加载值都是 Tensor；
      - 各 Tensor 的 shape、dtype 与目标参数严格一致。
    全部检查通过后：
      with torch.no_grad():
          对每个目标参数使用 copy_(source.to(device=target.device))。
    不得：
      - 替换 nn.Parameter 对象本身；
      - 修改基础权重、参数 requires_grad 或优化器；
      - 在结构/配置/shape/dtype 校验失败后留下部分加载的参数变化。
    成功返回 None；不兼容抛 ValueError/TypeError。
    """
    if not isinstance(package, dict):
        raise TypeError("package must be dict")
    if set(package.keys()) != PACKAGE_KEYS:
        raise ValueError("package keys mismatch")
    if package['format_version'] != FORMAT_VERSION:
        raise ValueError("format version mismatch")
    if package['base_model_id'] != expected_base_model_id:
        raise ValueError("base model id mismatch")
    
    cfg = package['adapter_config']
    if set(cfg) != set(expected_config):
        raise ValueError("config keys mismatch")
    
    for k in expected_config:
        a, b = expected_config[k], cfg[k]
        if isinstance(a, torch.Tensor) or isinstance(a, (int, float)):
            if not torch.allclose(torch.as_tensor(a), torch.as_tensor(b)):
                raise ValueError(f"{k} value mismatch")
        elif a != b:
            raise ValueError(f"{k} value mismatch")

    names = package['adapter_names']
    if not isinstance(names, list) or not names:
        raise ValueError("adapter_names must be a non-empty list")
    if len(set(names)) != len(names):
        raise ValueError("adapter names have duplicates")
    if set(names) != set(package['adapter_state']):
        raise ValueError("names and state keys mismatch")

    params = dict(model.named_parameters())
    for k in names:
        if k not in params:
            raise ValueError(f"{k} not in model")
        src = package['adapter_state'][k]
        if not isinstance(src, torch.Tensor):
            raise TypeError(f"{k} source must be Tensor")
        tgt = params[k]
        if src.dtype != tgt.dtype:
            raise TypeError(f"{k} dtype mismatch")
        if src.shape != tgt.shape:
            raise ValueError(f"{k} shape mismatch")

    with torch.no_grad():
        for k in names:
            tgt = params[k]
            params[k].copy_(package['adapter_state'][k].to(device=tgt.device))


def compare_parameter_updates(
    model: nn.Module,
    before: Mapping[str, torch.Tensor],
    *,
    adapter_names: Iterable[str],
) -> dict[str, Any]:
    """TODO 4：对照训练前的全部参数快照，检查基础权重与 Adapter 更新。

    输入：model（训练后）、before（训练前 snapshot_named_parameters 结果）、
          adapter_names（完整名称列表）。
    返回字典键集合恰好为：
      {'base_unchanged': bool, 'adapter_updated': bool,
       'changed_base': list[str], 'changed_adapter': list[str]}。
    实现要求：
      1. 使用 _get_names 校验 Adapter 名称；
      2. before 的键集合必须与当前 named_parameters 完全一致；
      3. 对应参数 shape 不同则抛 ValueError；
      4. 对各参数用 torch.equal 比较 CPU Tensor（严格一致）；
      5. changed_base / changed_adapter 为变化的参数名，按字典序排序；
      6. adapter_updated 表示至少一个 Adapter 参数发生变化；
      7. 函数只读，不得修改模型或快照。
    """
    names = _get_names(model, adapter_names)
    adapter_set = set(names)

    current = dict(model.named_parameters())
    if set(before.keys()) != set(current.keys()):
        raise ValueError("keys mismatch")
    
    changed_base: list[str] = []
    changed_adapter: list[str] = []

    for name, param in current.items():
        if name not in before:
            raise ValueError(f"{name} missing in snapshot")
        snap = before[name]
        if tuple(snap.shape) != tuple(param.shape):
            raise ValueError(f"{name} shape mismatch")

        cur_cpu = param.detach().cpu()
        if not torch.equal(snap, cur_cpu):
            if name in adapter_set:
                changed_adapter.append(name)
            else:
                changed_base.append(name)

    changed_base.sort()
    changed_adapter.sort()

    return {
        "base_unchanged": len(changed_base) == 0,
        "adapter_updated": len(changed_adapter) > 0,
        "changed_base": changed_base,
        "changed_adapter": changed_adapter,
    }
