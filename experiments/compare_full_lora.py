"""Week 04 Day 07：MiniGPT 全参数训练与 LoRA 的可控对照实验。

这是新增文件；请勿覆盖既有 model/、training/、experiments/run_toy.py。
本次是 CPU 周期性合成 Token 的工程验证，不是自然语言 SFT 效果证明。

学生仅需完成 4 个 TODO。函数之间的依赖关系见每个 TODO 的中文接口契约。
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from model.mini_gpt import MiniGPT
from training.adapter_io import (
    compare_parameter_updates,
    load_adapter_package,
    make_adapter_package,
    read_adapter_file,
    save_adapter_file,
    snapshot_named_parameters,
)
from training.data import build_loader
from training.loop import evaluate, train_for_steps
from training.lora_integration import LoRALinear, inject_lora_attention


@dataclass(frozen=True)
class ComparisonConfig:
    """实验的全部可控参数；只使用 CPU 与固定批次，便于验证。"""
    seed: int = 42
    vocab_size: int = 16
    block_size: int = 6
    train_tokens: int = 96
    val_tokens: int = 56
    batch_size: int = 6
    num_steps: int = 6
    eval_every: int = 3
    learning_rate: float = 2e-3
    d_model: int = 16
    d_ff: int = 32
    num_layers: int = 2
    dropout: float = 0.0
    rank: int = 2
    alpha: float = 4.0
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = ("W_q", "W_v")


def _validate_config(config: ComparisonConfig) -> None:
    """已提供：尽可能在创建模型之前阻止无效配置。"""
    if not isinstance(config, ComparisonConfig):
        raise TypeError("config 必须是 ComparisonConfig")
    integer_fields = (
        "vocab_size", "block_size", "train_tokens", "val_tokens", "batch_size",
        "num_steps", "eval_every", "d_model", "d_ff", "num_layers", "rank",
    )
    for field in integer_fields:
        value = getattr(config, field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{field} 必须是正整数")
    if type(config.seed) is not int or config.seed < 0:
        raise ValueError("seed 必须是非负整数")
    if config.vocab_size < 2:
        raise ValueError("vocab_size 必须不小于 2")
    if config.train_tokens <= config.block_size or config.val_tokens <= config.block_size:
        raise ValueError("训练集、验证集均需至少 block_size+1 个 Token")
    if config.eval_every > config.num_steps:
        raise ValueError("eval_every 不得大于 num_steps")
    if config.rank > config.d_model:
        raise ValueError("rank 不得大于模型投影维度")
    for field in ("learning_rate", "alpha", "dropout", "lora_dropout"):
        value = getattr(config, field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{field} 必须是有限实数")
    if config.learning_rate <= 0 or config.alpha <= 0:
        raise ValueError("学习率与 alpha 必须大于 0")
    # 今天严格控制随机性，暂不处理模型/LoRA Dropout 带来的额外随机数消耗。
    if config.dropout != 0 or config.lora_dropout != 0:
        raise ValueError("本日对照实验要求两个 dropout 均为 0")
    allowed = {"W_q", "W_k", "W_v", "out_proj"}
    if (not isinstance(config.target_modules, tuple)
            or not config.target_modules
            or any(not isinstance(t, str) or t not in allowed for t in config.target_modules)
            or len(set(config.target_modules)) != len(config.target_modules)):
        raise ValueError("target_modules 必须是非空、合法且不重复的名称元组")


def _new_model(config: ComparisonConfig) -> MiniGPT:
    """已提供：按统一配置创建基础模型，不设置随机种子。"""
    return MiniGPT(
        vocab_size=config.vocab_size,
        max_seq_len=config.block_size,
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )


def _adapter_config(config: ComparisonConfig) -> dict[str, Any]:
    """已提供：Adapter 元数据保持扁平基本类型，以兼容 Day 04 工具。"""
    return {
        "rank": config.rank,
        "alpha": float(config.alpha),
        "dropout": float(config.lora_dropout),
        "target_modules": ",".join(config.target_modules),
    }


def _parameter_stats(model: nn.Module) -> dict[str, int | float]:
    """已提供：统计训练参数比例。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_ratio": trainable / total if total else 0.0,
    }


def build_model_pair(config: ComparisonConfig) -> tuple[MiniGPT, MiniGPT, list[str]]:
    """TODO 1：构造共享【完全相同基础权重】的全参和 LoRA 模型。

    输入：合法的 ComparisonConfig。
    输出：(full_model, lora_model, adapter_names)。

    必须：
    1. 先调用 _validate_config，再设置 torch.manual_seed(config.seed)。
    2. 只初始化【一次】基础 MiniGPT；用 copy.deepcopy() 得到第二份。
       不能独立调用两次随机初始化后声称初始参数相同。
    3. full_model 保持所有参数 requires_grad=True。
    4. 仅对 lora_model 调用 inject_lora_attention()：传入 rank、alpha、
       lora_dropout 和 target_modules；函数内部复用 Day04 的冻结工具。
    5. 返回按名称排序的 adapter_names；两模型在 CPU 上。

    注意：今天的第一个可比性条件是两条路线共享同一份基础初始权重。
    """
    # TODO 1：基础权重复制、实际 LoRA 注入、冻结状态和返回值。
    _validate_config(config)
    torch.manual_seed(config.seed)
    full_model = _new_model(config)
    lora_model = copy.deepcopy(full_model)

    full_model.requires_grad_(True)
    adapter_names = inject_lora_attention(
        model=lora_model,
        rank=config.rank,
        alpha=config.alpha,
        dropout=config.lora_dropout,
        target_modules=config.target_modules
    )
    adapter_names.sort()
    return (full_model, lora_model, adapter_names)


def build_shared_batches(
    config: ComparisonConfig,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]]:
    """TODO 2：创建两个模型【完全相同】的训练 Batch 序列与验证 Batch。

    输入：ComparisonConfig。
    输出：(train_batches, val_batches)，均为非空 list；每项为(x,y)。
    x/y 都是 CPU torch.long，形状 [B,T]（最后一批允许较小 B）。

    必须：
    1. tokens = torch.arange(train_tokens+val_tokens) % vocab_size，long 类型。
    2. 【先切原始 Token，再建滑窗】以 train_tokens 为边界切开。
    3. 调用已有 build_loader：训练 shuffle=True、num_workers=0，
       generator=torch.Generator().manual_seed(config.seed+1)；
       验证 shuffle=False、num_workers=0。
    4. 对训练 loader 取出恰好 num_steps 个 Batch；迭代器耗尽时
       重新 iter(loader)，不能少取，也不能隐式重新随机生成不同数据。
    5. 验证集转成 list，不能为空。
    6. 不更改用户已有的 data.py，不复写 Dataset 或 DataLoader。

    注意：这不是消除所有数据偏差的真实训练评测，只是控制实验用批次顺序。
    """
    # TODO 2：切数据、复用 loader、固定完整 batch 序列。
    tokens = torch.arange(config.train_tokens + config.val_tokens,dtype=torch.long) % config.vocab_size
    train_tokens = tokens[:config.train_tokens]
    val_tokens = tokens[config.train_tokens:]
    train_loader = build_loader(train_tokens,
        config.block_size,
        config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(config.seed + 1)
    )
    val_loader = build_loader(val_tokens,
        config.block_size,
        config.batch_size,
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(config.seed + 1)
    )
    train_batches = []
    val_batches = []
    while len(train_batches) < config.num_steps:
        for batch in train_loader:
            train_batches.append(batch)
            if len(train_batches) == config.num_steps:
                break
    val_batches = list(val_loader)
    if not val_batches:
        raise ValueError("val_batches is empty")

    return (train_batches, val_batches)


def merge_lora_for_inference(model: nn.Module) -> nn.Module:
    """TODO 3：返回不含 LoRALinear 的独立合并副本，仅供推理。

    输入：已经实际注入 LoRA 的 nn.Module，不能是未注入的基础模型。
    输出：独立的 merged 模型，处于 eval()；不修改输入模型。

    操作：
    1. 先检查类型并确认至少有一个 LoRALinear，否则 ValueError。
    2. deepcopy(model)，只在副本中执行合并。
    3. 找出副本中的 LoRALinear 模块，逐个将低秩增量加入其 base.weight：
         delta = (lora_B @ lora_A) * scale  # [out_features,in_features]
       必须在 torch.no_grad() 下更新，并检查 dtype/device 一致。
    4. 将包装层替换回其内部的原生 nn.Linear(base)。
       提示：对模块路径如 net.0.attn.W_q：
         parent_path, _, attr = name.rpartition('.')
         parent = merged.get_submodule(parent_path)
         setattr(parent, attr, layer.base)
    5. merged.eval()，返回；输入 model 的参数、模块、模式均不改变。

    对照要求：在 eval() 下，合并前后 logits 近似一致 (atol=1e-5)。
    注意：Dropout 在训练状态下不能直接这样比较；本课比较 eval()。
    """
    # TODO 3：独立 deepcopy、矩阵合并、模块替换、eval。
    if not isinstance(model, nn.Module):
        raise TypeError(f"expected nn.Module, got {type(model).__name__}")
    lora_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    ]
    if not lora_modules:
        raise ValueError("No LoRALinear found in model")
    
    merged = copy.deepcopy(model)
    merged_lora = [
        (name, module)
        for name, module in merged.named_modules()
        if isinstance(module, LoRALinear)
    ]

    for name, layer in merged_lora:
            base = layer.base
            w = base.weight

            # dtype / device 一致性检查
            if not (w.device == layer.lora_A.device == layer.lora_B.device):
                raise ValueError(f"device mismatch in LoRALinear '{name}'")
            if not (w.dtype == layer.lora_A.dtype == layer.lora_B.dtype):
                raise ValueError(f"dtype mismatch in LoRALinear '{name}'")

            # 4. 合并低秩增量到 base.weight（必须在 no_grad 下原地更新）
            with torch.no_grad():
                delta = (layer.lora_B @ layer.lora_A) * layer.scale  # [out, in]
                if delta.shape != w.shape:
                    raise ValueError(
                        f"shape mismatch in LoRALinear '{name}': "
                        f"delta {tuple(delta.shape)} vs weight {tuple(w.shape)}"
                    )
                w.add_(delta)

            # 5. 把包装层替换回原生 nn.Linear
            parent_path, _, attr = name.rpartition(".")
            if parent_path:
                parent = merged.get_submodule(parent_path)
            else:
                parent = merged  # 顶层就是 LoRALinear
            setattr(parent, attr, base)

    merged.eval()
    return merged

def run_comparison(
    config: ComparisonConfig,
    *,
    adapter_path: str | Path,
) -> dict[str, Any]:
    """TODO 4：实际运行两路训练、Adapter 往返和 Merge 输出对照。

    输入：合法 ComparisonConfig、保存 Adapter 的路径。
    输出：可被 json.dumps(..., allow_nan=False) 序列化的报告 dict，
    严格具有下列顶层键：
      config, data_note, initial_max_abs_diff, full, lora, adapter

    具体要求：
    1. 调用 build_model_pair() 和 build_shared_batches()。
    2. 【在训练之前】deepcopy(full_model) 得到 reload_base，保存初始基础权重。
       不得复制已经训练过的 LoRA 模型伪装成重载成功。
    3. 将 full/lora 置于 eval()，使用相同 val_batches[0][0] 测量
       注入后未训练时的最大输出差异 initial_max_abs_diff。
    4. 使用现有 evaluate() 得到两路的 baseline_val_loss。
    5. 在注入之后创建两个独立的 AdamW：
       full 对所有参数优化；LoRA 仅对 requires_grad=True 的参数优化。
       两者 lr 相同，weight_decay=0.0。
    6. 对两路各自调用 train_for_steps()，训练相同的 train_batches、
       num_steps、val_batches、eval_every，获取两组 history。
    7. 两路分别计算 final_val_loss，统计可训练参数量。
    8. LoRA 训练前用 snapshot_named_parameters() 保存独立快照；
       训练后用 compare_parameter_updates() 检验基础参数不变且 Adapter 更新。
    9. 使用 make_adapter_package() → save_adapter_file() →
       read_adapter_file()；包的 base_model_id 为
       f'minigpt-synthetic-seed-{config.seed}'，配置由 _adapter_config 提供。
   10. 对 reload_base 注入同配置 LoRA，调用 load_adapter_package() 加载；
       在 eval() 下比较其输出与训练后的 LoRA 原模型输出。
   11. 调用 merge_lora_for_inference(lora_model)，eval() 下比较输出。
   12. 报告不得包含 Tensor、模型对象、Optimizer，且记录真实结果。

    规定报告子字段：
      full: {baseline_val_loss, final_val_loss, history, stats}
      lora: {baseline_val_loss, final_val_loss, history, stats}
      adapter: {base_unchanged, adapter_updated, changed_base,
                adapter_tensor_count, adapter_file_bytes,
                reload_max_abs_diff, merge_max_abs_diff}
    所有 Loss 和差异字段是普通 Python float；列表中也是普通数字。

    注意：同数据同初始权重不等于两种模型可训练容量相同；
    本实验只能展示合成数据下的数值与系统正确性，不可称为 SFT 胜负结论。
    """
    # TODO 4：组合已经完成的训练模块、IO 模块与前三个 TODO。
    # 1. 建模型对和共享批次
    full_model, lora_model, adapter_names = build_model_pair(config=config)
    train_batches, val_batches = build_shared_batches(config=config)
    
    # 2. 训练前保存初始基础权重（必须训练前）
    reload_base = copy.deepcopy(full_model)
    
    # 3. eval 下测初始差异
    full_model.eval()
    lora_model.eval()
    x0 = val_batches[0][0]
    with torch.no_grad():
        initial_max_abs_diff = (full_model(x0) - lora_model(x0)).abs().max().item()

    # 4. baseline val loss    
    full_baseline  = float(evaluate(full_model,val_batches))
    lora_baseline = float(evaluate(lora_model,val_batches))

    # 8. 训练前快照
    lora_snap = snapshot_named_parameters(lora_model)

    # 5. 两个独立 AdamW
    opt_full = torch.optim.AdamW(
        full_model.parameters(),
        weight_decay=0.,
        lr=config.learning_rate
    )
    opt_lora = torch.optim.AdamW(
        [param for param in lora_model.parameters() if param.requires_grad],
        weight_decay=0.,
        lr=config.learning_rate
    )

    # 6. 两路训练
    history_full = train_for_steps(
        model=full_model,
        optimizer=opt_full,
        train_loader=train_batches,
        num_steps=config.num_steps,
        val_loader=val_batches,
        eval_every=config.eval_every
    )
    history_lora = train_for_steps(
        model=lora_model,
        optimizer=opt_lora,
        train_loader=train_batches,
        num_steps=config.num_steps,
        val_loader=val_batches,
        eval_every=config.eval_every
    )
    
    # 7. final loss + stats
    full_final = float(evaluate(full_model, val_batches))
    lora_final = float(evaluate(lora_model, val_batches))

    cmp_upd = compare_parameter_updates(model=lora_model,
        before=lora_snap,
        adapter_names=adapter_names
    )
    adapter_tensor_count = sum(
        1 for n, _ in lora_model.named_parameters() if n in adapter_names)

    # 9. adapter IO 往返
    base_model_id = f"minigpt-synthetic-seed-{config.seed}"
    adapter_config = _adapter_config(config=config)
    pkg = make_adapter_package(
        model=lora_model, adapter_names=adapter_names,
        base_model_id=base_model_id, adapter_config=adapter_config)
    save_adapter_file(path=adapter_path, package=pkg)
    adapter_file_bytes = Path(adapter_path).stat().st_size
    loaded_pkg = read_adapter_file(path=adapter_path)

    # 10. reload_base 注入 LoRA + 加载，对比输出
    inject_lora_attention(
        model=reload_base,
        rank=config.rank,
        alpha=config.alpha,
        dropout=config.lora_dropout,
        target_modules=config.target_modules
    )
    load_adapter_package(
        reload_base, loaded_pkg,
        expected_base_model_id=base_model_id,
        expected_config=adapter_config
    )
    reload_base.eval()
    
    reload_max_abs_diff = 0.
    with torch.no_grad():
        for x,_ in val_batches:
            reload_max_abs_diff = max(reload_max_abs_diff, (reload_base(x) - lora_model(x)).abs().max().item())

    # 11. merge 对比
    merged_lora = merge_lora_for_inference(lora_model)
    merged_lora.eval()
    merge_max_abs_diff = 0.
    with torch.no_grad():
        for x,_ in val_batches:
                merge_max_abs_diff = max(merge_max_abs_diff, (merged_lora(x) - lora_model(x)).abs().max().item())


    # 12. 组装报告（全 Python 原生类型）
    def _stats(m):
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {"total_params": total, "trainable_params": trainable,"trainable": trainable,"trainable_ratio": trainable / total if total > 0 else 0.0,}
    return {
        "config": asdict(config) if is_dataclass(config) else dict(vars(config)),
        "data_note": "合成 tokens；仅验证数值与系统正确性，不是 SFT 胜负结论。",
        "initial_max_abs_diff": initial_max_abs_diff,
        "full": {
            "baseline_val_loss": full_baseline,
            "final_val_loss": full_final,
            "history": history_full,
            "stats": _stats(full_model),
        },
        "lora": {
            "baseline_val_loss": lora_baseline,
            "final_val_loss": lora_final,
            "history": history_lora,
            "stats": _stats(lora_model),
        },
        "adapter": {
            "base_unchanged": cmp_upd["base_unchanged"],
            "adapter_updated": cmp_upd["adapter_updated"],
            "changed_base": cmp_upd["changed_base"],
            "adapter_tensor_count": adapter_tensor_count,
            "adapter_file_bytes": adapter_file_bytes,
            "reload_max_abs_diff": reload_max_abs_diff,
            "merge_max_abs_diff": merge_max_abs_diff,
        },
    }