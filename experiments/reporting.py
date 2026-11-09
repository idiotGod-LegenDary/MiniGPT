"""Week 04 Day 05：实验记录、JSON 报告与两次运行的对照工具。

本文件只新增实验管理能力；不替换 data.py、loop.py 或 checkpoint.py。
只保存实际运行结果，不创建虚构的 Loss 数字。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


def _read_finite_loss(row: Mapping[str, Any], key: str) -> float:
    """已提供：读取一个 Loss 数值，拒绝布尔值、NaN、无穷大与负值。"""
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} 必须是实数")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{key} 必须是非负有限数")
    return value


def summarize_history(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """TODO 1：汇总一次实际训练的 History，不修改传入的数据。

    输入：非空序列，每条记录至少含 {'step': int, 'train_loss': float}，
         可能含有 'val_loss': float；step 必须依次为 1、2、...、N。
    输出：严格包含以下 6 个字段：
      - num_steps：总训练步数 int
      - first_train_loss：首步训练 Loss float
      - last_train_loss：末步训练 Loss float
      - best_val_loss：最小验证 Loss；没有验证记录则 None
      - best_val_step：最优验证 Loss 所在步；没有则 None
      - last_val_loss：最后一次验证 Loss；没有则 None
    规则：
      1. history 为空时抛 ValueError；非 dict/Mapping 的记录抛 TypeError。
      2. step 必须是非 bool 的 int 且恰好从 1 连续递增；否则 ValueError。
      3. 所有 Loss 都必须是非负有限实数，使用提供的 _read_finite_loss()。
      4. 最优验证 Loss 若出现并列，选择最早的 step。
      5. 返回普通 Python 数字与 None，不能返回 Tensor。

    提示：可以遍历 history 收集 (step, val_loss) 对，再用 min(..., key=...)。
    """
    # TODO 1：校验 History，并生成六字段总结。
    if not history:
        raise ValueError("history can't be None")
    pre_step = 0
    best_val_loss = math.inf
    best_step = -1
    last_val_loss = None
    first_train_loss = None
    last_train_loss = None
    for ss in history:
        if not isinstance(ss, Mapping):
            raise TypeError("每条 history 记录必须是 Mapping")
        
        step = ss.get("step")
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError("step 必须是非 bool 的 int")
        if step - pre_step != 1:
            raise ValueError("step 必须从 1 开始连续递增")
        pre_step = step

        train_loss = _read_finite_loss(ss,'train_loss')
        if first_train_loss is None:
            first_train_loss = train_loss
        last_train_loss = train_loss
        if 'val_loss' in ss:
            val_loss = _read_finite_loss(ss,'val_loss')
            last_val_loss = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
    return {
        "num_steps": pre_step,
        "first_train_loss": first_train_loss,
        "last_train_loss": last_train_loss,
        "best_val_loss": None if best_step == -1 else best_val_loss,
        "best_val_step": None if best_step == -1 else best_step,
        "last_val_loss": last_val_loss,
    }
   

def save_json_report(path: str | Path, record: dict[str, Any]) -> Path:
    """TODO 2：将实验记录安全保存为 UTF-8 JSON 文件。

    输入：目标文件路径与纯 JSON 可序列化的记录字典。
    输出：最终绝对路径 Path。
    规则：
      1. 先检查 record 为 dict，否则 TypeError。
      2. 使用 json.dumps(record, ensure_ascii=False, indent=2,
         allow_nan=False) 先生成文本；无法序列化时让异常传播。
      3. 创建目标父目录，使用 NamedTemporaryFile 在【同一父目录】
         生成临时文件（encoding='utf-8'、delete=False、mode='w'）。
      4. 成功写入后用 Path.replace() 替换目标文件。
      5. 使用 try/finally 清理残留临时文件，避免失败后留下 .tmp。
      6. 不能修改 record 本身。

    注意：这是 JSON 报告而非 torch.save 格式，所以与 Day 03 的
    Checkpoint 保存函数用途不同。
    """
    # TODO 2：序列化、同目录临时写入、替换、清理。
    if not isinstance(record, dict):
        raise TypeError("record type must be dict")
    text = json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            encoding="utf-8",
            delete=False,
            mode="w",
            suffix=".tmp",
        ) as fh:
            tmp_name = fh.name
            fh.write(text)              
        Path(tmp_name).replace(target)  
        tmp_name = None                 
    finally:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)

    return target


def compare_reports(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    atol: float = 1e-8,
) -> dict[str, Any]:
    """TODO 3：比较两次【同配置】实验的数值结果。

    输入：两份 run_experiment() 风格的报告，必须包含 config、history、
          baseline_val_loss、final_val_loss；atol 为非负有限浮点容差。
    输出：严格包含以下五个字段：
      reproduced: bool             # 所有比较项的差值都 <= atol
      max_train_loss_diff: float   # 每个 step 的训练 Loss 最大绝对差
      max_val_loss_diff: float     # 有验证记录的 step 的最大绝对差；无验证时 0
      baseline_diff: float         # 训练前验证 Loss 的绝对差
      final_diff: float            # 最终验证 Loss 的绝对差
    规则：
      1. atol 为 bool、负数、NaN、Inf 时抛 ValueError。
      2. 两份 config 必须严格相同，否则 ValueError；只比较同配置复现，
         不将不同超参数的实验误称为“复现”。
      3. 两份 history 长度、step 顺序、每步是否包含 val_loss 必须对应；
         不对应时抛 ValueError。
      4. 各 Loss 使用 _read_finite_loss 检查，不能出现非法数字。
      5. 不比较 environment 中的时间、运行耗时等非确定字段。
      6. 不修改任何输入报告。
    """
    # TODO 3：对齐两份 History，统计所有最大差值并判断是否复现。
    if isinstance(atol, bool) or not isinstance(atol, (int, float)):
        raise ValueError("atol 必须是实数")
    atol = float(atol)
    if not math.isfinite(atol) or atol < 0:           # 同时拒绝 NaN / inf / 负
        raise ValueError("atol 必须是非负有限数")
    
    if first['config'] != second['config']:
        raise ValueError("config mismatch")
    f_his, s_his = first['history'], second['history']
    if len(f_his) != len(s_his):
        raise ValueError("history len mismatch")
    
    max_train_diff = 0.0
    max_val_diff = 0.0

    for a, b in zip(f_his, s_his):
        if not isinstance(a, Mapping) or not isinstance(b, Mapping):
            raise TypeError("history 记录必须是 Mapping")
        if a.get("step") != b.get("step"):
            raise ValueError("step 序列不一致")

        # 每步是否包含 val_loss 必须对应
        a_has_val = "val_loss" in a
        b_has_val = "val_loss" in b
        if a_has_val != b_has_val:
            raise ValueError("val_loss 出现位置不一致")

        ta = _read_finite_loss(a, "train_loss")
        tb = _read_finite_loss(b, "train_loss")
        max_train_diff = max(max_train_diff, abs(ta - tb))

        if a_has_val:
            va = _read_finite_loss(a, "val_loss")
            vb = _read_finite_loss(b, "val_loss")
            max_val_diff = max(max_val_diff, abs(va - vb))

    # --- baseline / final ---
    baseline_diff = abs(
        _read_finite_loss(first, "baseline_val_loss")
        - _read_finite_loss(second, "baseline_val_loss")
    )
    final_diff = abs(
        _read_finite_loss(first, "final_val_loss")
        - _read_finite_loss(second, "final_val_loss")
    )

    reproduced = (
        max_train_diff <= atol
        and max_val_diff <= atol
        and baseline_diff <= atol
        and final_diff <= atol
    )

    return {
        "reproduced": reproduced,
        "max_train_loss_diff": max_train_diff,
        "max_val_loss_diff": max_val_diff,
        "baseline_diff": baseline_diff,
        "final_diff": final_diff,
    }
