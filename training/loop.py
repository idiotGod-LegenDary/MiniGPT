"""Week 04 Day 02: MiniGPT training loop (student TODOs).

Use only the interfaces taught today. This file is additive to the existing
MiniGPT repository; it does NOT replace your training/data.py or model/ code.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer


def _move_batch(
    batch: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PROVIDED: validate batch and place it on the model's device.

    Expect (input_ids, labels), both long [B,T] tensors with B,T>0.
    Labels may contain -100 (ignored); input_ids must not contain -100.
    """

    """已提供：校验 batch 并将其放到模型所在的设备上。

    期望输入为 (input_ids, labels)，两者都是 long 类型的 [B,T] 张量，
    且 B,T>0。labels 可以包含 -100（表示忽略）；input_ids 不得包含 -100。
    """
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError("batch must contain exactly (input_ids, labels)")
    x, y = batch
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("inputs and labels must be torch.Tensor")
    if x.ndim != 2 or y.shape != x.shape or x.numel() == 0:
        raise ValueError("inputs and labels must be nonempty matching [B,T]")
    if x.dtype != torch.long or y.dtype != torch.long:
        raise TypeError("input_ids and labels must be torch.long")
    return x.to(device), y.to(device)


def compute_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """TODO 1: token-level mean causal-LM cross entropy.

    Inputs:
      logits: float Tensor [B,T,V] from MiniGPT (raw, NOT softmax).
      labels: torch.long Tensor [B,T], ALREADY next-token shifted.
      ignore_index: label sentinel omitted from CE; default -100.
    Returns: scalar differentiable Tensor (0-D).
    Must:
      - raise ValueError if shapes disagree, logits not rank 3,
        or there are no valid labels;
      - raise TypeError if labels are not torch.long;
      - flatten B,T only; class dimension V must remain last;
      - use F.cross_entropy(..., ignore_index=ignore_index, reduction='mean').
    No backward/optimizer work here; do not shift labels again.
    """

    """TODO 1：token 级平均的因果语言模型交叉熵。

    输入：
      logits：来自 MiniGPT 的 float 张量 [B,T,V]（原始值，未做 softmax）。
      labels：torch.long 张量 [B,T]，已经做过 next-token 移位。
      ignore_index：从 CE 中忽略的 label 哨兵值；默认为 -100。
    返回：标量、可求导的 Tensor（0 维）。
    必须：
      - 当形状不一致、logits 不是 3 维、或没有有效 label 时，
        抛出 ValueError；
      - 当 labels 不是 torch.long 时抛出 TypeError；
      - 只展平 B,T 两维；类别维度 V 必须保持在最后；
      - 使用 F.cross_entropy(..., ignore_index=ignore_index, reduction='mean')。
    此处不做 backward/优化器相关工作；不要再次对 labels 做移位。
    """
    if logits.ndim != 3:
        raise ValueError("logits must be rank 3 [B,T,V]")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must match logits [B,T]")
    if labels.dtype != torch.long:
        raise TypeError("labels must be torch.long")
    if (labels != ignore_index).sum().item() == 0:
        raise ValueError("no valid labels")

    vocab = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, vocab),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )


def train_step(
    model: nn.Module,
    optimizer: Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
    *,
    device: str | torch.device = "cpu",
    max_grad_norm: float | None = None,
) -> float:
    """TODO 2: one optimizer update, return DETACHED Python loss float.

    Required order:
      1. validate max_grad_norm (if given: finite and > 0);
      2. model.train(); _move_batch(batch, device);
      3. optimizer.zero_grad(set_to_none=True);
      4. logits=model(x), loss=compute_lm_loss(logits,y);
      5. reject nonfinite loss with FloatingPointError;
      6. loss.backward(); optionally clip_grad_norm_(model.parameters(),max_norm=...);
      7. optimizer.step(); return float(loss.detach().item()).

    Assumption: caller already called model.to(device), e.g. CPU today.
    Exactly one backward and one optimizer step. Do not reset RNG.
    """
    """TODO 2：执行一次优化器更新，返回已 detach 的 Python float 损失值。

    要求的执行顺序：
      1. 校验 max_grad_norm（如果给定：必须有限且 > 0）；
      2. model.train()；_move_batch(batch, device)；
      3. optimizer.zero_grad(set_to_none=True)；
      4. logits=model(x), loss=compute_lm_loss(logits,y)；
      5. 若 loss 非有限值，抛出 FloatingPointError；
      6. loss.backward()；可选地调用 clip_grad_norm_(model.parameters(),max_norm=...)；
      7. optimizer.step()；返回 float(loss.detach().item())。

    前提假设：调用方已经调用了 model.to(device)，例如今天使用 CPU。
    恰好执行一次 backward 和一次优化器 step。不要重置 RNG。
    """
    if max_grad_norm is not None:
      if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
          raise ValueError("max_grad_norm must be finite and > 0")
    model.train()
    x,y =_move_batch(batch,device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = compute_lm_loss(logits,y)
    if not torch.isfinite(loss):
      raise FloatingPointError(f"non-finite loss: {loss.item()}")
    loss.backward()
    if max_grad_norm is not None:
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()
    return float(loss.detach().item())    


def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: str | torch.device = "cpu",
) -> float:
    """TODO 3: token-WEIGHTED validation loss, no optimizer update.

    Must:
      - remember original model.training flag, switch to model.eval();
      - use torch.no_grad() around all model forward passes;
      - for each batch, use _move_batch, compute_lm_loss;
      - count valid labels (labels != -100), aggregate sum(loss * count) /
        sum(count) (not plain average across batches);
      - restore model.train(original_training_flag) even on exception
        (try/finally recommended);
      - return Python float; raise ValueError on empty loader.
    """
    """TODO 3：token 加权的验证损失，不执行优化器更新。

    必须：
      - 记住原始的 model.training 标志，切换到 model.eval()；
      - 在所有模型前向传播外使用 torch.no_grad()；
      - 对每个 batch，使用 _move_batch、compute_lm_loss；
      - 统计有效 label 数量（labels != -100），按
        sum(loss * count) / sum(count) 聚合（而不是对各个 batch 做简单平均）；
      - 即使发生异常，也要恢复 model.train(原始训练标志)
        （推荐使用 try/finally）；
      - 返回 Python float；当 loader 为空时抛出 ValueError。
    """    
    flag = model.training
    try:
      model.eval()
      with torch.no_grad():
        ttloss = 0.
        ttcnt = 0
        seen_any = False
        for batch in loader:
            x,y = _move_batch(batch=batch, device=device)
            loss = compute_lm_loss(model(x) ,y)
            cnt = (y != -100).sum().item()
            ttcnt += cnt
            ttloss += loss.item()*cnt
            seen_any = True
        if not seen_any or ttcnt == 0:
            raise ValueError("empty loader")
        return ttloss / ttcnt
    finally:
        model.train(flag)

def train_for_steps(
    model: nn.Module,
    optimizer: Optimizer,
    train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    num_steps: int,
    *,
    device: str | torch.device = "cpu",
    val_loader: Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None,
    eval_every: int | None = None,
    max_grad_norm: float | None = None,
) -> list[dict[str, int | float]]:
    """TODO 4: perform exactly num_steps optimizer updates, record history.

    Contract:
      - require num_steps >= 1 (ValueError otherwise);
      - if val_loader exists, eval_every must be positive integer;
        if val_loader is None, eval_every must be None (else ValueError);
      - obtain iterator = iter(train_loader); on exhaustion make a NEW iterator;
        if re-iterating is still empty, raise ValueError;
      - call train_step exactly once per global step;
      - append {'step': 1-based int, 'train_loss': float} for every step;
      - on step divisible by eval_every, call evaluate and add
        'val_loss': float to that same history record;
      - return list of records, NOT tensors or printed-only output.

    No checkpoint writing or scheduler today; those belong to later days.
    """
    """TODO 4：执行恰好 num_steps 次优化器更新，并记录历史。

    约定：
      - 要求 num_steps >= 1（否则抛出 ValueError）；
      - 如果 val_loader 存在，eval_every 必须是正整数；
        如果 val_loader 为 None，eval_every 必须为 None（否则抛出 ValueError）；
      - 获取 iterator = iter(train_loader)；耗尽后创建一个新的 iterator；
        如果重新迭代后仍然为空，抛出 ValueError；
      - 每个全局 step 恰好调用一次 train_step；
      - 每个 step 都追加 {'step': 从 1 开始的 int, 'train_loss': float}；
      - 当 step 能被 eval_every 整除时，调用 evaluate，并在同一条历史记录中
        添加 'val_loss': float；
      - 返回记录列表，而不是张量或仅打印的输出。

    今天不写 checkpoint，也不使用 scheduler；这些属于后面的天数。
    """
    if num_steps < 1:
      raise ValueError("num_steps must be >= 1")
    if val_loader is not None:
        if (not isinstance(eval_every, int) or isinstance(eval_every, bool)
                or eval_every <= 0):
            raise ValueError("eval_every must be a positive integer")
    else:
        if eval_every is not None:
            raise ValueError("eval_every must be None when val_loader is None")
    iterator = iter(train_loader)

    def get_batch():
        nonlocal iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            try:
                return next(iterator)
            except StopIteration:
                raise ValueError("train_loader is empty")
    history = []
    for step in range(1, 1+num_steps):
        loss = train_step(model=model,
          optimizer=optimizer,
          batch=get_batch(),
          device=device,
          max_grad_norm=max_grad_norm
        )
        history.append( {'step': step, 'train_loss': loss})
        if val_loader is not None and step % eval_every == 0:
          history[-1]['val_loss']= evaluate(model=model,loader=val_loader,device=device)
    return history