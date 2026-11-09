"""Week 04 Day 01: causal language-model training-data pipeline.

Keep the existing model/ and tests/ code unchanged. Complete the TODOs below.
"""
from __future__ import annotations

from numbers import Integral
from typing import Sequence

import torch
from torch.utils.data import Dataset, DataLoader


def _to_token_tensor(tokens: Sequence[int] | torch.Tensor) -> torch.Tensor:
    """Convert a 1-D integer token sequence to a CPU torch.long tensor.

    Validation helper is intentionally PROVIDED (not a TODO):
    - reject non-1D input, floating-point tokens and negative IDs;
    - avoid silently changing a non-integer tensor into integer IDs.
    """
    if isinstance(tokens, torch.Tensor):
        if tokens.ndim != 1:
            raise ValueError("tokens must be a 1-D sequence")
        if tokens.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
        ):
            raise TypeError("token IDs must have an integer dtype")
        result = tokens.detach().to(device="cpu", dtype=torch.long).clone()
    else:
        if any(not isinstance(v, Integral) or isinstance(v, bool) for v in tokens):
            raise TypeError("every token ID must be an integer (not bool)")
        result = torch.tensor(list(tokens), dtype=torch.long)
    if bool((result < 0).any()):
        raise ValueError("token IDs must be non-negative")
    return result


def make_batch(
    tokens: Sequence[int] | torch.Tensor,
    block_size: int,
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TODO 1: Randomly sample aligned (inputs, labels) for next-token prediction.

    Args:
        tokens: 1-D contiguous token IDs. At least block_size + 1 IDs.
        block_size: positive context length T (do not exceed data capacity).
        batch_size: positive number B of independently sampled windows.
        generator: optional CPU torch.Generator for reproducible sampling.

    Returns:
        (x, y), both CPU long tensors with shape [B, T]. Each row starts
        at an integer s in [0, len(tokens) - block_size - 1] (inclusive):
        x[row] == tokens[s:s+T]
        y[row] == tokens[s+1:s+T+1]

    Behavior:
        - sample starting positions WITH replacement (duplicates allowed);
        - use torch.randint(..., generator=generator), NOT Python random;
        - reject block_size <= 0 or batch_size <= 0 with ValueError;
        - reject insufficient tokens with ValueError;
        - use _to_token_tensor (provided) to validate the IDs;
        - never mutate the caller's token sequence.
    """
    # TODO 1: Validate sizes, generate starts, and stack aligned windows.
    if block_size <= 0 or batch_size <= 0:
        raise ValueError("parameters error")
    result = _to_token_tensor(tokens)
    if len(result) < block_size + 1:
        raise ValueError("need at least block_size + 1 tokens")
    start = torch.randint(0, len(result) - block_size,(batch_size,),generator=generator)
    tmp = torch.arange(0, block_size)
    x = start[:,None] + tmp
    y = x + 1
    return result[x], result[y] # 要stack干嘛？

class TokenWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Map-style Dataset: one legal start index corresponds to one sample."""

    def __init__(
        self, tokens: Sequence[int] | torch.Tensor, block_size: int
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.tokens = _to_token_tensor(tokens)
        if len(self.tokens) < block_size + 1:
            raise ValueError("need at least block_size + 1 tokens")
        self.block_size = block_size

    def __len__(self) -> int:
        """TODO 2: Return the number of legal starts; type must be int.

        For N tokens and context length T, legal starts are
        0, 1, ..., N - T - 1. No sampling happens here.
        """
        return len(self.tokens) - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """TODO 3: Return one aligned (x, y) of shape [T], dtype long.

        - x = tokens[index:index+T]; y = tokens[index+1:index+T+1].
        - For index < 0 or index >= len(self), raise IndexError.
        - For non-integer index, raise TypeError.
        - No randomness, and do not mutate stored tokens.
        """
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("index type error")
        if index < 0 or index >= len(self):
            raise IndexError("index error")

        return self.tokens[index:index + self.block_size], self.tokens[index + 1: index + 1 + self.block_size]

def build_loader(
    tokens: Sequence[int] | torch.Tensor,
    block_size: int,
    batch_size: int,
    *,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """TODO 4: Build a DataLoader for TokenWindowDataset.

    - use TokenWindowDataset(tokens, block_size) as the dataset;
    - use the given DataLoader parameters by name;
    - return DataLoader, do not eagerly read all batches;
    - default collate_fn should stack [T] samples into [B, T];
    - num_workers >= 0 and batch_size > 0 (otherwise ValueError);
    - keep num_workers=0 by default so the exercise runs on Windows/CPU;
    - generator controls the DataLoader's shuffle order when used.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    ds = TokenWindowDataset(tokens,block_size)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        generator=generator,
    )
