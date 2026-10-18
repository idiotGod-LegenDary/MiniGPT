"""阶段 1 基础 MiniGPT 验证：前向 shape、反向传播、Causal Mask、超长输入。

由最初的 test_model.py 脚本整理为 pytest 用例，断言保持不变。
"""

import pytest
import torch

from model.mini_gpt import MiniGPT

B, L = 4, 16
VOCAB_SIZE = 5000
MAX_SEQ_LEN = 64
D_MODEL = 128
D_FF = 512
NUM_LAYERS = 3
DROPOUT = 0.1


def _make_model(max_seq_len: int = MAX_SEQ_LEN) -> MiniGPT:
    torch.manual_seed(42)
    return MiniGPT(
        vocab_size=VOCAB_SIZE,
        max_seq_len=max_seq_len,
        d_model=D_MODEL,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )


def test_forward_shape_and_backward():
    model = _make_model()

    input_ids = torch.randint(low=0, high=VOCAB_SIZE, size=(B, L))

    logits = model(input_ids)

    assert logits.shape == (B, L, VOCAB_SIZE)

    logits.mean().backward()
    assert model.token_embedding.weight.grad is not None


def test_causal_mask_blocks_future_positions():
    model = _make_model()

    x = torch.randn(2, 5, D_MODEL)
    _, weights = model.net[0].attn(model.net[0].LN1(x))

    future = torch.triu(weights, diagonal=1)

    assert future.abs().max().item() < 1e-6


def test_sequence_length_exceeds_max_raises():
    model = _make_model(max_seq_len=L - 1)

    input_ids = torch.randint(low=0, high=VOCAB_SIZE, size=(B, L))

    with pytest.raises(ValueError):
        model(input_ids)
