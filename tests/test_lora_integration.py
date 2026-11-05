"""Week 04 · Day 06：LoRA 与真实 MiniGPT 的集成测试。直接运行 pytest。"""

import copy
import math

import pytest
import torch
from torch import nn

from model.attention import CausalSelfAttention
from model.mini_gpt import MiniGPT
from training.lora_integration import LoRALinear, inject_lora_attention
from training.adapter_io import snapshot_named_parameters, compare_parameter_updates
from training.loop import train_step


def make_model(*, layers=2, dropout=0.0):
    torch.manual_seed(17)
    return MiniGPT(vocab_size=16, max_seq_len=8, d_model=16, d_ff=32,
                   num_layers=layers, dropout=dropout)


def make_batch():
    x = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)
    y = torch.tensor([[1, 2, 3, 4], [2, 1, 0, 3]], dtype=torch.long)
    return x, y


def test_linear_initial_parameters_and_identity():
    base = nn.Linear(7, 5)
    original_weight = base.weight.detach().clone()
    original_bias = base.bias.detach().clone()
    wrapped = LoRALinear(base, rank=3, alpha=6.0)
    assert wrapped.base is base
    assert tuple(wrapped.lora_A.shape) == (3, 7)
    assert tuple(wrapped.lora_B.shape) == (5, 3)
    assert torch.count_nonzero(wrapped.lora_B) == 0
    assert torch.count_nonzero(wrapped.lora_A) > 0
    assert wrapped.lora_A.requires_grad and wrapped.lora_B.requires_grad
    assert not wrapped.base.weight.requires_grad
    assert not wrapped.base.bias.requires_grad
    assert torch.equal(original_weight, base.weight)
    assert torch.equal(original_bias, base.bias)
    assert wrapped.scale == 2.0


@pytest.mark.parametrize("rank,alpha,dropout", [
    (0,8.0,0.0), (-1,8.0,0.0), (True,8.0,0.0), (8,8.0,0.0),
    (2,0.0,0.0), (2,math.inf,0.0), (2,8.0,-0.1), (2,8.0,1.0),
    (2,8.0,math.nan), (2,True,0.0),
])
def test_invalid_linear_hyperparameters(rank, alpha, dropout):
    base = nn.Linear(7, 5)
    with pytest.raises(ValueError):
        LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout)
    assert base.weight.requires_grad


def test_invalid_base_type():
    with pytest.raises(TypeError):
        LoRALinear(nn.ReLU())


def test_linear_zero_delta_preserves_original_output():
    base = nn.Linear(8, 6)
    x = torch.randn(2, 4, 8)
    before = base(x).detach().clone()
    wrapped = LoRALinear(base, rank=2, alpha=4.0, dropout=0.0)
    got = wrapped(x)
    assert got.shape == (2, 4, 6)
    torch.testing.assert_close(before, got, atol=0, rtol=0)


def test_linear_trainable_gradient_and_update():
    base = nn.Linear(8, 6, bias=False)
    wrapped = LoRALinear(base, rank=2, alpha=4.0)
    x = torch.randn(3, 8)
    before = base.weight.detach().clone()
    opt = torch.optim.SGD([wrapped.lora_A, wrapped.lora_B], lr=0.2)
    opt.zero_grad()
    wrapped(x).square().mean().backward()
    assert wrapped.lora_B.grad is not None
    assert wrapped.lora_B.grad.abs().sum() > 0
    assert base.weight.grad is None
    opt.step()
    assert torch.equal(before, base.weight)
    assert torch.count_nonzero(wrapped.lora_B) > 0


def test_real_minigpt_injection_and_trainability():
    model = make_model(layers=2)
    named_before = dict(model.named_modules())
    wq = named_before["net.0.attn.W_q"]
    names = inject_lora_attention(model, rank=2, alpha=4.0)
    assert len(names) == 8
    assert names == sorted(names)
    assert "net.0.attn.W_q.lora_A" in names
    assert "net.1.attn.W_v.lora_B" in names
    assert isinstance(model.net[0].attn.W_q, LoRALinear)
    assert isinstance(model.net[0].attn.W_v, LoRALinear)
    assert model.net[0].attn.W_q.base is wq
    assert isinstance(model.net[0].attn.W_k, nn.Linear)
    assert isinstance(model.net[0].attn.out_proj, nn.Linear)
    for name, p in model.named_parameters():
        assert p.requires_grad == (name in names), name
    x, _ = make_batch()
    assert model(x).shape == (2, 4, 16)


def test_injected_model_matches_base_at_step_zero():
    model = make_model(dropout=0.0)
    model.eval()
    x, _ = make_batch()
    baseline = model(x).detach().clone()
    inject_lora_attention(model, rank=2, alpha=4.0, dropout=0.0)
    model.eval()
    torch.testing.assert_close(baseline, model(x), atol=1e-6, rtol=1e-6)


def test_real_minigpt_optimizer_step_only_updates_adapter():
    model = make_model(layers=1)
    names = inject_lora_attention(model, rank=2, alpha=4.0)
    before = snapshot_named_parameters(model)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=1e-2, weight_decay=0.0)
    loss = train_step(model, opt, make_batch())
    assert isinstance(loss, float) and math.isfinite(loss)
    report = compare_parameter_updates(model, before, adapter_names=names)
    assert report["base_unchanged"] is True
    assert report["adapter_updated"] is True
    assert report["changed_base"] == []


def test_double_injection_rejected_without_changing_structure():
    model = make_model(layers=1)
    names = inject_lora_attention(model, rank=2, alpha=4)
    first = model.net[0].attn.W_q
    with pytest.raises(ValueError):
        inject_lora_attention(model, rank=2, alpha=4)
    assert model.net[0].attn.W_q is first
    assert len([n for n, _ in model.named_parameters() if n.endswith(".lora_A")]) == 2


@pytest.mark.parametrize("target_modules", [(), ("W_q", "W_q"), ("W_bad",), ("W_q", "W_bad")])
def test_bad_target_names_no_partial_modification(target_modules):
    model = make_model(layers=2)
    before = {n: p.requires_grad for n, p in model.named_parameters()}
    old_q = model.net[0].attn.W_q
    with pytest.raises(ValueError):
        inject_lora_attention(model, rank=2, alpha=4, target_modules=target_modules)
    assert model.net[0].attn.W_q is old_q
    assert {n: p.requires_grad for n, p in model.named_parameters()} == before


def test_bad_rank_no_partial_injection():
    model = make_model(layers=2)
    with pytest.raises(ValueError):
        inject_lora_attention(model, rank=32)
    assert isinstance(model.net[0].attn.W_q, nn.Linear)
    assert isinstance(model.net[1].attn.W_v, nn.Linear)


def test_no_attention_no_injection():
    model = nn.Sequential(nn.Linear(16, 16))
    with pytest.raises(ValueError):
        inject_lora_attention(model, rank=2)


def test_custom_target_and_nonzero_drop_rate():
    model = make_model(layers=1)
    names = inject_lora_attention(model, rank=2, alpha=2.0, dropout=0.1,
                                  target_modules=("out_proj",))
    assert names == ["net.0.attn.out_proj.lora_A", "net.0.attn.out_proj.lora_B"]
    assert model.net[0].attn.out_proj.lora_dropout.p == 0.1
    assert isinstance(model.net[0].attn.W_q, nn.Linear)


def test_adapter_only_excludes_base_weight():
    model = make_model(layers=1)
    names = inject_lora_attention(model, rank=2, alpha=4.0)
    from training.adapter_io import make_adapter_package
    package = make_adapter_package(model, adapter_names=names,
                                   base_model_id="minigpt-day06-demo",
                                   adapter_config={"rank":2, "alpha":4.0,"dropout":0.0})
    assert set(package["adapter_state"]) == set(names)
    assert not any(k.endswith("base.weight") for k in package["adapter_state"])
    assert not any(k.endswith("base.bias") for k in package["adapter_state"])


def test_isolated_check_preserves_base_after_failure():
    model = make_model(layers=2)
    model.net[1].attn.W_v = nn.ReLU()
    old_q = model.net[0].attn.W_q
    with pytest.raises(ValueError):
        inject_lora_attention(model, rank=2, target_modules=("W_q", "W_v"))
    assert model.net[0].attn.W_q is old_q
