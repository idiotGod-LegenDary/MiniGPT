"""Day 07：全参数与 LoRA 对照实验的自动化测试。只有新增文件。"""

import copy
import json
import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from experiments.compare_full_lora import (
    ComparisonConfig, _adapter_config, _validate_config,
    build_model_pair, build_shared_batches,
    merge_lora_for_inference, run_comparison,
)
from training.lora_integration import LoRALinear


@pytest.fixture
def small():
    """缩小训练规模，避免 CPU 测试耗时过长。"""
    return ComparisonConfig(
        seed=17, vocab_size=12, block_size=5, train_tokens=42,
        val_tokens=27, batch_size=3, num_steps=3, eval_every=1,
        d_model=12, d_ff=24, num_layers=1, rank=2,
        alpha=4.0, learning_rate=3e-3,
    )


@pytest.mark.parametrize("changes", [
    {"seed": True},
    {"num_steps": 0},
    {"eval_every": 99},
    {"dropout": 0.2},
    {"lora_dropout": 0.1},
    {"rank": 99},
    {"learning_rate": float("nan")},
    {"target_modules": ("W_q", "W_q")},
    {"target_modules": ("not_a_projection",)},
])
def test_invalid_config_rejected(small, changes):
    with pytest.raises((ValueError, TypeError)):
        _validate_config(replace(small, **changes))


def test_model_pair_initial_logits_are_identical(small):
    full, lora, names = build_model_pair(small)
    assert len(names) == 2 * len(small.target_modules) * small.num_layers
    assert names == sorted(names)
    assert all(p.requires_grad for p in full.parameters())
    full.eval()
    lora.eval()
    x = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    with torch.no_grad():
        assert torch.allclose(full(x), lora(x), atol=1e-6, rtol=0)
    assert sum(p.numel() for p in lora.parameters() if p.requires_grad) < sum(p.numel() for p in lora.parameters())


def test_model_pair_repeatable(small):
    full1, _, names1 = build_model_pair(small)
    full2, _, names2 = build_model_pair(small)
    assert names1 == names2
    for (na, pa), (nb, pb) in zip(full1.named_parameters(), full2.named_parameters()):
        assert na == nb
        assert torch.equal(pa, pb)


def test_model_pair_has_same_base_weights(small):
    full, lora, _ = build_model_pair(small)
    lora_params = dict(lora.named_parameters())
    for name, p in full.named_parameters():
        wrapped_name = name
        for target in small.target_modules:
            wrapped_name = wrapped_name.replace(f".{target}.", f".{target}.base.")
        assert torch.equal(p, lora_params[wrapped_name]), name


def test_shared_batches_are_aligned_and_repeatable(small):
    train1, val1 = build_shared_batches(small)
    train2, val2 = build_shared_batches(small)
    assert len(train1) == small.num_steps
    assert val1
    for collection in (train1, val1):
        for x, y in collection:
            assert x.dtype == y.dtype == torch.long
            assert x.ndim == y.ndim == 2
            assert x.shape == y.shape
            assert x.shape[1] == small.block_size
            assert torch.equal(y, (x + 1) % small.vocab_size)
    for a, b in zip(train1, train2):
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    for a, b in zip(val1, val2):
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_shared_batches_restart_short_loader(small):
    cfg = replace(small, train_tokens=8, block_size=5, num_steps=11)
    train, val = build_shared_batches(cfg)
    assert len(train) == 11 and len(val) > 0


def test_merge_rejects_unadapted_model(small):
    full, _, _ = build_model_pair(small)
    with pytest.raises(ValueError):
        merge_lora_for_inference(full)
    with pytest.raises(TypeError):
        merge_lora_for_inference("不是模型")


def test_merge_is_independent_and_output_matches(small):
    _, lora, _ = build_model_pair(small)
    # 手动制造非零 LoRA 增量，避免初始 B=0 让测试没有辨别力。
    with torch.no_grad():
        for module in lora.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.fill_(0.05)
    before = {name: p.detach().clone() for name, p in lora.named_parameters()}
    lora.train()
    merged = merge_lora_for_inference(lora)
    assert lora.training is True
    assert merged.training is False
    assert not any(isinstance(m, LoRALinear) for m in merged.modules())
    assert any(isinstance(m, LoRALinear) for m in lora.modules())
    x = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
    lora.eval()
    with torch.no_grad():
        assert torch.allclose(lora(x), merged(x), atol=1e-5, rtol=1e-5)
    for name, p in lora.named_parameters():
        assert torch.equal(p, before[name])
    # 比较前恢复训练模式不影响原先“合并时未改变输入模式”的断言。


def test_adapter_config_only_primitive_values(small):
    cfg = _adapter_config(small)
    assert cfg["rank"] == small.rank
    assert all(type(v) in (int, float, str, bool) for v in cfg.values())


def test_full_comparison_integrated(tmp_path, small):
    adapter_file = tmp_path / "nested" / "adapter.pt"
    report = run_comparison(small, adapter_path=adapter_file)
    assert adapter_file.is_file()
    assert report.keys() == {"config", "data_note", "initial_max_abs_diff", "full", "lora", "adapter"}
    assert report["config"]["num_steps"] == small.num_steps
    assert "合成" in report["data_note"]
    assert report["initial_max_abs_diff"] < 1e-5
    for mode in ("full", "lora"):
        r = report[mode]
        assert len(r["history"]) == small.num_steps
        assert all(math.isfinite(float(row["train_loss"])) for row in r["history"])
        assert math.isfinite(r["baseline_val_loss"])
        assert math.isfinite(r["final_val_loss"])
    assert report["full"]["stats"]["trainable_ratio"] == 1.0
    assert report["lora"]["stats"]["trainable_ratio"] < 1.0
    adapter = report["adapter"]
    assert adapter["base_unchanged"] and adapter["adapter_updated"]
    assert adapter["changed_base"] == []
    assert adapter["adapter_tensor_count"] == 2 * len(small.target_modules) * small.num_layers
    assert adapter["adapter_file_bytes"] == adapter_file.stat().st_size > 0
    assert adapter["reload_max_abs_diff"] < 1e-5
    assert adapter["merge_max_abs_diff"] < 1e-5
    json.dumps(report, allow_nan=False)
    package = torch.load(adapter_file, map_location="cpu", weights_only=True)
    assert set(package["adapter_state"]) == set(package["adapter_names"])
    assert not any("base.weight" in n for n in package["adapter_state"])


def test_same_config_repeatable(tmp_path, small):
    cfg = replace(small, num_steps=2, eval_every=1)
    r1 = run_comparison(cfg, adapter_path=tmp_path / "first.pt")
    r2 = run_comparison(cfg, adapter_path=tmp_path / "second.pt")
    assert r1["full"]["history"] == r2["full"]["history"]
    assert r1["lora"]["history"] == r2["lora"]["history"]
    assert r1["full"]["final_val_loss"] == r2["full"]["final_val_loss"]
    assert r1["lora"]["final_val_loss"] == r2["lora"]["final_val_loss"]
