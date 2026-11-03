"""Week 04 Day 04：Adapter IO 自动化接口测试（不是答案代码）。"""
from __future__ import annotations
import copy
import pytest
import torch
from torch import nn

from training.adapter_io import (
    set_adapter_trainable, make_adapter_package, load_adapter_package,
    compare_parameter_updates, snapshot_named_parameters,
    save_adapter_file, read_adapter_file,
)

ADAPTER = ("projection.lora_A", "projection.lora_B")
CONFIG = {"rank": 2, "alpha": 4.0, "target": "projection"}
BASE_ID = "toy-base-v1"


class TinyLoRALinear(nn.Module):
    """测试专用微型 LoRA：不要求学生重新实现该算法。"""
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 3, bias=False)
        self.lora_A = nn.Parameter(torch.randn(2, 4) * 0.05)
        self.lora_B = nn.Parameter(torch.zeros(3, 2))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T) @ self.lora_B.T


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = TinyLoRALinear()

    def forward(self, x):
        return self.projection(x)


@pytest.fixture
def model():
    torch.manual_seed(11)
    m = TinyModel()
    set_adapter_trainable(m, ADAPTER)
    return m


def train_once(model):
    opt = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.5)
    x = torch.randn(8, 4)
    target = torch.randn(8, 3)
    opt.zero_grad(set_to_none=True)
    loss = (model(x) - target).square().mean()
    loss.backward()
    opt.step()


def get_package(model):
    return make_adapter_package(model, adapter_names=ADAPTER,
                                base_model_id=BASE_ID, adapter_config=CONFIG)


def test_freeze_only_adapter_trainable(model):
    report = set_adapter_trainable(model, ADAPTER)
    params = dict(model.named_parameters())
    assert all(params[n].requires_grad == (n in ADAPTER) for n in params)
    assert report["total"] == sum(p.numel() for p in params.values())
    assert report["trainable"] == sum(params[n].numel() for n in ADAPTER)
    assert report["frozen"] == report["total"] - report["trainable"]
    assert report["ratio"] == pytest.approx(report["trainable"] / report["total"])


@pytest.mark.parametrize("names", [[], ["missing"], [ADAPTER[0], ADAPTER[0]], "foo"])
def test_freeze_bad_names_does_not_mutate(model, names):
    before = {n: p.requires_grad for n,p in model.named_parameters()}
    with pytest.raises((ValueError, TypeError)):
        set_adapter_trainable(model, names)
    assert before == {n: p.requires_grad for n,p in model.named_parameters()}


def test_package_contains_only_adapter_and_independent_snapshot(model):
    config = dict(CONFIG)
    pkg = make_adapter_package(model, adapter_names=ADAPTER,
                               base_model_id=BASE_ID, adapter_config=config)
    assert set(pkg) == {"format_version", "base_model_id", "adapter_config", "adapter_names", "adapter_state"}
    assert pkg["format_version"] == 1
    assert pkg["adapter_names"] == sorted(ADAPTER)
    assert set(pkg["adapter_state"]) == set(ADAPTER)
    for k,v in pkg["adapter_state"].items():
        assert v.device.type == "cpu" and not v.requires_grad
        assert torch.equal(v, dict(model.named_parameters())[k].detach())
    config["rank"] = 99
    with torch.no_grad():
        model.projection.lora_A.add_(1)
    assert pkg["adapter_config"]["rank"] == 2
    assert not torch.equal(pkg["adapter_state"][ADAPTER[0]], model.projection.lora_A)


def test_package_requires_frozen_base_and_trainable_adapter(model):
    model.projection.base.weight.requires_grad_(True)
    with pytest.raises(ValueError):
        get_package(model)
    model.projection.base.weight.requires_grad_(False)
    model.projection.lora_A.requires_grad_(False)
    with pytest.raises(ValueError):
        get_package(model)


@pytest.mark.parametrize("bad_config", [None, {}, {"rank": [2]}, {"": 2}])
def test_package_rejects_bad_metadata(model, bad_config):
    with pytest.raises((ValueError, TypeError)):
        make_adapter_package(model, adapter_names=ADAPTER,
                             base_model_id=BASE_ID, adapter_config=bad_config)


def test_package_rejects_empty_base_id(model):
    with pytest.raises(ValueError):
        make_adapter_package(model, adapter_names=ADAPTER,
                             base_model_id=" ", adapter_config=CONFIG)


def test_disk_roundtrip_and_output_equivalence(model, tmp_path):
    base = copy.deepcopy(model)
    train_once(model)
    pkg = get_package(model)
    path = save_adapter_file(tmp_path / "adapter_only.pt", pkg)
    assert path.is_file()
    disk = read_adapter_file(path)
    target = copy.deepcopy(base)
    load_adapter_package(target, disk,
                         expected_base_model_id=BASE_ID, expected_config=CONFIG)
    x = torch.randn(6, 4)
    assert torch.allclose(model(x), target(x), atol=0, rtol=0)
    assert set(disk["adapter_state"]) == set(ADAPTER)


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(format_version=999),
    lambda p: p.update(base_model_id="wrong"),
    lambda p: p.update(adapter_config={"rank": 3}),
    lambda p: p["adapter_state"][ADAPTER[0]].resize_(1),
    lambda p: p["adapter_state"].update({ADAPTER[0]: torch.ones(2,4,dtype=torch.float64)}),
    lambda p: p["adapter_state"].update({"fake": torch.zeros(1)}),
    lambda p: p.pop("adapter_names"),
])
def test_load_rejects_bad_metadata_shape_dtype_before_mutation(model, mutate):
    pkg = get_package(model)
    mutate(pkg)
    before = snapshot_named_parameters(model)
    with pytest.raises((ValueError, TypeError, KeyError)):
        load_adapter_package(model, pkg,
                             expected_base_model_id=BASE_ID, expected_config=CONFIG)
    after = snapshot_named_parameters(model)
    assert all(torch.equal(before[n], after[n]) for n in before)


def test_load_preserves_parameter_identity_and_trainability(model):
    pkg = get_package(model)
    original_ids = {n: id(p) for n,p in model.named_parameters()}
    original_grad = {n: p.requires_grad for n,p in model.named_parameters()}
    with torch.no_grad():
        model.projection.lora_B.add_(1)
    load_adapter_package(model, pkg,
                         expected_base_model_id=BASE_ID, expected_config=CONFIG)
    assert {n: id(p) for n,p in model.named_parameters()} == original_ids
    assert {n: p.requires_grad for n,p in model.named_parameters()} == original_grad
    assert torch.equal(model.projection.lora_B, pkg["adapter_state"][ADAPTER[1]])


def test_compare_no_update(model):
    before = snapshot_named_parameters(model)
    result = compare_parameter_updates(model, before, adapter_names=ADAPTER)
    assert result == {"base_unchanged": True, "adapter_updated": False,
                      "changed_base": [], "changed_adapter": []}


def test_compare_real_adapter_step(model):
    before = snapshot_named_parameters(model)
    train_once(model)
    result = compare_parameter_updates(model, before, adapter_names=ADAPTER)
    assert result["base_unchanged"] is True
    assert result["adapter_updated"] is True
    assert result["changed_base"] == []
    assert ADAPTER[1] in result["changed_adapter"]  # 初始 B 为零，首步更新 B


def test_compare_detects_base_change(model):
    before = snapshot_named_parameters(model)
    with torch.no_grad():
        model.projection.base.weight.add_(1)
    result = compare_parameter_updates(model, before, adapter_names=ADAPTER)
    assert result["base_unchanged"] is False
    assert result["changed_base"] == ["projection.base.weight"]
    assert result["adapter_updated"] is False


def test_compare_bad_snapshot(model):
    before = snapshot_named_parameters(model)
    before.pop("projection.base.weight")
    with pytest.raises(ValueError):
        compare_parameter_updates(model, before, adapter_names=ADAPTER)
