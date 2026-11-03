"""Day 04 演示：通过最小的演示模型验证 Adapter-only 保存和重载。

本例的 LoRA 计算只用于展示接口；学生不需要再实现低秩公式。
正式使用时将 adapter_names 改为你的真实 LoRA 模型参数名。
"""
from __future__ import annotations
import copy
from pathlib import Path
import tempfile
import torch
from torch import nn

from training.adapter_io import (
    set_adapter_trainable, snapshot_named_parameters, make_adapter_package,
    save_adapter_file, read_adapter_file, load_adapter_package,
    compare_parameter_updates,
)


class DemoLoRA(nn.Module):
    """仅用于验证管理接口的最小 LoRA 演示模型。"""
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 3, bias=False)
        self.lora_A = nn.Parameter(torch.randn(2, 4) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(3, 2))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T) @ self.lora_B.T


def main():
    torch.manual_seed(17)
    model = DemoLoRA()
    names = ["lora_A", "lora_B"]
    cfg = {"rank": 2, "alpha": 2.0, "target": "demo"}
    report = set_adapter_trainable(model, names)
    before = snapshot_named_parameters(model)
    base_before = copy.deepcopy(model)

    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
    x = torch.randn(6, 4)
    target = torch.randn(6, 3)
    optimizer.zero_grad(set_to_none=True)
    loss = (model(x) - target).square().mean()
    loss.backward()
    optimizer.step()

    change = compare_parameter_updates(model, before, adapter_names=names)
    package = make_adapter_package(
        model, adapter_names=names, base_model_id="demo-base-v1", adapter_config=cfg)
    with tempfile.TemporaryDirectory() as directory:
        path = save_adapter_file(Path(directory) / "demo_adapter.pt", package)
        loaded = read_adapter_file(path)
        restored = copy.deepcopy(base_before)
        set_adapter_trainable(restored, names)
        load_adapter_package(restored, loaded,
                             expected_base_model_id="demo-base-v1", expected_config=cfg)
        max_diff = (model(x) - restored(x)).abs().max().item()
    print("参数统计:", report)
    print("训练前后变化:", change)
    print("Adapter 文件键:", sorted(package["adapter_state"]))
    print("重载后输出最大差异:", max_diff)
    assert change["base_unchanged"] and change["adapter_updated"]
    assert max_diff == 0.0


if __name__ == "__main__":
    main()
