"""Day 06：在真实 MiniGPT 上展示 LoRA 注入和单步参数更新。

使用：python -m demos.demo_lora_integration
这是结构与梯度的集成演示，不是自然语言 SFT 评测。
"""

import torch

from model.mini_gpt import MiniGPT
from training.adapter_io import compare_parameter_updates, snapshot_named_parameters
from training.lora_integration import inject_lora_attention
from training.loop import train_step


def main() -> None:
    torch.manual_seed(42)
    model = MiniGPT(vocab_size=16, max_seq_len=8, d_model=32,
                    d_ff=64, num_layers=2, dropout=0.0)
    model.eval()
    x = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    y = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], dtype=torch.long)
    original_output = model(x).detach().clone()

    # TODO 3 完成后，这里会真正替换 MiniGPT 的 Attention 投影。
    adapter_names = inject_lora_attention(model, rank=4, alpha=8.0,
                                          dropout=0.0,
                                          target_modules=("W_q", "W_v"))
    model.eval()
    delta_at_init = (model(x) - original_output).abs().max().item()
    print("注入 LoRA 参数数量（Tensor 个数）：", len(adapter_names))
    print("注入后初始化输出最大差异：", delta_at_init)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("总参数量：", total)
    print("可训练参数量：", trainable)
    print("可训练参数占比：", round(100 * trainable / total, 4), "%")

    before = snapshot_named_parameters(model)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=1e-2, weight_decay=0.0
    )
    loss = train_step(model, optimizer, (x, y))
    report = compare_parameter_updates(model, before, adapter_names=adapter_names)
    print("单步训练 Loss：", loss)
    print("基础权重完全不变：", report["base_unchanged"])
    print("至少一个 Adapter 参数更新：", report["adapter_updated"])
    print("异常更新的基础参数：", report["changed_base"])
    assert delta_at_init < 1e-5
    assert report["base_unchanged"] and report["adapter_updated"]


if __name__ == "__main__":
    main()
