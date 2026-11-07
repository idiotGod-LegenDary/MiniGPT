"""Day 07：一键运行全参 / LoRA 对照、保存 Adapter 和 JSON 报告。

Windows CMD：
    python -m demos.demo_full_lora_comparison --steps 6
    python -m demos.demo_full_lora_comparison --steps 6 --seed 42

产物默认放入 runs/day07/，不会覆盖之前的任何模型或训练源码。
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from experiments.compare_full_lora import ComparisonConfig, run_comparison
from experiments.reporting import save_json_report


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniGPT 全参数与 LoRA 受控实验")
    parser.add_argument("--steps", type=int, default=6, help="两路训练各自更新次数")
    parser.add_argument("--seed", type=int, default=42, help="基础模型初始化与数据顺序的种子")
    parser.add_argument("--lr", type=float, default=2e-3, help="两路共同学习率")
    parser.add_argument("--rank", type=int, default=2, help="LoRA 低秩维度")
    parser.add_argument("--adapter", default="runs/day07/adapter.pt", help="Adapter 保存位置")
    parser.add_argument("--report", default="runs/day07/comparison.json", help="JSON 结果位置")
    args = parser.parse_args()

    cfg = replace(
        ComparisonConfig(), seed=args.seed, num_steps=args.steps,
        eval_every=min(3, args.steps), learning_rate=args.lr, rank=args.rank,
    )
    report = run_comparison(cfg, adapter_path=args.adapter)
    path = save_json_report(args.report, report)
    print("实验报告：", path)
    for name, title in (("full", "全参数训练"), ("lora", "LoRA 训练")):
        arm = report[name]
        print(f"{title}：训练前验证 Loss={arm['baseline_val_loss']:.6f}，"
              f"训练后验证 Loss={arm['final_val_loss']:.6f}，"
              f"可训练参数={arm['stats']['trainable']}")
    print("初始输出差异：", report["initial_max_abs_diff"])
    print("Adapter 恢复输出差异：", report["adapter"]["reload_max_abs_diff"])
    print("合并权重输出差异：", report["adapter"]["merge_max_abs_diff"])
    print("基础权重不变：", report["adapter"]["base_unchanged"])
    print("Adapter 更新：", report["adapter"]["adapter_updated"])


if __name__ == "__main__":
    main()
