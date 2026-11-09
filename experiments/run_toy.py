"""Week 04 Day 05：MiniGPT 合成 Token 训练的统一运行入口。

使用 Windows CMD，在 MiniGPT 仓库根目录执行：
    python -m experiments.run_toy --steps 12 --output runs/w04d05_a.json
    python -m experiments.run_toy --steps 12 --output runs/w04d05_b.json

这是玩具规模的工程复现实验，不是正式自然语言预训练或 SFT。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import platform
from typing import Any

import torch

from experiments.reporting import summarize_history, save_json_report


@dataclass(frozen=True)
class ExperimentConfig:
    """已提供的实验配置。默认值适合 CPU 小规模测试。"""
    seed: int = 42
    vocab_size: int = 16
    block_size: int = 8
    train_tokens: int = 192
    val_tokens: int = 96
    batch_size: int = 8
    num_steps: int = 12
    eval_every: int = 4
    learning_rate: float = 1e-3
    d_model: int = 32
    d_ff: int = 64
    num_layers: int = 1
    dropout: float = 0.0


def _validate_config(config: ExperimentConfig) -> None:
    """已提供：提前拦截无效配置，而不是训练一半才报错。"""
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config 必须是 ExperimentConfig")
    ints = {
        'vocab_size': config.vocab_size, 'block_size': config.block_size,
        'train_tokens': config.train_tokens, 'val_tokens': config.val_tokens,
        'batch_size': config.batch_size, 'num_steps': config.num_steps,
        'eval_every': config.eval_every, 'd_model': config.d_model,
        'd_ff': config.d_ff, 'num_layers': config.num_layers,
    }
    for name, value in ints.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} 必须是正整数")
    if type(config.seed) is not int or config.seed < 0:
        raise ValueError('seed 必须是非负整数')
    if config.vocab_size < 2:
        raise ValueError('vocab_size 必须至少为 2')
    if config.train_tokens <= config.block_size or config.val_tokens <= config.block_size:
        raise ValueError('训练集和验证集都至少需要 block_size+1 个 Token')
    if not (0 <= config.dropout < 1):
        raise ValueError('dropout 取值必须位于 [0,1)')
    if not (config.learning_rate > 0 and float(config.learning_rate) < float('inf')):
        raise ValueError('learning_rate 必须是正有限数')


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """TODO 4：真正调用已有 MiniGPT、DataLoader 与训练循环，生成报告。

    输入：通过 _validate_config() 校验的 ExperimentConfig。
    输出：严格包含以下七个字段的字典：
      config: asdict(config)
      environment: {'python': str, 'torch': str, 'platform': str, 'device':'cpu'}
      baseline_val_loss: float  # 训练前 evaluate()，不修改模型
      final_val_loss: float     # 训练后 evaluate()
      history: list[dict]      # 现有 train_for_steps() 的结果
      summary: dict            # summarize_history(history) 的结果
      data_note: str           # 明确声明是周期性合成数据

    操作步骤与工程契约：
      1. _validate_config(config)，并 torch.manual_seed(config.seed)。
      2. 生成长度 train_tokens+val_tokens 的 Long Tensor：
            tokens = torch.arange(total, dtype=torch.long) % config.vocab_size
         这是周期性合成数据，不是 Tokenizer 处理的真实文本。
      3. 【先切原始 Token 再建窗口】按 train_tokens 切开；
         不能先对整条序列建窗口再随机划分，以免窗口跨边界。
      4. 调用既有 training.data.build_loader 构建：
         训练 loader: shuffle=True，generator=torch.Generator().manual_seed(seed)
         验证 loader: shuffle=False；两个 loader 均 num_workers=0。
      5. 按配置新建基础 model.mini_gpt.MiniGPT，构造参数：
         vocab_size、max_seq_len=block_size、d_model、d_ff、num_layers、dropout。
         只在 CPU 运行。建立 AdamW(model.parameters(),lr=learning_rate,
         weight_decay=0.0)。
      6. 用已有 training.loop.evaluate(model,val_loader) 得到 baseline。
      7. 使用已有 training.loop.train_for_steps(...) 训练 num_steps 次，
         val_loader=验证 loader，eval_every=配置值，获取 history。
      8. 再调用 evaluate 得到 final_val_loss；使用 summarize_history。
      9. 记录 asdict(config) 和 Python/PyTorch/平台版本（platform 模块）；
         data_note 必须如实写明“周期性合成 Token，无自然语言泛化结论”。
     10. 返回普通 JSON 可序列化对象，不在本函数里保存文件；
         不更改已有数据、模型、训练模块源码。

    注意：记录中的任何 Loss 必须来自实际模型运行，不得编造。
    """
    # TODO 4：将已有模块串成完整实验入口。
    _validate_config(config)
    torch.manual_seed(config.seed)
    tokens = torch.arange(config.train_tokens + config.val_tokens,dtype = torch.long) % config.vocab_size
    train_tokens, val_tokens = tokens[:config.train_tokens], tokens[config.train_tokens:] 
    from training import data,loop
    train_loader =data.build_loader(
        tokens=train_tokens,
        block_size=config.block_size,
        batch_size=config.batch_size,
        shuffle = True,
        generator = torch.Generator().manual_seed(config.seed),
        num_workers = 0
    )
    val_loader = data.build_loader(
        tokens=val_tokens,
        block_size=config.block_size,
        batch_size=config.batch_size,
        shuffle = False,
        generator = torch.Generator().manual_seed(config.seed),
        num_workers = 0
    )
    from model import mini_gpt
    model = mini_gpt.MiniGPT(
        vocab_size=config.vocab_size,
        max_seq_len=config.block_size,
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        dropout = config.dropout
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.0
    )
    baseline = loop.evaluate(model,val_loader)
    history = loop.train_for_steps(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        num_steps= config.num_steps,
        val_loader=val_loader,
        eval_every=config.eval_every,
    ) 
    final_val_loss = loop.evaluate(model,val_loader)
    import sys
    return {
        'config': asdict(config),
        'environment': {'python': sys.version, 'torch': torch.__version__, 'platform': platform.system(), 'device':'cpu'},
        'baseline_val_loss':baseline,  # 训练前 evaluate()，不修改模型
        'final_val_loss':final_val_loss ,    # 训练后 evaluate()
        'history': history ,     # 现有 train_for_steps() 的结果
        'summary': summarize_history(history)      ,     # summarize_history(history) 的结果
        'data_note': '周期性合成 Token，无自然语言泛化结论'      ,   # 明确声明是周期性合成数据
    }


def main() -> None:
    """已提供：命令行入口，不需要修改。"""
    parser = argparse.ArgumentParser(description='MiniGPT 周期性 Token 训练复现实验')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--steps', type=int, default=12, help='训练更新次数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--output', default='runs/week04_day05.json', help='JSON 输出路径')
    args = parser.parse_args()

    config = ExperimentConfig(seed=args.seed, num_steps=args.steps, learning_rate=args.lr)
    record = run_experiment(config)
    output = save_json_report(args.output, record)
    print('已写出实验报告：', output)
    print('训练前验证 Loss：', record['baseline_val_loss'])
    print('训练后验证 Loss：', record['final_val_loss'])
    print('训练步数：', record['summary']['num_steps'])


if __name__ == '__main__':
    main()
