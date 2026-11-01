"""Run from MiniGPT root: python -m demos.demo_training_loop.

Uses the REAL existing MiniGPT constructor and Week04 Day01 data loader.
"""
import torch
from model.mini_gpt import MiniGPT
from training.data import build_loader
from training.loop import train_for_steps


def main() -> None:
    torch.manual_seed(42)
    vocab = 16
    model = MiniGPT(
        vocab_size=vocab,
        max_seq_len=8,
        d_model=32,
        d_ff=64,
        num_layers=1,
        dropout=0.0,
    ).to('cpu')
    # Separate raw token streams BEFORE creating sliding-window datasets.
    train_tokens = [(i * 3 + 1) % vocab for i in range(256)]
    valid_tokens = [(i * 5 + 2) % vocab for i in range(80)]
    train_loader = build_loader(
        train_tokens, block_size=8, batch_size=8,
        shuffle=True, generator=torch.Generator().manual_seed(7),
    )
    valid_loader = build_loader(valid_tokens,block_size=8,batch_size=8,shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=0.0)
    history = train_for_steps(
        model,optimizer,train_loader,num_steps=12,
        val_loader=valid_loader,eval_every=4,max_grad_norm=1.0,
    )
    print('History records (toy data, not a real SFT benchmark):')
    for record in history:
        print(record)
    assert len(history)==12
    assert [h['step'] for h in history] == list(range(1,13))


if __name__=='__main__':
    main()
