"""Run with: python -m demos.demo_training_data (after TODO completion)."""
import torch
from training.data import make_batch, build_loader


def main() -> None:
    tokens = list(range(16))  # toy token IDs, not raw text
    x, y = make_batch(tokens, block_size=4, batch_size=2,
                      generator=torch.Generator().manual_seed(7))
    print("make_batch inputs:", x.tolist())
    print("make_batch labels:", y.tolist())
    loader = build_loader(tokens, block_size=4, batch_size=3,
                          shuffle=False)
    bx, by = next(iter(loader))
    print("DataLoader inputs:", bx.tolist())
    print("DataLoader labels:", by.tolist())
    assert torch.equal(bx[:, 1:], by[:, :-1])
    print("alignment OK")


if __name__ == "__main__":
    main()
