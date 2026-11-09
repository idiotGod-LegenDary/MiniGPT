# Week 04 · Day 01 — MiniGPT data pipeline

## Integration

Copy `training/`, `demos/demo_training_data.py`, and
`tests/test_training_data.py` into the root of the existing MiniGPT repo.
Do not replace `model/`, existing tests, or the original README.
Existing requirements.txt includes PyTorch and pytest; no additional
third-party dependency is needed. Work on a new Git branch.

```bash
git switch -c feat/w04d01-data-pipeline
python -m pytest tests/test_training_data.py -q
python -m demos.demo_training_data
python -m pytest tests/ -q
```

## Data contract

- 1-D token IDs, non-negative integers, dtype `torch.long` after validation.
- For context length `T`, each sample uses `T+1` consecutive IDs.
- `x` is the first `T` IDs, `y` is the last `T` IDs.
- `make_batch`: sample starts with replacement; output `[B,T]`.
- `TokenWindowDataset`: enumerate every legal start once, `len=N-T`.
- `build_loader`: batch dataset samples using PyTorch's default collate.

## Independent mastery check (do not copy answer code)

1. Explain why a data sequence of length T is insufficient for a sample of
   length T when predicting next tokens.
2. Explain the difference between random `make_batch` sampling with
   replacement and `DataLoader(shuffle=True)` over a map-style Dataset.
3. For N=12, T=5, calculate legal starts and Dataset length.
4. Explain the purpose of `drop_last` and why setting it to True can
   produce no batches when the dataset is smaller than batch_size.
5. Change batch_size from 2 to 3 and predict the new shapes before running.

Do not claim a full GPT training pipeline is complete after this day.
`train_step`, optimizer, checkpoint, and logging are separate later tasks.
