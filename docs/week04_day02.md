# Week 04 · Day 02 — MiniGPT training loop

## Integration

Merge ONLY the new files under `training/`, `tests/`, `demos/` and `docs/` into the EXISTING MiniGPT root.
Keep all Day 01 files (`training/data.py`) and all original model/tests. Do not replace `model/`.

```
MiniGPT/
├─ model/mini_gpt.py                    # original
├─ training/__init__.py                 # Day01 existing
├─ training/data.py                     # Day01 existing
├─ training/loop.py                     # Day02 4 TODOs
├─ tests/test_training_data.py          # Day01 existing
├─ tests/test_training_loop.py          # Day02 tests
├─ demos/demo_training_loop.py          # Day02 demo
└─ docs/week04_day02.md                # this guide
```

## Windows CMD commands

```
git status
git switch -c feat/w04d02-training-loop
python -m pytest tests/test_training_loop.py -k ce -v
python -m pytest tests/test_training_loop.py -k train_step -v
python -m pytest tests/test_training_loop.py -k evaluate -v
python -m pytest tests/test_training_loop.py -k for_steps -v
python -m pytest tests/test_training_loop.py -q
python -m demos.demo_training_loop
python -m pytest tests/ -q
```

The four TODOs must initially fail. No reference implementation is included in this student archive.

## Interface contracts

- `compute_lm_loss(logits[B,T,V],labels[B,T],ignore_index=-100)` => scalar differentiable loss.
- `train_step(model,opt,(x,y),device='cpu',max_grad_norm=None)` => one update, Python float loss.
- `evaluate(model,loader,device='cpu')` => token-weighted float mean loss; no graph/no updates; restore original model mode.
- `train_for_steps(model,opt,train_loader,num_steps,device='cpu',val_loader=None,eval_every=None,max_grad_norm=None)` => list of per-step dicts with step/train_loss and optional val_loss. Re-iterate on exhaustion.

## Reminders

- Day01 shifted labels already. Never double shift them.
- model.to(device) is the CALLER's responsibility. Start with CPU.
- `eval()` does not disable gradient; use `torch.no_grad()` for evaluation.
- Never hold onto `loss` graph tensors inside `history`: record floats.
- Validation loss must weight by unmasked target token counts, not batch counts.
- `num_steps` is optimizer update count; dataloader exhaustion is not a stopping condition.
- Toy synthetic tokens are for smoke tests only, not real training claims.

## Independent mastery check

1. Why does the toy training example set `dropout=0.0`? What changes if dropout is nonzero?
2. Why should `evaluate` restore the incoming train/eval mode even on failure?
3. Compute the correct validation mean for batches with 12 valid tokens at loss 2 and 3 tokens at loss 5.
4. What would break if history appended the original `loss` tensor instead of `.item()`?
5. When `num_steps` exceeds len(loader), how and why does the iterator restart?
6. Explain why validation must never update parameters, including buffers/BN considerations.

## Git commit on Windows CMD

```
git add training/loop.py tests/test_training_loop.py demos/demo_training_loop.py docs/week04_day02.md
git commit -m "feat: add MiniGPT training and validation loop"
git status
```

Do not claim real pretrained LLM/SFT or benchmark improvements from this toy experiment.
