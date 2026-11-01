"""Day02 executable interface specification; tests do not supply TODO solutions."""
import copy
import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from training.loop import compute_lm_loss, train_step, evaluate, train_for_steps


class ToyLM(nn.Module):
    """Small deterministic teaching model, NOT part of student's MiniGPT code."""
    def __init__(self, vocab=9):
        super().__init__()
        self.emb = nn.Embedding(vocab, 12)
        self.dropout = nn.Dropout(p=0.4)
        self.head = nn.Linear(12, vocab)

    def forward(self, x):
        return self.head(self.dropout(self.emb(x)))


def batch():
    x = torch.tensor([[0,1,2],[2,3,4]], dtype=torch.long)
    y = torch.tensor([[1,2,3],[3,4,5]], dtype=torch.long)
    return x,y


def loader_for_batches(batch_size=2):
    x,y = batch()
    return DataLoader(TensorDataset(x,y), batch_size=batch_size, shuffle=False)


def test_ce_matches_pytorch_reference():
    torch.manual_seed(123)
    logits = torch.randn(2,3,8, requires_grad=True)
    y = torch.tensor([[1,2,3],[4,5,6]])
    actual = compute_lm_loss(logits,y)
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1,8),y.reshape(-1))
    assert actual.shape == () and actual.requires_grad
    assert torch.allclose(actual,expected,atol=1e-7)


def test_ce_ignore_index_and_gradient():
    logits = torch.randn(2,2,6,requires_grad=True)
    y = torch.tensor([[1,-100],[3,2]])
    actual = compute_lm_loss(logits,y)
    actual.backward()
    assert torch.all(logits.grad[0,1] == 0)
    assert logits.grad[0,0].abs().sum() > 0


@pytest.mark.parametrize('logits,labels,error', [
    (torch.randn(2,4), torch.ones(2, dtype=torch.long), ValueError),
    (torch.randn(2,3,5), torch.ones(2,4, dtype=torch.long), ValueError),
    (torch.randn(2,3,5), torch.ones(2,3, dtype=torch.float32), TypeError),
    (torch.randn(2,3,5), torch.full((2,3),-100,dtype=torch.long), ValueError),
])
def test_ce_rejects_bad_input(logits,labels,error):
    with pytest.raises(error):
        compute_lm_loss(logits,labels)


def test_train_step_changes_weights_and_returns_float():
    torch.manual_seed(3)
    model = ToyLM()
    opt = torch.optim.SGD(model.parameters(),lr=0.1)
    before = model.head.weight.detach().clone()
    model.eval()
    out = train_step(model,opt,batch())
    assert type(out) is float and math.isfinite(out)
    assert model.training
    assert not torch.equal(before,model.head.weight)


def test_train_step_one_update_matches_manual():
    torch.manual_seed(2)
    left = ToyLM()
    right = copy.deepcopy(left)
    left.eval(); right.eval()
    x,y = batch()
    opt_left = torch.optim.SGD(left.parameters(),lr=0.01)
    opt_right = torch.optim.SGD(right.parameters(),lr=0.01)
    torch.manual_seed(99)
    actual = train_step(left,opt_left,(x,y))
    torch.manual_seed(99)
    right.train(); opt_right.zero_grad(set_to_none=True)
    expected = torch.nn.functional.cross_entropy(right(x).reshape(-1,9),y.reshape(-1))
    expected.backward(); opt_right.step()
    assert actual == pytest.approx(float(expected.item()),abs=1e-6)
    for p,q in zip(left.parameters(),right.parameters()):
        assert torch.allclose(p,q,atol=1e-6)


def test_train_step_clears_old_grad_and_can_clip():
    model = ToyLM()
    opt = torch.optim.SGD(model.parameters(),lr=0.01)
    for p in model.parameters():
        p.grad = torch.full_like(p, 1e6)
    value = train_step(model,opt,batch(),max_grad_norm=0.2)
    assert math.isfinite(value)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
    assert grad_norm.item() <= 0.2 + 1e-5


def test_train_step_bad_clip_value():
    model=ToyLM()
    opt=torch.optim.SGD(model.parameters(),lr=0.1)
    for invalid in (0,-2,float('nan')):
        with pytest.raises(ValueError):
            train_step(model,opt,batch(),max_grad_norm=invalid)


def test_evaluate_restores_model_mode_and_grads():
    model=ToyLM()
    opt=torch.optim.SGD(model.parameters(),lr=0.1)
    train_step(model,opt,batch())
    stored=[None if p.grad is None else p.grad.clone() for p in model.parameters()]
    for was_training in (True,False):
        model.train(was_training)
        before=[p.detach().clone() for p in model.parameters()]
        result=evaluate(model,loader_for_batches(batch_size=1))
        assert math.isfinite(result)
        assert model.training == was_training
        assert all(torch.equal(p,q) for p,q in zip(model.parameters(),before))
        assert all(torch.equal(p.grad, old) for p,old in zip(model.parameters(),stored))


def test_evaluate_token_weighted_with_uneven_batches():
    torch.manual_seed(5)
    model=ToyLM()
    model.eval()
    x=torch.tensor([[0,1,2],[3,4,5],[1,2,3]],dtype=torch.long)
    y=torch.tensor([[1,2,3],[4,5,6],[2,3,-100]],dtype=torch.long)
    loader=DataLoader(TensorDataset(x,y),batch_size=2,shuffle=False)
    with torch.no_grad():
        logits=model(x)
        expected=torch.nn.functional.cross_entropy(logits.reshape(-1,9),y.reshape(-1),ignore_index=-100).item()
    result=evaluate(model,loader)
    assert result == pytest.approx(expected,rel=1e-6,abs=1e-6)


def test_evaluate_empty_loader_and_mode_restore():
    model=ToyLM()
    model.train()
    with pytest.raises(ValueError):
        evaluate(model,[])
    assert model.training


@pytest.mark.parametrize('steps,eval_every,has_val', [
    (0,None,False),(-1,None,False),
    (3,0,True),(3,None,True),(3,1,False),
])
def test_for_steps_rejects_invalid_args(steps,eval_every,has_val):
    model=ToyLM(); opt=torch.optim.SGD(model.parameters(),lr=0.01)
    val=loader_for_batches() if has_val else None
    with pytest.raises(ValueError):
        train_for_steps(model,opt,loader_for_batches(),steps,val_loader=val,eval_every=eval_every)


def test_for_steps_restarts_loader_and_returns_plain_numbers():
    model=ToyLM(); opt=torch.optim.SGD(model.parameters(),lr=0.01)
    history=train_for_steps(model,opt,loader_for_batches(batch_size=2),num_steps=4)
    assert len(history) == 4
    assert [r['step'] for r in history] == [1,2,3,4]
    assert all(set(r) == {'step','train_loss'} for r in history)
    assert all(type(r['train_loss']) is float and math.isfinite(r['train_loss']) for r in history)


def test_for_steps_periodic_validation():
    model=ToyLM(); opt=torch.optim.SGD(model.parameters(),lr=0.01)
    history=train_for_steps(model,opt,loader_for_batches(),5,
                            val_loader=loader_for_batches(batch_size=1),eval_every=2)
    assert len(history)==5
    assert ['val_loss' in r for r in history] == [False,True,False,True,False]
    assert all(type(r['val_loss']) is float for r in history if 'val_loss' in r)
    assert model.training  # evaluate() restores training mode


def test_for_steps_empty_loader():
    model=ToyLM(); opt=torch.optim.SGD(model.parameters(),lr=0.01)
    with pytest.raises(ValueError):
        train_for_steps(model,opt,[],2)


def test_integration_existing_minigpt_if_present():
    """In the real repository, this MUST run; standalone template may skip."""
    model_module=pytest.importorskip('model.mini_gpt')
    MiniGPT=model_module.MiniGPT
    from training.data import build_loader
    model=MiniGPT(vocab_size=16,max_seq_len=6,d_model=16,d_ff=32,num_layers=1,dropout=0.0)
    opt=torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=0.0)
    ld=build_loader([i%16 for i in range(50)],block_size=5,batch_size=2,shuffle=False)
    history=train_for_steps(model,opt,ld,2,val_loader=ld,eval_every=2)
    assert len(history)==2 and math.isfinite(history[1]['val_loss'])
