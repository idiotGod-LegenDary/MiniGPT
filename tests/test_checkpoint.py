"""Day 03 的行为规范测试：只定义预期结果，不提供 TODO 实现。"""
import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from training.checkpoint import (
    FORMAT_VERSION, REQUIRED_KEYS, make_checkpoint, save_checkpoint,
    load_checkpoint, compare_model_parameters,
)


class TinyLM(nn.Module):
    def __init__(self, vocab_size=13, hidden=8, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, ids):
        return self.proj(self.drop(self.emb(ids)))


CFG = {"vocab_size": 13, "hidden": 8, "dropout": 0.3}
BATCH = (torch.tensor([[1, 2, 3], [3, 2, 1]]),
         torch.tensor([[2, 3, 4], [2, 1, 0]]))


def build_pair(seed=7):
    torch.manual_seed(seed)
    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    return model, optimizer


def step_once(model, optimizer):
    from training.loop import train_step
    return train_step(model, optimizer, BATCH)


def trained_pair():
    model, opt = build_pair()
    step_once(model, opt)
    step_once(model, opt)
    return model, opt


def test_snapshot_keys_and_metadata():
    m,o = trained_pair()
    snap = make_checkpoint(m, o, step=2, model_config=CFG)
    assert set(snap) == REQUIRED_KEYS
    assert snap['format_version'] == FORMAT_VERSION
    assert snap['step'] == 2
    assert snap['model_config'] == CFG
    assert snap['optimizer_class'] == 'AdamW'
    assert isinstance(snap['rng_state'], torch.Tensor)
    assert snap['rng_state'].dtype == torch.uint8


def test_snapshot_is_independent_of_future_training():
    m,o = trained_pair()
    snap = make_checkpoint(m, o, step=2, model_config=CFG)
    saved_weight = {k:v.clone() for k,v in snap['model_state'].items()}
    saved_moment = copy.deepcopy(snap['optimizer_state'])
    step_once(m, o)
    assert any(not torch.equal(m.state_dict()[k], v) for k,v in saved_weight.items())
    assert all(torch.equal(snap['model_state'][k], v) for k,v in saved_weight.items())
    for k in saved_moment['state']:
        for name,v in saved_moment['state'][k].items():
            if torch.is_tensor(v):
                assert torch.equal(snap['optimizer_state']['state'][k][name], v)


@pytest.mark.parametrize('step', [-1, 1.5, True])
def test_bad_step(step):
    m,o=build_pair()
    with pytest.raises(ValueError): make_checkpoint(m,o,step=step,model_config=CFG)


def test_bad_config():
    m,o=build_pair()
    with pytest.raises(TypeError): make_checkpoint(m,o,step=0,model_config={'shape':[1,2]})


def test_save_creates_directories_and_is_readable(tmp_path):
    m,o=trained_pair()
    checkpoint=make_checkpoint(m,o,step=2,model_config=CFG)
    path=tmp_path/'nested'/'deeper'/'ckpt.pt'
    returned=save_checkpoint(path,checkpoint)
    assert returned == path.resolve()
    assert path.is_file()
    raw=torch.load(path,map_location='cpu',weights_only=True)
    assert raw['step'] == 2
    assert not list(path.parent.glob('*.tmp'))


def test_save_overwrites_existing_file(tmp_path):
    m,o=trained_pair()
    path=tmp_path/'ckpt.pt'
    c1=make_checkpoint(m,o,step=2,model_config=CFG)
    save_checkpoint(path,c1)
    c2=make_checkpoint(m,o,step=3,model_config=CFG)
    save_checkpoint(str(path),c2)
    assert torch.load(path,weights_only=True)['step'] == 3


def test_model_and_optimizer_round_trip(tmp_path):
    original,opt=trained_pair()
    snapshot=make_checkpoint(original,opt,step=2,model_config=CFG)
    path=tmp_path/'roundtrip.pt'
    save_checkpoint(path,snapshot)
    restored,newopt=build_pair(seed=99)
    step=load_checkpoint(path,restored,newopt,expected_config=CFG)
    assert step == 2
    assert compare_model_parameters(original,restored)['allclose']
    a,b=opt.state_dict(),newopt.state_dict()
    assert len(a['param_groups']) == len(b['param_groups'])
    for k in a['state']:
        for name in a['state'][k]:
            av,bv=a['state'][k][name],b['state'][k][name]
            if torch.is_tensor(av): assert torch.equal(av,bv)
            else: assert av == bv


def test_reject_bad_config_without_modifying_model(tmp_path):
    m,o=trained_pair()
    p=tmp_path/'state.pt'; save_checkpoint(p,make_checkpoint(m,o,step=2,model_config=CFG))
    fresh,fo=build_pair(seed=88)
    before=copy.deepcopy(fresh.state_dict())
    with pytest.raises(ValueError):
        load_checkpoint(p,fresh,fo,expected_config={**CFG,'hidden':99})
    assert all(torch.equal(before[k],fresh.state_dict()[k]) for k in before)


def test_reject_bad_model_key_and_shape_before_modifying(tmp_path):
    m,o=trained_pair()
    c=make_checkpoint(m,o,step=2,model_config=CFG)
    fresh,fo=build_pair(seed=88)
    before=copy.deepcopy(fresh.state_dict())
    bad=copy.deepcopy(c)
    bad['model_state']['unexpected'] = torch.ones(1)
    p=tmp_path/'bad-key.pt'; torch.save(bad,p)
    with pytest.raises(ValueError): load_checkpoint(p,fresh,fo,expected_config=CFG)
    bad=copy.deepcopy(c)
    key=next(iter(bad['model_state']))
    bad['model_state'][key] = torch.zeros(1)
    p=tmp_path/'bad-shape.pt'; torch.save(bad,p)
    with pytest.raises(ValueError): load_checkpoint(p,fresh,fo,expected_config=CFG)
    assert all(torch.equal(before[k],fresh.state_dict()[k]) for k in before)


def test_reject_bad_format_and_missing_key(tmp_path):
    m,o=trained_pair()
    c=make_checkpoint(m,o,step=2,model_config=CFG)
    fresh,fo=build_pair()
    for transform in ['version', 'missing']:
        bad=copy.deepcopy(c)
        if transform=='version': bad['format_version'] = 999
        else: bad.pop('optimizer_class')
        p=tmp_path/f'{transform}.pt'; torch.save(bad,p)
        with pytest.raises(ValueError): load_checkpoint(p,fresh,fo,expected_config=CFG)


def test_reject_optimizer_group_mismatch(tmp_path):
    m,o=trained_pair()
    c=make_checkpoint(m,o,step=2,model_config=CFG)
    c['optimizer_state']['param_groups'][0]['params'].pop()
    p=tmp_path/'bad-optim.pt'; torch.save(c,p)
    fresh,fo=build_pair(seed=88)
    before=copy.deepcopy(fresh.state_dict())
    with pytest.raises(ValueError): load_checkpoint(p,fresh,fo,expected_config=CFG)
    assert all(torch.equal(before[k],fresh.state_dict()[k]) for k in before)


def test_rng_restore_and_exact_continuation(tmp_path):
    model,opt=trained_pair()
    save_checkpoint(tmp_path/'resume.pt',make_checkpoint(model,opt,step=2,model_config=CFG))
    # 恢复前后必须使用相同 Batch；Dropout 使用 PyTorch 的 CPU 随机数状态。
    for _ in range(3): step_once(model,opt)
    fresh,newopt=build_pair(seed=101)  # 消耗随机数并生成与原模型无关的初始权重
    step=load_checkpoint(tmp_path/'resume.pt',fresh,newopt,expected_config=CFG,restore_rng=True)
    assert step == 2
    for _ in range(3): step_once(fresh,newopt)
    assert compare_model_parameters(model,fresh)['allclose']
    assert compare_model_parameters(model,fresh)['max_abs_diff']==0.0


def test_restoring_rng_can_be_disabled(tmp_path):
    m,o=trained_pair()
    p=tmp_path/'c.pt'; save_checkpoint(p,make_checkpoint(m,o,step=2,model_config=CFG))
    fresh,fo=build_pair(seed=1)
    torch.manual_seed(765432)
    before=torch.get_rng_state().clone()
    load_checkpoint(p,fresh,fo,expected_config=CFG,restore_rng=False)
    assert torch.equal(torch.get_rng_state(),before)


def test_compare_parameters_reports_changes_and_is_symmetric():
    a,_=build_pair(seed=1); b=copy.deepcopy(a)
    assert compare_model_parameters(a,b)=={'allclose':True,'max_abs_diff':0.0,'different':[]}
    with torch.no_grad(): b.proj.bias[0].add_(0.25)
    result=compare_model_parameters(a,b)
    assert not result['allclose']; assert result['max_abs_diff'] == pytest.approx(0.25)
    assert result['different'] == ['proj.bias']
    assert compare_model_parameters(b,a)['different'] == result['different']


def test_compare_parameters_invalid_tolerance_or_shape():
    a,_=build_pair(seed=1)
    with pytest.raises(ValueError): compare_model_parameters(a,a,atol=-1)
    with pytest.raises(ValueError): compare_model_parameters(a,a,rtol=float('nan'))
    with pytest.raises(ValueError): compare_model_parameters(a,a,atol=True)
    b=TinyLM(hidden=10)
    with pytest.raises(ValueError): compare_model_parameters(a,b)


def test_minigpt_integration_when_available(tmp_path):
    try:
        from model.mini_gpt import MiniGPT
        from training.data import build_loader
    except ImportError:
        pytest.skip('Run this integration in the existing MiniGPT repo')
    cfg=dict(vocab_size=16,max_seq_len=8,d_model=32,d_ff=64,num_layers=1,dropout=0.0)
    model=MiniGPT(**cfg)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    x,y=next(iter(build_loader(list(range(16))*4,block_size=4,batch_size=2,shuffle=False)))
    # 原始数据为 0～15 的重复序列，全部 Token ID 都属于词表范围。
    from training.loop import train_step
    train_step(model,opt,(x,y))
    p=tmp_path/'mini.pt';save_checkpoint(p,make_checkpoint(model,opt,step=1,model_config=cfg))
    target=MiniGPT(**cfg); newopt=torch.optim.AdamW(target.parameters(),lr=1e-3)
    assert load_checkpoint(p,target,newopt,expected_config=cfg)==1
    assert compare_model_parameters(model,target)['allclose']
