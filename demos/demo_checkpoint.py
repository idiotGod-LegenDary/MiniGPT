"""完成 Day 03 的 TODO 后执行：python -m demos.demo_checkpoint。"""
from pathlib import Path
import tempfile

import torch

from model.mini_gpt import MiniGPT
from training.data import build_loader
from training.loop import train_step
from training.checkpoint import (
    make_checkpoint, save_checkpoint, load_checkpoint, compare_model_parameters,
)


def main():
    cfg=dict(vocab_size=16,max_seq_len=8,d_model=32,d_ff=64,num_layers=1,dropout=0.0)
    torch.manual_seed(123)
    m=MiniGPT(**cfg)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=0.0)
    loader=build_loader(list(range(16))*4,block_size=4,batch_size=2,shuffle=False)
    it=iter(loader)
    for _ in range(3): train_step(m,opt,next(it))
    with tempfile.TemporaryDirectory() as tmp:
        saved=save_checkpoint(Path(tmp)/'models'/'step_3.pt',make_checkpoint(m,opt,step=3,model_config=cfg))
        restored=MiniGPT(**cfg)
        newopt=torch.optim.AdamW(restored.parameters(),lr=1e-3,weight_decay=0.0)
        step=load_checkpoint(saved,restored,newopt,expected_config=cfg)
        report=compare_model_parameters(m,restored)
        print('restored_step:',step)
        print('parameter_comparison:',report)
        print('optimizer_state_parameters:',len(newopt.state_dict()['state']))
        assert report['allclose']


if __name__=='__main__': main()
