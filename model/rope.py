import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, 
        head_dim,
        base = 10000
    ) -> None:
        super().__init__()
        self.head_dim = head_dim

        i = torch.arange(0,head_dim,2)
        self.register_buffer("inv_freq", 1.0 / (base ** (i / head_dim)), persistent=True)

    def forward(self,
        q,
        k,
        position_offset = 0
    ):
        B,H,L,_ = q.shape
        m = torch.arange(position_offset, position_offset + L,device = q.device, dtype = torch.float32)
        freqs = torch.outer(m,self.inv_freq)
        cosv = torch.cos(freqs).to(q.dtype).unsqueeze(0).unsqueeze(0)
        sinv = torch.sin(freqs).to(q.dtype).unsqueeze(0).unsqueeze(0)

        q1,q2 = q[...,0::2], q[..., 1::2]
        q_rot = torch.stack([cosv * q1 - sinv * q2, sinv * q1 + cosv * q2],dim = -1).flatten(-2)

        k1,k2 = k[...,0::2], k[..., 1::2]
        k_rot = torch.stack([cosv * k1 - sinv * k2, sinv * k1 + cosv * k2],dim = -1).flatten(-2)

        return q_rot,k_rot