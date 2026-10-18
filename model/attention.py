import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class CausalSelfAttention(nn.Module):
    '''
    [B, L, D]
    ↓
    Q/K/V
    ↓
    Attention Scores
    ↓
    Causal Mask
    ↓
    Softmax
    ↓
    Weighted Sum
    ↓
    [B, L, D]
    '''
    def __init__(
            self,
            d_model
        ):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model,d_model)


    def forward(self,x):
        L = x.size(-2)        
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        d_k = q.size(-1)
        attn_score = q @ k.transpose(-1,-2) / math.sqrt(d_k)
        causal_mask = torch.tril(torch.ones(L, L, device=x.device)).bool().unsqueeze(0)
        weights = F.softmax(attn_score.masked_fill(~causal_mask, float("-inf")), dim=-1)
        return self.out_proj(weights @ v), weights