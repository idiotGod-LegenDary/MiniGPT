import torch
import torch.nn as nn
import torch.nn.functional as F

from model.feed_forward import FeedForward
from model.attention import CausalSelfAttention

class DecoderBlock(nn.Module):
    '''
    Causal Self-Attention

    LayerNorm

    Residual

    FeedForward
    '''
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()

        self.attn = CausalSelfAttention(d_model)
        self.LN1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model,d_ff,dropout=dropout)
        self.LN2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,x):
        attn_out, attn_weights = self.attn(self.LN1(x))
        x = x + self.dropout1(attn_out)

        ffn_out = self.dropout2(self.ffn(self.LN2(x)))
        x = x + ffn_out
        return x,attn_weights