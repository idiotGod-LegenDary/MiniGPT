import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    '''
    D
    ↓
    d_ff
    ↓
    activation
    ↓
    D
    '''
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff,d_model)
        )


    def forward(self,x):
        return self.net(x)