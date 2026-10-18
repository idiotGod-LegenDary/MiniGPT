import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, 
        d_model,
        hidden_dim
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model,hidden_dim)
        self.up_proj = nn.Linear(d_model,hidden_dim)

        self.down_proj = nn.Linear(hidden_dim,d_model)
    
    def forward(self,x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))