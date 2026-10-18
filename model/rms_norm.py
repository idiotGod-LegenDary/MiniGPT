import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class ManualRMSNorm(nn.Module):
    def __init__(self, 
        d_model,
        eps = 1e-4
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self,x):
        norm_X = x * torch.rsqrt(
            x.pow(2).mean(
                dim = -1,
                keepdim = True
            ) + self.eps
        )
        return self.weight * norm_X