import torch.nn as nn

from model.rms_norm import ManualRMSNorm
from model.swiglu import SwiGLU
from model.cached_attention import CachedCausalSelfAttention


class CachedDecoderBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = ManualRMSNorm(d_model)
        self.norm2 = ManualRMSNorm(d_model)

        self.attn = CachedCausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.ffn = SwiGLU(d_model, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        past_key_value=None,
        use_cache=False,
    ):
        """
        TODO B1

        实现 Pre-Norm Decoder Block。

        输入:
            x: [B,L,D]
            past_key_value: None 或 (K,V)
            use_cache: bool

        数据流:
            x
            ↓
            RMSNorm
            ↓
            Cached Attention
            ↓
            Residual
            ↓
            RMSNorm
            ↓
            SwiGLU
            ↓
            Residual

        返回:
            output: [B,L,D]
            present: (K,V) 或 None

        要求:
            past_key_value 和 use_cache
            正确传入 Attention。
        """
        attn_out, present = self.attn(self.norm1(x),past_key_value,use_cache)
        x = x + self.dropout(attn_out)

        ffn_out = self.ffn(self.norm2(x))
        output = x + self.dropout(ffn_out)
        return output, present