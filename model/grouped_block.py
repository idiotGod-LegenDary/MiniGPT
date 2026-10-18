import torch.nn as nn

from model.rms_norm import ManualRMSNorm
from model.swiglu import SwiGLU
from model.grouped_attention import GroupedQueryAttention


class GroupedDecoderBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = ManualRMSNorm(d_model)
        self.norm2 = ManualRMSNorm(d_model)

        self.attn = GroupedQueryAttention(
            d_model=d_model,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
        )

        self.ffn = SwiGLU(d_model, hidden_dim)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        past_key_value=None,
        use_cache=False,
    ):
        """
        TODO B1

        与上一日 CachedDecoderBlock 相同：

        Pre-RMSNorm
        → GroupedQueryAttention
        → Residual
        → Pre-RMSNorm
        → SwiGLU
        → Residual

        返回:
            out: [B,L,D]
            present: compact (K,V) 或 None
        """
        attn_out, present = self.attn(self.norm1(x), past_key_value,use_cache)
        x = x + self.resid_dropout(attn_out)

        ffn_out = self.ffn(self.norm2(x))
        output = x + self.resid_dropout(ffn_out)
        return output,present
        raise NotImplementedError("Implement GroupedDecoderBlock")