import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rope import RotaryEmbedding


KVPair = Tuple[torch.Tensor, torch.Tensor]


class CachedCausalSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        if d_model <= 0 or num_heads <= 0:
            raise ValueError("dimensions must be positive")

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        if (d_model // num_heads) % 2 != 0:
            raise ValueError("RoPE head_dim must be even")

        if not 0 <= dropout < 1:
            raise ValueError("invalid dropout")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RotaryEmbedding(self.head_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        TODO A1

        输入:
            [B,L,D]

        输出:
            [B,H,L,Dh]

        使用 reshape + transpose。
        """
        B,L,D = x.shape
        return x.reshape(B,L,self.num_heads,self.head_dim).transpose(1,2)
        

    def merge_heads(self, x):
        """
        TODO A2

        输入:
            [B,H,L,Dh]

        输出:
            [B,L,D]

        注意 transpose 可能产生非连续 Tensor。
        """
        B,_,L,_ = x.shape
        return x.transpose(1,2).contiguous().reshape(B,L,-1)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[KVPair] = None,
        use_cache: bool = False,
    ):
        """
        输入:
            x: [B,L,D]

            past_key_value:
                None
                或 (past_k, past_v)

            past_k/past_v:
                [B,H,P,Dh]

        输出:
            output:
                [B,L,D]

            present:
                (K_all,V_all) 或 None

        约束:
            - 不允许空序列
            - 历史 Cache 必须和当前模型配置兼容
            - Cache 中的 K 已经过 RoPE
            - Cache 中的 V 未经过 RoPE
        """

        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError("invalid input shape")

        B, L, D = x.shape

        if B < 1 or L < 1:
            raise ValueError("batch and length must be positive")

        if past_key_value is not None and not use_cache:
            raise ValueError(
                "past cache requires use_cache=True"
            )

        # TODO A3：校验历史 Cache
        #
        # past_key_value is None:
        #     past_len = 0
        #
        # 否则:
        #     必须是长度为 2 的 tuple
        #     past_k.shape == [B,H,P,Dh]
        #     past_v.shape == [B,H,P,Dh]
        #     两者 device、dtype 与当前输入投影兼容
        #     past_len = past_k.size(2)
        #
        # 非法输入统一 raise ValueError
        if past_key_value is None:
            past_len = 0
            past_k = past_v = None
        else:
            if not isinstance(past_key_value, (tuple, list)) or len(past_key_value) != 2:
                raise ValueError("past_key_value must be a 2-tuple")

            past_k, past_v = past_key_value

            if past_k.shape != past_v.shape:
                raise ValueError("past_k and past_v shape mismatch")

            expected = (B, self.num_heads, past_k.size(2), self.head_dim)
            if past_k.shape != expected:
                raise ValueError(
                    f"past cache shape {tuple(past_k.shape)} != {expected}"
                )

            if past_k.device != x.device or past_v.device != x.device:
                raise ValueError("past cache device mismatch")

            if past_k.dtype != x.dtype or past_v.dtype != x.dtype:
                raise ValueError("past cache dtype mismatch")

            past_len = past_k.size(2)
        # TODO A4：生成本轮 Q/K/V
        #
        # x: [B,L,D]
        #
        # q_new: [B,H,L,Dh]
        # k_new: [B,H,L,Dh]
        # v_new: [B,H,L,Dh]

        q_new = self.split_heads(self.q_proj(x))
        k_new = self.split_heads(self.k_proj(x))
        v_new = self.split_heads(self.v_proj(x))

        # TODO A5：RoPE
        #
        # 对 q_new、k_new 使用 RoPE
        #
        # position_offset = past_len
        #
        # 不旋转 v_new
        # 不重新旋转 past_k

        q_new,k_new = self.rope(q_new, k_new, past_len)

        # TODO A6：拼接 Cache
        #
        # 如果没有历史 Cache:
        #     k_all = k_new
        #     v_all = v_new
        #
        # 否则:
        #     k_all = cat([past_k,k_new], dim=2)
        #     v_all = cat([past_v,v_new], dim=2)
        #
        # 输出 Shape:
        # [B,H,P+L,Dh]
        if past_len == 0:
            k_all = k_new
            v_all = v_new
        else:
            k_all = torch.cat([past_k, k_new], dim = 2)
            v_all = torch.cat([past_v, v_new], dim = 2)

        # TODO A7：Attention Scores
        #
        # q_new @ k_all.transpose(-2,-1)
        #
        # 除以 sqrt(head_dim)
        #
        # Shape:
        # [B,H,L,P+L]
        attn_score = q_new @ k_all.transpose(-2,-1) / math.sqrt(self.head_dim)

        # TODO A8：矩形 Causal Mask
        #
        # Query 绝对位置:
        # [P, P+1, ..., P+L-1]
        #
        # Key 绝对位置:
        # [0, 1, ..., P+L-1]
        #
        # mask[i,j] = True:
        #     key_position[j] <= query_position[i]
        #
        # Shape:
        # [L,P+L]
        #
        # 必须创建在 x.device

        pos_q = torch.arange(past_len, past_len + L, device=x.device)
        pos_k = torch.arange(0, past_len + L, device=x.device)

        # 广播：pos_q[:, None] [L,1]  vs pos_k[None,:] [1, P+L]
        causal_mask = (pos_q[:, None] >= pos_k[None, :]).bool()

        # TODO A9：Attention 输出
        #
        # Masked Scores
        # -> Softmax(dim=-1)
        # -> Attention Dropout
        # -> @ v_all
        # -> merge_heads
        # -> out_proj
        #
        # output.shape == [B,L,D]

        masked_scores = attn_score.masked_fill(~causal_mask,float("-inf"))
        output = self.out_proj(self.merge_heads(self.attn_dropout(F.softmax(masked_scores, dim = -1)) @ v_all))

        # TODO A10：返回缓存
        #
        # if use_cache:
        #     present = (k_all,v_all)/
        # else:
        #     present = None
        #
        # return output, present
        if use_cache:
            present = (k_all, v_all)
        else:
            present = None
        return output,present