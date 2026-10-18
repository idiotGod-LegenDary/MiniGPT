import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rope import RotaryEmbedding


KVPair = Tuple[torch.Tensor, torch.Tensor]


class GroupedQueryAttention(nn.Module):
    """
    输入:
        x: [B,L,D]

    返回:
        out: [B,L,D]
        present: (K,V) 或 None

    紧凑 KV Cache:
        K/V: [B,Hkv,P+L,Dh]
    """

    def __init__(
        self,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError("invalid d_model")

        if num_q_heads <= 0 or num_kv_heads <= 0:
            raise ValueError("head counts must be positive")

        if d_model % num_q_heads != 0:
            raise ValueError("d_model must divide by num_q_heads")

        if num_q_heads % num_kv_heads != 0:
            raise ValueError("Q Heads must divide by KV Heads")

        if not 0 <= dropout < 1:
            raise ValueError("invalid dropout")

        head_dim = d_model // num_q_heads

        if head_dim % 2 != 0:
            raise ValueError("RoPE requires even head_dim")

        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.group_size = num_q_heads // num_kv_heads

        # TODO A1
        # q_proj:
        #     D -> Hq * Dh
        #
        # k_proj / v_proj:
        #     D -> Hkv * Dh
        #
        # out_proj:
        #     D -> D
        #
        # 属性名称:
        # q_proj, k_proj, v_proj, out_proj
        self.q_proj = nn.Linear(d_model, num_q_heads * head_dim)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim)

        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RotaryEmbedding(head_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def split_heads(self, x, num_heads):
        """
        TODO A2

        输入:
            x: [B,L,num_heads*Dh]

        输出:
            [B,num_heads,L,Dh]

        注意:
            不允许始终使用 num_q_heads。
            K/V 需要使用 num_kv_heads。
        """
        B,L,D = x.shape
        return x.reshape(B,L,num_heads,D // num_heads).transpose(1,2)
        raise NotImplementedError

    def merge_heads(self, x):
        """
        TODO A3

        输入:
            [B,Hq,L,Dh]

        输出:
            [B,L,D]

        要求:
            正确处理 transpose 后的内存布局。
        """
        B,Hn,L,Dh = x.shape
        return x.transpose(1,2).contiguous().reshape(B,L,-1)
        raise NotImplementedError

    def repeat_kv(self, x):
        """
        TODO A4

        输入:
            x: [B,Hkv,T,Dh]

        输出:
            [B,Hq,T,Dh]

        方法:
            torch.repeat_interleave

        repeats:
            self.group_size

        dim:
            Head Dimension 所在轴

        约束:
            只用于 Attention 计算阶段。
            不用于扩大 KV Cache。
        """
        return torch.repeat_interleave(
            x,
            self.group_size,
            dim = 1
        )

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
                None 或 (past_k,past_v)

            past_k/past_v:
                [B,Hkv,P,Dh]

        输出:
            out: [B,L,D]

            present:
                [B,Hkv,P+L,Dh] 的 (K,V)
                或 None
        """

        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("invalid input shape")

        B, L, D = x.shape

        if B < 1 or L < 1:
            raise ValueError("invalid B or L")

        if past_key_value is not None and not use_cache:
            raise ValueError("past requires use_cache=True")

        # TODO A5：缓存验证
        #
        # None:
        #     past_len = 0
        #
        # 非 None:
        #     要求长度为 2 的 tuple
        #     检查两个元素均为 Tensor
        #     shape == [B,Hkv,P,Dh]
        #     检查 device 和 dtype
        #     past_len = P
        #
        # 错误时 raise ValueError

        if past_key_value is None:
            past_len = 0
        else:
            if not isinstance(past_key_value,(tuple, list)) or len(past_key_value)!= 2:
                raise ValueError("past_key_value must be a 2-tuple")

            past_k, past_v = past_key_value

            if past_k.shape != past_v.shape:
                raise ValueError("past_k and past_v shape mismatch")

            expected = (B, self.num_kv_heads, past_k.size(2), self.head_dim)

            if past_k.shape != expected:
                raise ValueError(
                    f"past cache shape {tuple(past_k.shape)} != {expected}"
                )

            if past_k.device != x.device or past_v.device != x.device:
                raise ValueError("past cache device mismatch")

            if past_k.dtype != x.dtype or past_v.dtype != x.dtype:
                raise ValueError("past cache dtype mismatch")

            past_len = past_k.size(2)

        # TODO A6：Q/K/V 投影
        #
        # q:
        # [B,Hq,L,Dh]
        #
        # k/v:
        # [B,Hkv,L,Dh]

        q = self.split_heads(self.q_proj(x),self.num_q_heads)
        k = self.split_heads(self.k_proj(x),self.num_kv_heads)
        v = self.split_heads(self.v_proj(x),self.num_kv_heads)

        # TODO A7：RoPE
        #
        # 只旋转本轮 q/k
        # position_offset=past_len
        #
        # 不旋转 v
        # 不重复旋转缓存的 K

        q_rot, k_rot = self.rope(q,k,position_offset=past_len)

        # TODO A8：更新紧凑 KV
        #
        # k_all/v_all:
        # [B,Hkv,P+L,Dh]
        #
        # 注意:
        # 此时不能调用 repeat_kv 后再缓存。

        if past_key_value is None:
            k_all = k_rot
            v_all = v
        else:
            k_all = torch.cat([past_k, k_rot],dim = 2)
            v_all = torch.cat([past_v, v],dim = 2)

        # TODO A9：仅在计算时展开 K/V
        #
        # k_expanded/v_expanded:
        # [B,Hq,P+L,Dh]

        k_expanded, v_expanded = self.repeat_kv(k_all),self.repeat_kv(v_all)

        # TODO A10：Scaled Attention Scores
        #
        # q @ k_expanded.transpose(-2,-1)
        # / sqrt(head_dim)
        #
        # [B,Hq,L,P+L]

        attn_score = q_rot @ k_expanded.transpose(-1,-2) / math.sqrt(self.head_dim)

        # TODO A11：矩形 Causal Mask
        #
        # Query position:
        # P ... P+L-1
        #
        # Key position:
        # 0 ... P+L-1
        #
        # mask:
        # [L,P+L]
        #
        # 位于 x.device

        pos_q = torch.arange(past_len,past_len + L, device=x.device)
        pos_k = torch.arange(0, past_len + L, device= x.device)

        causal_mask = (pos_q[:, None] >= pos_k[None, :]).bool()

        # TODO A12：Softmax、Dropout、Value 聚合
        #
        # masked scores -> softmax
        # -> dropout -> @ v_expanded
        #
        # context:
        # [B,Hq,L,Dh]
        #
        # merge_heads -> out_proj
        #
        # out:
        # [B,L,D]

        context = self.attn_dropout(F.softmax(attn_score.masked_fill(~causal_mask,float("-inf")), dim = -1)) @ v_expanded
        output = self.out_proj(self.merge_heads(context))

        # TODO A13：返回紧凑 Cache
        #
        # use_cache=True:
        #    present = (k_all,v_all)
        #
        # use_cache=False:
        #    present = None
        #
        # return out, present

        if use_cache:
            present = (k_all, v_all)
        else:
            present = None
        return output, present