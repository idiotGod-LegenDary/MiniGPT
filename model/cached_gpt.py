from typing import Optional

import torch
import torch.nn as nn

from model.rms_norm import ManualRMSNorm
from model.cached_block import CachedDecoderBlock


class CachedMiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        if vocab_size <= 0 or num_layers <= 0:
            raise ValueError("invalid model config")

        if max_seq_len <= 0:
            raise ValueError("invalid max_seq_len")

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.blocks = nn.ModuleList([
            CachedDecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = ManualRMSNorm(d_model)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ):
        """
        输入:
            input_ids:
                LongTensor [B,L]

            past_key_values:
                None
                或长度为 num_layers 的 list

            use_cache:
                是否返回更新后的 KV Cache

        输出:
            logits: [B,L,V]
            new_past: list 或 None

        限制:
            P + L <= max_seq_len
            不使用额外 Absolute Position Embedding
            不支持 Padding、GQA、Sliding Window Cache
        """

        # TODO C1：检查 input_ids
        #
        # dtype 必须是 torch.long
        # ndim == 2
        # B >= 1
        # L >= 1
        #
        # Token ID 范围：
        # [0, vocab_size)
        #
        # 非法输入统一 raise ValueError                
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must be long")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be 2D")
        B, L = input_ids.shape
        if B < 1 or L < 1:
            raise ValueError("batch and length must be positive")
        if input_ids.min() < 0 or input_ids.max() >= self.vocab_size: # 不需要遍历
            raise ValueError("token id out of range")

        # TODO C2：检查 past_key_values
        #
        # 如果是 None:
        #     past_len = 0
        #     每个 Block 对应 None
        #
        # 否则:
        #     use_cache 必须为 True
        #     列表长度必须为 num_layers
        #     每层 Cache 都是 (K,V)
        #     所有层历史长度一致
        #     B/H/Dh、device、dtype 均兼容
        #
        # 错误时 raise ValueError

        if past_key_values is None:
            past_len = 0
            past_list = [None] * self.num_layers
        else:
            if not use_cache:
                raise ValueError("past_key_values requires use_cache=True")
            if not isinstance(past_key_values, (list, tuple)):
                raise ValueError("past_key_values must be a list")
            if len(past_key_values) != self.num_layers:
                raise ValueError("past_key_values length mismatch")

            past_list = list(past_key_values)
            first = past_list[0]

            if (
                not isinstance(first, tuple)
                or len(first) != 2
                or not all(isinstance(t, torch.Tensor) for t in first)
            ):
                raise ValueError("invalid first layer cache")

            past_len = first[0].size(2)

            num_heads = self.blocks[0].attn.num_heads
            head_dim = self.blocks[0].attn.head_dim

            for i, kv in enumerate(past_list):
                if kv is None or len(kv) != 2:
                    raise ValueError(f"layer {i} cache invalid")
                k, v = kv
                if k.shape != v.shape:
                    raise ValueError(f"layer {i} k/v shape mismatch")
                if k.shape != (B, num_heads, past_len, head_dim):
                    raise ValueError(f"layer {i} cache shape mismatch")
                if k.device != input_ids.device:
                    raise ValueError(f"layer {i} cache device mismatch")
                if k.dtype != self.token_embedding.weight.dtype:
                    raise ValueError(f"layer {i} cache dtype mismatch")
            
        # TODO C3：检查序列长度
        #
        # past_len + L <= max_seq_len
        #
        # 超限:
        # raise ValueError
        if past_len + L > self.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        
        # TODO C4：Token Embedding
        #
        # input_ids: [B,L]
        # x: [B,L,D]
        x = self.token_embedding(input_ids)

        # TODO C5：逐层计算
        #
        # blocks[i] 使用 past_key_values[i]
        #
        # 每个 Block 返回:
        # x, present
        #
        # use_cache=True:
        # 将各层 present 收集到 new_past
        #
        # use_cache=False:
        # new_past = None
        new_past = [] if use_cache else None
        for i in range(self.num_layers):
            x, present = self.blocks[i](x,past_list[i],use_cache)
            if use_cache:
                new_past.append(present)


        # TODO C6：Final RMSNorm
        x = self.final_norm(x)

        # TODO C7：LM Head
        #
        # x: [B,L,D]
        # logits: [B,L,V]
        #
        # return logits, new_past

        logits = self.lm_head(x)
        return logits, new_past
        raise NotImplementedError(
            "Implement CachedMiniGPT"
        )