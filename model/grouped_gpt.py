import torch
import torch.nn as nn

from model.rms_norm import ManualRMSNorm
from model.grouped_block import GroupedDecoderBlock


class GroupedMiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_q_heads: int,
        num_kv_heads: int,
        num_layers: int,
        hidden_dim: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        if vocab_size <= 0 or num_layers <= 0:
            raise ValueError("invalid config")

        if max_seq_len <= 0:
            raise ValueError("invalid max_seq_len")

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads

        self.token_embedding = nn.Embedding(
            vocab_size, d_model
        )

        self.blocks = nn.ModuleList([
            GroupedDecoderBlock(
                d_model=d_model,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
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
        input_ids,
        past_key_values=None,
        use_cache=False,
    ):
        """
        输入:
            input_ids: LongTensor [B,L]
            past_key_values: None 或 list[tuple[K,V]]
            use_cache: bool

        输出:
            logits: [B,L,Vocab]
            new_past: list 或 None
        """

        # TODO C1：验证输入
        #
        # input_ids ndim == 2
        # dtype == torch.long
        # B,L 都大于 0
        # ID 在 [0,vocab_size) 之间
        B,L = input_ids.shape

        if input_ids.ndim != 2:
            raise ValueError("input_ids.ndim must be 2")

        if input_ids.dtype != torch.long:
            raise ValueError("input_ids dtype mismatch")
        
        if B <= 0 or L <= 0:
            raise ValueError("input_ids shape mismatch")

        if input_ids.min() < 0 or input_ids.max() >= self.vocab_size:
            raise ValueError("token id out of range")


        # TODO C2：验证 Cache
        #
        # 无历史 Cache:
        #     past_list = [None] * num_layers
        #     P = 0
        #
        # 有历史 Cache:
        #     必须 use_cache=True
        #     list 长度为 num_layers
        #     每层 (K,V) 均为 Tensor
        #     Shape [B,Hkv,P,Dh]
        #     各层历史长度相同
        #     device、dtype 一致
        #
        # 必须先验证结构，再访问下标/shape。
        # 非法情况 raise ValueError。

        if past_key_values is None:
            past_list = [None] * self.num_layers
            past_len = 0
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

            num_heads = self.blocks[0].attn.num_kv_heads
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

        # TODO C3：检查 P + L <= max_seq_len
        if past_len + L > self.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")

        # TODO C4：Token Embedding
        x = self.token_embedding(input_ids)

        # TODO C5：逐层执行 Block
        #
        # 第 i 层只能使用第 i 层的 Cache
        # 按层收集新 Cache

        new_past = [] if use_cache else None
        for i in range(self.num_layers):
            x, present = self.blocks[i](x,past_list[i],use_cache)
            if use_cache:
                new_past.append(present)

        # TODO C6：Final RMSNorm
        norm_x = self.final_norm(x)

        # TODO C7：LM Head
        #
        # [B,L,D] → [B,L,Vocab]
        #
        # return logits, new_past
        logits = self.lm_head(x)
        return logits, new_past