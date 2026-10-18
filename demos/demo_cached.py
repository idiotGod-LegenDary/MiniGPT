import torch

from model.cached_gpt import CachedMiniGPT


def main():

    torch.manual_seed(42)

    model = CachedMiniGPT(
        vocab_size=32,
        d_model=64,
        num_heads=4,
        num_layers=2,
        hidden_dim=192,
        max_seq_len=64,
        dropout=0.0,
    )

    model.eval()

    prompt = torch.tensor(
        [[1, 4, 2, 8, 3]],
        dtype=torch.long,
    )

    with torch.inference_mode():

        # TODO D1：Prefill
        #
        # logits_prefill, past = model(...)
        #
        # use_cache=True
        #
        # logits_prefill:
        # [1,5,32]
        #
        # 每层 Cache 长度:
        # 5
        logits_prefill, past = model(prompt,use_cache = True)

        # TODO D2：预测第一个新 Token
        #
        # 从 logits_prefill 最后一个位置:
        # argmax(dim=-1)
        #
        # new_token:
        # [1,1]
        new_token = logits_prefill[:, -1, :].argmax(
            dim=-1,
            keepdim=True
        )   

        # TODO D3：构造完整序列
        #
        # full_tokens =
        # cat([prompt,new_token],dim=1)
        #
        # shape:
        # [1,6]
        full_tokens = torch.cat([prompt,new_token],dim =-1)

        # TODO D4：完整前缀重算
        #
        # logits_full, _ = model(
        #     full_tokens,
        #     use_cache=False
        # )
        #
        # 只取最后一个位置
        #
        # [1,1,32]
        logits_full, _ = model(
            full_tokens,
            use_cache = False
        )
        
        logits_full_last = logits_full[:, -1:, :]

        # TODO D5：增量 Decode
        #
        # logits_cached, updated_past = model(
        #     new_token,
        #     past_key_values=past,
        #     use_cache=True
        # )
        #
        # logits_cached:
        # [1,1,32]
        #
        # Cache 长度:
        # 6
        logits_cached, updated_past = model(
            new_token,
            past_key_values=past,
            use_cache=True
        )

        # TODO D6：验证数值一致性
        #
        # torch.allclose(
        #     logits_cached,
        #     logits_full,
        #     atol=1e-5,
        #     rtol=1e-5
        # )
        #
        # 打印：
        # max absolute difference


        assert torch.allclose(
            logits_cached,
            logits_full_last,
            atol=1e-5,
            rtol=1e-5
        )

        max_diff = (
            logits_cached - logits_full_last
        ).abs().max().item()

        print("max absolute difference:", max_diff)
        print("KV Cache consistency: passed")
        pass


if __name__ == "__main__":
    main()