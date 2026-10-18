import torch

from model.grouped_gpt import GroupedMiniGPT


def cache_bytes(past):
    """
    TODO D1

    输入:
        past: list[(K,V)]

    输出:
        int：所有层 K/V 的理论字节总和

    推荐:
        Tensor.numel()
        Tensor.element_size()
    """
    total = 0
    for k, v in past:
        total += k.numel() * k.element_size()
        total += v.numel() * v.element_size()
    return total

def run_one_mode(name, num_kv_heads):
    torch.manual_seed(42)

    model = GroupedMiniGPT(
        vocab_size=64,
        d_model=64,
        num_q_heads=8,
        num_kv_heads=num_kv_heads,
        num_layers=2,
        hidden_dim=192,
        max_seq_len=64,
        dropout=0.0,
    ).eval()

    prompt = torch.tensor(
        [[1, 5, 8, 12, 3, 9]],
        dtype=torch.long,
    )

    with torch.inference_mode():

        # TODO D2
        # Prefill
        # logits_prefill, past = model(prompt,use_cache=True)

        logits_prefill, past = model(prompt,use_cache = True)

        # TODO D3
        # 从 logits_prefill[:, -1:, :]
        # 计算下一个 token
        # next_token: [B,1]

        next_token = logits_prefill[:, -1:, :].argmax(dim = -1)

        # TODO D4
        # 用 prompt + next_token 进行完整重算
        # full_last_logits: [B,1,Vocab]

        logits_full, _ = model(torch.cat([prompt, next_token],dim = -1))
        full_last_logits = logits_full[:, -1:, :]

        # TODO D5
        # 仅输入 next_token，使用历史 past
        # cached_logits: [B,1,Vocab]

        cached_logits, updated_past = model(next_token,use_cache = True,past_key_values = past)

        # TODO D6
        # assert torch.allclose(...)
        # 计算 max_abs_diff

        assert torch.allclose(
            cached_logits,
            full_last_logits,
            atol=1e-5,
            rtol=1e-5
        )

        # TODO D7
        # 输出:
        # name
        # compact K/V shape
        # cache_bytes(past)
        # max_abs_diff

        max_diff = (
            cached_logits - full_last_logits
        ).abs().max().item()
        
        print(f"=== {name} (num_kv_heads={num_kv_heads}) ===")
        print("  compact K shape:", tuple(past[0][0].shape))
        print("  compact V shape:", tuple(past[0][1].shape))
        print("  cache bytes:", cache_bytes(past))
        print("  max abs diff:", max_diff)
        print("  KV Cache consistency: passed")


def main():
    for name, heads in [
        ("MHA", 8),
        ("GQA", 2),
        ("MQA", 1),
    ]:
        run_one_mode(name, heads)


if __name__ == "__main__":
    main()