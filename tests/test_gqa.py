import pytest
import torch

from model.grouped_attention import GroupedQueryAttention
from model.grouped_gpt import GroupedMiniGPT


def make_model(kv_heads):
    torch.manual_seed(42)

    return GroupedMiniGPT(
        vocab_size=64,
        d_model=64,
        num_q_heads=8,
        num_kv_heads=kv_heads,
        num_layers=2,
        hidden_dim=192,
        max_seq_len=64,
        dropout=0.0,
    ).eval()


@pytest.mark.parametrize("kv_heads", [8, 2, 1])
def test_cache_shapes(kv_heads):
    model = make_model(kv_heads)

    ids = torch.randint(0, 64, (2, 7))

    with torch.inference_mode():
        logits, past = model(ids, use_cache=True)

    assert logits.shape == (2, 7, 64)
    assert len(past) == 2

    for k, v in past:
        assert k.shape == (2, kv_heads, 7, 8)
        assert v.shape == (2, kv_heads, 7, 8)


@pytest.mark.parametrize("kv_heads", [8, 2, 1])
def test_single_step_equivalence(kv_heads):
    model = make_model(kv_heads)

    prompt = torch.randint(0, 64, (2, 6))
    next_token = torch.randint(0, 64, (2, 1))

    with torch.inference_mode():
        _, past = model(prompt, use_cache=True)

        cached_logits, updated = model(
            next_token,
            past_key_values=past,
            use_cache=True,
        )

        full = torch.cat([prompt, next_token], dim=1)
        full_logits, _ = model(full)

    assert torch.allclose(
        cached_logits,
        full_logits[:, -1:, :],
        atol=1e-5,
        rtol=1e-5,
    )

    assert updated[0][0].shape[2] == 7


@pytest.mark.parametrize("kv_heads", [8, 2, 1])
def test_chunked_prefill(kv_heads):
    model = make_model(kv_heads)

    ids = torch.randint(0, 64, (2, 13))

    with torch.inference_mode():
        full_logits, _ = model(ids)

        logits_a, past = model(
            ids[:, :5],
            use_cache=True,
        )

        logits_b, _ = model(
            ids[:, 5:],
            past_key_values=past,
            use_cache=True,
        )

    joined = torch.cat([logits_a, logits_b], dim=1)

    assert torch.allclose(
        joined,
        full_logits,
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("kv_heads", [8, 2, 1])
def test_multi_step_decode(kv_heads):
    model = make_model(kv_heads)

    prompt = torch.randint(0, 64, (1, 5))

    with torch.inference_mode():
        _, past = model(prompt, use_cache=True)
        full = prompt

        for _ in range(4):
            next_token = torch.randint(0, 64, (1, 1))
            full = torch.cat([full, next_token], dim=1)

            cached_logits, past = model(
                next_token,
                past_key_values=past,
                use_cache=True,
            )

            full_logits, _ = model(full)

            assert torch.allclose(
                cached_logits,
                full_logits[:, -1:, :],
                atol=1e-5,
                rtol=1e-5,
            )


def test_memory_reduction():
    prompt = torch.randint(0, 64, (1, 10))

    sizes = {}

    for hkv in [8, 2, 1]:
        model = make_model(hkv)

        with torch.inference_mode():
            _, cache = model(prompt, use_cache=True)

        total = sum(
            k.numel() * k.element_size()
            + v.numel() * v.element_size()
            for k, v in cache
        )

        sizes[hkv] = total

    assert sizes[8] == sizes[2] * 4
    assert sizes[8] == sizes[1] * 8


def test_invalid_head_counts():
    with pytest.raises(ValueError):
        GroupedQueryAttention(
            d_model=64,
            num_q_heads=8,
            num_kv_heads=3,
        )


def test_mha_matches_legacy_attention():
    """
    本测试验证：
        Hq == Hkv 时，
        GroupedQueryAttention 与上一日 Attention 等价。

    属性名称兼容：
        q_proj, k_proj, v_proj, out_proj, rope

    不依赖带 Cache 的模型层。
    """
    from model.grouped_attention import GroupedQueryAttention

    torch.manual_seed(42)

    gqa = GroupedQueryAttention(
        d_model=32,
        num_q_heads=4,
        num_kv_heads=4,
        dropout=0.0,
    ).eval()

    x = torch.randn(2, 7, 32)

    with torch.inference_mode():
        result, present = gqa(x, use_cache=True)

    assert result.shape == x.shape
    assert present[0].shape == (2, 4, 7, 8)

    # TODO T1：附加等价对照
    #
    # 若复制了上一日的 CachedCausalSelfAttention，
    # 可以增加：
    #
    # 1. 使用相同配置构建 legacy attention
    # 2. 复制同名投影参数
    # 3. 对同一个 x 做前向
    # 4. 比较输出 allclose
    #
    # 此处暂不预先断言不存在的 legacy 文件。

    '''
    没复制
    '''