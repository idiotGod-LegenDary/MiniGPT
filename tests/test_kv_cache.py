import torch
import pytest

from model.cached_attention import CachedCausalSelfAttention
from model.cached_gpt import CachedMiniGPT


def build_model():
    torch.manual_seed(42)

    model = CachedMiniGPT(
        vocab_size=50,
        d_model=32,
        num_heads=4,
        num_layers=3,
        hidden_dim=96,
        max_seq_len=64,
        dropout=0.0,
    )

    return model.eval()


def test_prefill_cache_shapes():
    model = build_model()

    ids = torch.randint(0, 50, (2, 7))

    with torch.inference_mode():
        logits, caches = model(
            ids,
            use_cache=True,
        )

    assert logits.shape == (2, 7, 50)
    assert len(caches) == 3

    for k, v in caches:
        assert k.shape == (2, 4, 7, 8)
        assert v.shape == (2, 4, 7, 8)


def test_single_token_decode():
    model = build_model()

    prompt = torch.randint(0, 50, (2, 6))
    new_token = torch.randint(0, 50, (2, 1))

    with torch.inference_mode():
        _, past = model(
            prompt,
            use_cache=True,
        )

        logits_cached, updated = model(
            new_token,
            past_key_values=past,
            use_cache=True,
        )

        full = torch.cat(
            [prompt, new_token],
            dim=1,
        )

        logits_full, _ = model(
            full,
            use_cache=False,
        )

    assert logits_cached.shape == (2, 1, 50)

    assert torch.allclose(
        logits_cached,
        logits_full[:, -1:, :],
        atol=1e-5,
        rtol=1e-5,
    )

    for k, v in updated:
        assert k.shape[2] == 7
        assert v.shape[2] == 7


def test_chunked_prefill():
    model = build_model()

    ids = torch.randint(0, 50, (2, 12))

    with torch.inference_mode():
        full_logits, _ = model(ids)

        logits_a, past = model(
            ids[:, :5],
            use_cache=True,
        )

        logits_b, updated = model(
            ids[:, 5:],
            past_key_values=past,
            use_cache=True,
        )

    joined = torch.cat(
        [logits_a, logits_b],
        dim=1,
    )

    assert joined.shape == full_logits.shape

    assert torch.allclose(
        joined,
        full_logits,
        atol=1e-5,
        rtol=1e-5,
    )

    assert updated[0][0].shape[2] == 12


def test_multiple_decode_steps():
    model = build_model()

    ids = torch.randint(0, 50, (1, 5))

    with torch.inference_mode():
        _, past = model(
            ids,
            use_cache=True,
        )

        full = ids

        for _ in range(5):
            next_ids = torch.randint(
                0, 50, (1, 1)
            )

            full = torch.cat(
                [full, next_ids],
                dim=1,
            )

            logits_cached, past = model(
                next_ids,
                past_key_values=past,
                use_cache=True,
            )

            logits_full, _ = model(full)

            assert torch.allclose(
                logits_cached,
                logits_full[:, -1:, :],
                atol=1e-5,
                rtol=1e-5,
            )


def test_max_length_check():
    model = build_model()

    ids = torch.randint(0, 50, (1, 64))
    next_id = torch.randint(0, 50, (1, 1))

    with torch.inference_mode():
        _, past = model(
            ids,
            use_cache=True,
        )

        with pytest.raises(ValueError):
            model(
                next_id,
                past_key_values=past,
                use_cache=True,
            )


def test_invalid_token_id():
    model = build_model()

    ids = torch.tensor([[1, 2, 50]])

    with pytest.raises(ValueError):
        model(ids)


def test_rectangular_attention_mask():
    torch.manual_seed(42)

    attn = CachedCausalSelfAttention(
        d_model=32,
        num_heads=4,
        dropout=0.0,
    ).eval()

    x = torch.randn(1, 8, 32)

    with torch.inference_mode():
        full_out, _ = attn(x)

        _, past = attn(
            x[:, :5, :],
            use_cache=True,
        )

        chunk_out, updated = attn(
            x[:, 5:, :],
            past_key_value=past,
            use_cache=True,
        )

    assert chunk_out.shape == (1, 3, 32)
    assert updated[0].shape == (1, 4, 8, 8)

    assert torch.allclose(
        chunk_out,
        full_out[:, 5:, :],
        atol=1e-5,
        rtol=1e-5,
    )


if __name__ == "__main__":
    raise SystemExit(
        pytest.main([__file__, "-q"])
    )