from functools import partial

import jax
import jax.numpy as jnp
from einops import rearrange

has_cuda = jax.default_backend == "gpu"


def encode(inputs, embed_table):
    embeddings = jnp.take(embed_table, inputs, axis=0)
    return embeddings


def decode(inputs, lm_head):
    logits = jnp.einsum("ij,kj->ik", inputs, lm_head)
    return logits


def rms_norm(x, weight, eps=1e-6):
    orig_dtype = x.dtype
    x = x.astype(jnp.float32)
    normed = x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)
    out = normed * weight
    return out.astype(orig_dtype)


def rope(embeds, base):
    seq_len, d_key = embeds.shape

    half_dim = d_key // 2
    positions = jnp.arange(seq_len)[:, None]
    dim_idx = jnp.arange(half_dim)[None, :]

    theta = base ** (-2 * dim_idx / d_key)

    angles = positions * theta
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)

    comp1, comp2 = jnp.split(embeds, 2, axis=-1)

    rotated_first = comp1 * cos - comp2 * sin
    rotated_second = comp1 * sin + comp2 * cos

    outputs = jnp.concatenate([rotated_first, rotated_second], axis=-1)

    return jnp.astype(outputs, embeds.dtype)


@partial(jax.jit, static_argnums=(2,))
def ffn(x, params, act_fn):
    gate = jnp.einsum("ij,kj->ik", x, params["W_gate"])
    act = act_fn(gate)
    up = jnp.einsum("ij,kj->ik", x, params["W_up"])
    output = jnp.einsum("ij,kj->ik", act * up, params["W_down"])
    return output


@partial(jax.jit, static_argnums=(3,))
def attention(inputs, lengths, params, config):
    attn_impl = "cudnn" if has_cuda else "xla"

    query = jnp.einsum("ij,kj->ik", inputs, params["W_q"])
    query = rope(query, config["rope_base"])
    query = rearrange(query, "t (n h) -> t n h", n=config["num_heads"])

    key = jnp.einsum("ij,kj->ik", inputs, params["W_k"])
    key = rope(key, config["rope_base"])
    key = rearrange(key, "t (n h) -> t n h", n=config["num_kv_heads"])

    value = jnp.einsum("ij,kj->ik", inputs, params["W_v"])
    value = rearrange(value, "t (n h) -> t n h", n=config["num_kv_heads"])

    # raise Exception(f"Query shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}")

    x = jax.nn.dot_product_attention(
        query=query,
        key=key,
        value=value,
        is_causal=True,
        query_seq_lengths=lengths,
        key_value_seq_lengths=lengths,
        implementation=attn_impl,
    )

    x = rearrange(x, "t n h -> t (n h)")
    x = jnp.einsum("ij,kj->ik", x, params["W_o"])

    return x


@partial(jax.vmap, in_axes=(0, 0, None, None))
def run_decoder(inputs, lengths, params, config):
    x = jnp.take(params["embed_table"], inputs, axis=0)

    for layer_id, layer_params in enumerate(params["layers"]):
        # 1) Pre-attn norm
        y = rms_norm(x, layer_params["input_norm"], eps=config["norm_eps"])

        # 2) Self-attn
        attn_out = attention(y, lengths, layer_params["attn"], config)
        # 3) Residual
        x = x + attn_out

        # 4) Post-attn norm
        y = rms_norm(x, layer_params["post_attn_norm"], eps=config["norm_eps"])

        # 5) FFN
        ffn_out = ffn(y, layer_params["ffn"], config["act_fn"])
        x = x + ffn_out

    x = rms_norm(x, params["out_norm"], eps=config["norm_eps"])
    logits = decode(x, params["lm_head"])
    return logits
