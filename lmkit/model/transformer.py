from functools import partial

import jax
import jax.numpy as jnp
from einops import rearrange

has_cuda = jax.default_backend == "gpu"


def encode(inputs, embed_table):
    embeddings = jnp.take(embed_table, inputs, axis=0)
    return embeddings


def decode(inputs, lm_head):
    logits = inputs @ lm_head
    return logits


def rms_norm(x, weight, eps=1e-6):
    orig_dtype = x.dtype
    x = x.astype(jnp.float32)
    normed = x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)
    out = weight * normed.astype(orig_dtype)
    return out


@partial(jax.jit, static_argnums=(2,))
def ffn(x, params, act_fn):
    gate = x @ params["W_gate"]
    act = act_fn(gate)
    up = x @ params["W_up"]
    output = (act * up) @ params["W_down"]
    return output


def rope(x, cos, sin):
    x1, x2 = jnp.split(x, 2, axis=-1)
    x_rot = jnp.concatenate([-x2, x1], axis=-1)
    return ((x * cos) + (x_rot * sin)).astype(x.dtype)


def build_rope_cache(seq_len, head_dim, base):
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2) / head_dim))
    positions = jnp.arange(seq_len)
    angles = positions[:, None] * inv_freq[None, :]
    angles = jnp.concatenate([angles, angles], axis=-1)
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)
    return cos, sin


@partial(jax.jit, static_argnums=(4,))
def attention(inputs, lengths, params, rope_cache, config):
    attn_impl = "cudnn" if has_cuda else "xla"
    cos, sin = rope_cache

    query = inputs @ params["W_q"]
    query = rearrange(query, "t (n h) -> t n h", n=config["num_heads"])
    query = rope(query, cos[:, None, :], sin[:, None, :])

    key = inputs @ params["W_k"]
    key = rearrange(key, "t (n h) -> t n h", n=config["num_kv_heads"])
    key = rope(key, cos[:, None, :], sin[:, None, :])

    value = inputs @ params["W_v"]
    value = rearrange(value, "t (n h) -> t n h", n=config["num_kv_heads"])

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
    x = x @ params["W_o"]

    return x


@partial(jax.vmap, in_axes=(0, 0, None, None))
def run_decoder(inputs, lengths, params, config):
    x = encode(inputs, params["embed_table"])
    seq_len = x.shape[0]
    head_dim = config["hidden_size"] // config["num_heads"]

    rope_cache = build_rope_cache(seq_len, head_dim, base=config["rope_base"])

    for layer_params in params["layers"]:
        y = rms_norm(x, layer_params["input_norm"], eps=config["norm_eps"])
        attn_out = attention(y, lengths, layer_params["attn"], rope_cache, config)
        x = x + attn_out
        y = rms_norm(x, layer_params["post_attn_norm"], eps=config["norm_eps"])
        ffn_out = ffn(y, layer_params["ffn"], config["act_fn"])
        x = x + ffn_out

    x = rms_norm(x, params["out_norm"], eps=config["norm_eps"])
    logits = decode(x, params["lm_head"])
    return logits
