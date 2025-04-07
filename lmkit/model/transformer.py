from functools import partial

import jax
import jax.numpy as jnp
from einops import rearrange

has_cuda = jax.default_backend() == "gpu"
print(f"Cuda processing allowed: {has_cuda}")


def rms_norm(x, weight, eps=1e-6):
    orig_dtype = x.dtype
    x = x.astype(jnp.float32)
    normed = x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)
    out = weight * normed.astype(orig_dtype)
    return out


def ffn(x, params, act_fn):
    gate = x @ params["W_gate"]
    act = act_fn(gate)
    up = x @ params["W_up"]
    output = (act * up) @ params["W_down"]
    return output


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def rope(x, sin, cos):
    if x.ndim == 4 and sin.ndim == 3:
        sin = sin[:, :, None, :]
        cos = cos[:, :, None, :]
    elif x.ndim > sin.ndim and x.shape[-1] == sin.shape[-1]:
        num_broadcast_dims = x.ndim - sin.ndim
        new_shape = list(sin.shape)
        for _ in range(num_broadcast_dims):
            new_shape.insert(-1, 1)
        sin = jnp.reshape(sin, new_shape)
        cos = jnp.reshape(cos, new_shape)
        if sin.shape[:-1] != x.shape[:-1] or cos.shape[:-1] != x.shape[:-1]:
            try:
                sin = sin[..., None, :]
                cos = cos[..., None, :]
            except IndexError:
                raise ValueError(
                    f"Cannot broadcast sin/cos shapes {sin.shape} to x shape {x.shape}"
                )

    rotated_x = (x * cos) + (rotate_half(x) * sin)
    return rotated_x.astype(x.dtype)


@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0))
def update_2d(arr1, arr2, update1, update2, start_idx):
    arr1_update = jax.lax.dynamic_update_slice_in_dim(arr1, update1, start_idx, axis=0)
    arr2_update = jax.lax.dynamic_update_slice_in_dim(arr2, update2, start_idx, axis=0)
    return arr1_update, arr2_update


def attention(inputs, cache, params, config):
    positions = cache.positions
    seq_lens = jnp.max(positions, axis=-1).astype(jnp.int32) + 1

    sin, cos = cache.sin, cache.cos

    query = inputs @ params["W_q"]
    key = inputs @ params["W_k"]
    value = inputs @ params["W_v"]

    query = rearrange(query, "... t (n h) -> ... t n h", n=config["num_heads"])
    query = rope(query, sin, cos)
    key = rearrange(key, "... t (n h) -> ... t n h", n=config["num_kv_heads"])
    key = rope(key, sin, cos)
    value = rearrange(value, "... t (n h) -> ... t n h", n=config["num_kv_heads"])

    full_key, full_value = key, value
    if cache.keys is not None:
        full_key, full_value = update_2d(
            cache.keys, cache.values, key, value, cache.cached_lens
        )

    x = jax.nn.dot_product_attention(
        query=query,
        key=full_key,
        value=full_value,
        is_causal=cache.keys is None,
        query_seq_lengths=seq_lens,
        key_value_seq_lengths=seq_lens,
        implementation="cudnn" if has_cuda else "xla",
    )

    x = rearrange(x, "... t n h -> ... t (n h)")
    x = x @ params["W_o"]

    return x, cache.replace(keys=full_key, values=full_value)


@partial(jax.jit, static_argnums=(3,))
def run(inputs, cache, params, config):
    x = jnp.take(params["embed_table"], inputs, axis=0, fill_value=-1e6)

    new_layer_cache = []

    for i, layer_params in enumerate(params["layers"]):
        y = rms_norm(x, layer_params["input_norm"], eps=config["norm_eps"])
        attn_out, layer_cache = attention(
            y, cache.layers[i], layer_params["attn"], config
        )
        new_layer_cache.append(layer_cache)

        x = x + attn_out
        y = rms_norm(x, layer_params["post_attn_norm"], eps=config["norm_eps"])
        ffn_out = ffn(y, layer_params["ffn"], config["act_fn"])
        x = x + ffn_out

    x = rms_norm(x, params["out_norm"], eps=config["norm_eps"])
    logits = x @ params["lm_head"]

    if cache.use_kv:
        cache = cache.replace(layers=new_layer_cache)
    return logits, cache


def create(key, config, dtype=jnp.bfloat16, stddev=0.006):
    def normal_init(key, shape, dim_in, stddev, dtype=jnp.float32):
        if stddev is None:
            stddev = 1.0 / jnp.sqrt(dim_in)
        return stddev * jax.random.normal(key, shape, dtype=dtype)

    def ones_init(shape, dtype=jnp.float32):
        return jnp.ones(shape, dtype=dtype)

    hidden_size = config["hidden_size"]
    vocab_size = config["vocab_size"]
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    num_kv_heads = config.get("num_kv_heads", num_heads)
    intermediate_size = config["intermediate_size"]
    head_dim = config.get("head_dim", hidden_size // num_heads)

    keys = jax.random.split(
        key, num_layers + 3
    )  # Keys for layers + embed + out_norm + lm_head
    params = {}

    # Embedding table
    params["embed_table"] = normal_init(
        keys[0],
        (vocab_size, hidden_size),
        hidden_size,
        stddev=stddev,
        dtype=dtype,
    )

    # Transformer Layers
    params["layers"] = []
    layer_keys = jax.random.split(keys[1], num_layers)
    for i in range(num_layers):
        layer_key = layer_keys[i]
        attn_key, ffn_key = jax.random.split(layer_key, 2)
        attn_q_key, attn_k_key, attn_v_key, attn_o_key = jax.random.split(attn_key, 4)
        ffn_g_key, ffn_u_key, ffn_d_key = jax.random.split(ffn_key, 3)

        layer_params = {
            "input_norm": ones_init(
                (hidden_size,), dtype=dtype
            ),  # RMSNorm weight init to 1
            "attn": {
                "W_q": normal_init(
                    attn_q_key,
                    (hidden_size, num_heads * head_dim),
                    hidden_size,
                    dtype=dtype,
                ),
                "W_k": normal_init(
                    attn_k_key,
                    (hidden_size, num_kv_heads * head_dim),
                    hidden_size,
                    dtype=dtype,
                ),
                "W_v": normal_init(
                    attn_v_key,
                    (hidden_size, num_kv_heads * head_dim),
                    hidden_size,
                    dtype=dtype,
                ),
                "W_o": normal_init(
                    attn_o_key,
                    (num_heads * head_dim, hidden_size),
                    num_heads * head_dim,
                    dtype=dtype,
                ),
            },
            "post_attn_norm": ones_init((hidden_size,), dtype=dtype),
            "ffn": {
                "W_gate": normal_init(
                    ffn_g_key,
                    (hidden_size, intermediate_size),
                    hidden_size,
                    dtype=dtype,
                ),
                "W_up": normal_init(
                    ffn_u_key,
                    (hidden_size, intermediate_size),
                    hidden_size,
                    dtype=dtype,
                ),
                "W_down": normal_init(
                    ffn_d_key,
                    (intermediate_size, hidden_size),
                    intermediate_size,
                    dtype=dtype,
                ),
            },
        }
        params["layers"].append(layer_params)

    params["out_norm"] = ones_init((hidden_size,), dtype=dtype)

    params["lm_head"] = normal_init(
        keys[-1], (hidden_size, vocab_size), hidden_size, dtype=dtype
    )

    return params
