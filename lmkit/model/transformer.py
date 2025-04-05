from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import struct

has_cuda = jax.default_backend() == "gpu"
print(f"Cuda processing allowed: {has_cuda}")


@struct.dataclass
class LayerCache:
    sin: jnp.array
    cos: jnp.array
    positions: jnp.array
    keys: Optional[jnp.array]
    values: Optional[jnp.array]


@struct.dataclass
class TransformerCache:
    use_kv: bool = struct.field(pytree_node=False)

    sin: jnp.array = struct.field(pytree_node=False)
    cos: jnp.array = struct.field(pytree_node=False)
    positions: jnp.array = struct.field(pytree_node=True)

    keys: Optional[jnp.array] = struct.field(pytree_node=True)
    values: Optional[jnp.array] = struct.field(pytree_node=True)

    def __getitem__(self, idx):
        keys = self.keys[idx] if self.keys is not None else None
        values = self.values[idx] if self.values is not None else None
        return LayerCache(
            sin=self.sin,
            cos=self.cos,
            positions=self.positions,
            keys=keys,
            values=values,
        )

    def update_layers(self, layer_caches):
        if not self.use_kv:
            return
        new_keys = jnp.array([layer_cache.keys for layer_cache in layer_caches])
        new_values = jnp.array([layer_cache.values for layer_cache in layer_caches])
        return self.replace(keys=new_keys, values=new_values)

    def next(self):
        def update_single_sequence(positions):
            positions = jnp.concatenate([positions, jnp.array([-1])], axis=-1)
            seq_len = jnp.sum(positions >= 0).astype(jnp.int32)
            positions = positions.at[seq_len].set(seq_len)
            return positions.astype(jnp.int32)

        new_positions = jax.vmap(update_single_sequence)(self.positions)
        return self.replace(positions=new_positions)


def encode(inputs, embed_table):
    embeddings = jnp.take(embed_table, inputs, axis=0, fill_value=0)
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


@partial(jax.jit, static_argnums=(1, 2))
def build_rope_cache(positions, head_dim, base):
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2) / head_dim))[None, :]
    angles = positions[:, None] * inv_freq
    angles = jnp.concatenate([angles, angles], axis=-1)
    sin = jnp.sin(angles)
    cos = jnp.cos(angles)
    return sin, cos


@partial(jax.jit, static_argnums=(3,))
def attention(inputs, params, cache, config):
    attn_impl = "cudnn" if has_cuda else "xla"

    valid_positions = cache.positions >= 0
    effective_seq_len = jnp.sum(valid_positions, axis=-1).astype(jnp.int32)

    input_length = inputs.shape[0]
    start_idx = effective_seq_len - input_length
    start_idx = jnp.maximum(0, start_idx)

    query = inputs @ params["W_q"]
    query = rearrange(query, "t (n h) -> t n h", n=config["num_heads"])

    cos = jax.lax.dynamic_slice_in_dim(cache.cos, start_idx, inputs.shape[0])[
        :, None, :
    ]
    sin = jax.lax.dynamic_slice_in_dim(cache.sin, start_idx, inputs.shape[0])[
        :, None, :
    ]

    query = rope(query, cos, sin)

    key = inputs @ params["W_k"]
    key = rearrange(key, "t (n h) -> t n h", n=config["num_kv_heads"])
    key = rope(key, cos, sin)

    value = inputs @ params["W_v"]
    value = rearrange(value, "t (n h) -> t n h", n=config["num_kv_heads"])

    if cache.keys is not None:
        full_key = jnp.concatenate(
            [cache.keys, -1e6 * jnp.ones(key.shape, dtype=key.dtype)], axis=0
        )
        full_value = jnp.concatenate(
            [cache.values, -1e6 * jnp.ones(value.shape, dtype=value.dtype)], axis=0
        )
        full_key = jax.lax.dynamic_update_slice_in_dim(full_key, key, start_idx, axis=0)
        full_value = jax.lax.dynamic_update_slice_in_dim(
            full_value, value, start_idx, axis=0
        )
    else:
        full_key = key
        full_value = value

    x = jax.nn.dot_product_attention(
        query=query,
        key=full_key,
        value=full_value,
        is_causal=True,
        query_seq_lengths=[query.shape[0]],
        key_value_seq_lengths=[effective_seq_len],
        implementation=attn_impl,
    )

    x = rearrange(x, "t n h -> t (n h)")
    x = x @ params["W_o"]

    return x, cache.replace(keys=full_key, values=full_value)


@partial(jax.vmap, in_axes=(0, 0, None, None))
def run_decoder(inputs, cache, params, config):
    x = encode(inputs, params["embed_table"])

    layer_caches = []

    for i, layer_params in enumerate(params["layers"]):
        y = rms_norm(x, layer_params["input_norm"], eps=config["norm_eps"])
        attn_out, new_layer_cache = attention(y, layer_params["attn"], cache[i], config)
        layer_caches.append(new_layer_cache)
        x = x + attn_out
        y = rms_norm(x, layer_params["post_attn_norm"], eps=config["norm_eps"])
        ffn_out = ffn(y, layer_params["ffn"], config["act_fn"])
        x = x + ffn_out

    cache = cache.update_layers(layer_caches)

    x = rms_norm(x, params["out_norm"], eps=config["norm_eps"])
    logits = decode(x, params["lm_head"])

    return logits, cache
