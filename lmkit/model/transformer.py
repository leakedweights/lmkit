import jax
import jax.numpy as jnp
from einops import rearrange
from functools import partial

has_cuda = jax.default_backend == "gpu"

def encode(inputs, embed_table):
    embeddings = jnp.take(embed_table, inputs, axis=1)
    return embeddings

def decode(inputs, embed_table):
    dec = jnp.einsum('12,32->13', inputs, embed_table)
    logits = dec * jnp.sqrt(embed_table.shape[-1])
    return logits

def rms_norm(x, w_bias, weight, eps, convert_w=False):
    dtype = x.dtype
    x = jnp.astype(x, jnp.float32)

    scale = jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)
    x = x * scale

    if convert_w:
        weight = jnp.astype(weight, jnp.float32)
    else:
        scale = jnp.astype(scale, x.dtype)
    weight = weight + w_bias

    x = x * weight
    return jnp.astype(x, dtype)

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
    
    rotated_first  = comp1 * cos - comp2 * sin
    rotated_second = comp1 * sin + comp2 * cos
    
    return jnp.concatenate([rotated_first, rotated_second], axis=-1)

@partial(jax.jit, static_argnums=(2,))
def ffn(x, params, act_fn):
    gate = jnp.dot(x, params["W_gate"])
    act = act_fn(gate)
    up = jnp.dot(x, params["W_up"])
    output = jnp.dot(act * up, params["W_down"])

    return output

@partial(jax.jit, static_argnums=(3,4))
def attention(inputs, lengths, cache, params, config):
    attn_impl = "cudnn" if has_cuda else "xla"
    window_size = None
    if config.get("apply_sliding_window", False):
        window_size = (config["sliding_window_size"], 0)

    query = jnp.dot(inputs, params["W_q"])
    query = rearrange(query, "t (n h) -> t n h", n=config["num_heads"])
    query = rope(query, config["rope_base"])

    key = jnp.dot(inputs, params["W_k"])
    key = rearrange(key, "t (n h) -> t n h", n=config["num_kv_heads"])
    key = rope(key, config["rope_base"])
    
    new_value = jnp.dot(inputs, params["W_v"])

    if cache is not None:
        cached_key = cache.get("k", None)
        cached_value = cache.get("v", None)
        if cached_key is None:
            full_key = key
            full_value = new_value
        else:
            full_key = jnp.concatenate([cached_key, key], axis=0) # concat along token dim
            full_value = jnp.concatenate([cached_value, new_value], axis=0)
        cache["k"] = full_key
        cache["v"] = full_value
    else:
        full_key = key
        full_value = new_value

    x = jax.nn.dot_product_attention(
        query=query,
        key=full_key,
        value=full_value,
        is_causal=True,
        query_seq_lengths=lengths,
        key_value_seq_lengths=lengths,
        local_window_size=window_size,
        implementation=attn_impl,
    )

    x = rearrange(x, "t n h -> t (n h)")
    x = jnp.dot(x, params["W_out"])
        
    return x, cache

@partial(jax.vmap, in_axes=(0,0,0,None,None))
def run_decoder(inputs, lengths, cache, params, config):
    
    x = encode(inputs, params["embed_table"])

    for layer_id, (layer_params, layer_cache) in enumerate(zip(params["layers"], cache)):
        y = rms_norm(x, layer_params["input_norm"])
        y, cache[layer_id] = attention(x, lengths, layer_cache, layer_params["attn"], config)

        if config.get("pre_ffn_norm"): # gemma 2
            x = rms_norm(x, layer_params["post_attn_norm"])
            y = x + y
            y = rms_norm(y, layer_params["pre_ffn_norm"])
        else:
            y = x + y
            x = rms_norm(y, layer_params["post_attn_norm"])            
        
        y = ffn(y, layer_params["ffn"])
        
        if layer_params.get("post_ffn_norm"):
            y = rms_norm(y, layer_params["post_ffn_norm"])

        x = x + y

    x = rms_norm(x, params["out_norm"])

    if config["weight_tying"]:
        outputs = decode(x, params["embed_table"])
    else:
        outputs = decode(x, params["lm_head"])

    return outputs