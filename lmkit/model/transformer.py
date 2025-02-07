import jax
from functools import partial
import jax.numpy as jnp

has_cuda = jax.default_backend == "gpu"

def encode(inputs, embed_table):
    embeddings = jnp.take(embed_table, inputs, axis=1)
    return embeddings

def decode(inputs, embed_table):
    pass

def rms_norm(x, scale):
    pass

def rope(x, rope_base):
    pass

@jax.jit
def ffn(x, params):
    pass

@partial(jax.jit, static_argnums=(3,4))
def attention(inputs, lengths, cache, params, config):
    attn_impl = "cudnn" if has_cuda else "xla"
    if config["apply_sliding_window"]:
        window_size = (config["sliding_window_size"], 0)

    query = jnp.dot(inputs, params["W_q"])
    query = rope(query, config["rope_base"])

    key = jnp.dot(inputs, params["W_k"])
    key = rope(key, config["rope_base"])
    
    value = jnp.dot(inputs, params["W_v"])

    x = jax.nn.dot_product_attention(
        query=query,
        key=key,
        value=value,
        is_causal=True,
        query_seq_lengths=lengths,
        key_value_seq_lengths=lengths,
        local_window_size=window_size or None,
        implementation=attn_impl,
    )

    x = ...
    
    x = jnp.dot(x, params["W_out"])
    
    return x

@partial(jax.vmap, in_axes=(0,0,0,None,None))
def run_decoder(inputs, lengths, cache, params, config):
    
    x = encode(inputs, params["embed_table"])

    for layer_id, (layer_params, layer_cache) in enumerate(zip(params["layers"], cache)):
        y = rms_norm(x, layer_params["input_norm"])
        y = attention(x, lengths, layer_cache, layer_params["attn"], config)

        if config.get("pre_ffn_norm"): # gemma 2 config
            x = rms_norm(x, layer_params["post_attn_norm"])
            y = x + y
            y = rms_norm(y, layer_params["pre_ffn_norm"])
        else: # gemma config
            y = x + y
            x = rms_norm(y, layer_params["post_attn_norm"])            
        
        y = ffn(y, layer_params["ffn"])
        
        if layer_params.get("post_ffn_norm"):
            y = rms_norm(y, layer_params["post_ffn_norm"])

        x = x + y

    if config["weight_tying"]:
        outputs = decode(x, params["embed_table"])
    else:
        outputs = decode(x, params["lm_head"])

    return outputs