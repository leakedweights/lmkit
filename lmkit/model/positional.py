import jax.numpy as jnp

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