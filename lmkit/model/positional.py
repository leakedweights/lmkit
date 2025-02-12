import jax.numpy as jnp


def rope(embeds, base, offset=0):
    seq_len = embeds.shape[0]
    d_key = embeds.shape[-1]
    half_dim = d_key // 2

    positions = jnp.arange(offset, offset + seq_len)[:, None, None]
    dim_idx = jnp.arange(half_dim)[None, None, :]
    theta = base ** (-2 * dim_idx / d_key)
    angles = positions * theta
    
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)

    comp1, comp2 = jnp.split(embeds, 2, axis=-1)
    rotated_first = comp1 * cos - comp2 * sin
    rotated_second = comp1 * sin + comp2 * cos

    return jnp.concatenate([rotated_first, rotated_second], axis=-1).astype(
        embeds.dtype
    )
