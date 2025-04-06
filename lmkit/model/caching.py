from typing import List, Optional

import jax.numpy as jnp
from flax import struct

def build_rope(positions, head_dim, base):
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    inv_freq = 1.0 / (
        base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    positions = positions.astype(jnp.float32)
    freqs = positions[:, :, None] * inv_freq[None, None, :]
    emb = jnp.concatenate((freqs, freqs), axis=-1)

    pad_mask = positions >= 0
    pad_mask = pad_mask[:, :, None].repeat(head_dim, axis=-1)
    emb = jnp.where(pad_mask, emb, 0.0)

    sin_values = jnp.sin(emb)
    cos_values = jnp.cos(emb)

    return sin_values, cos_values

@struct.dataclass
class LayerCache:
    sin: jnp.array
    cos: jnp.array
    cached_lens: jnp.array
    positions: jnp.array
    keys: Optional[jnp.array] = None
    values: Optional[jnp.array] = None


@struct.dataclass
class TransformerCache:
    use_kv: bool = struct.field(pytree_node=False)
    layers: List[LayerCache]
    full_positions: jnp.array
    full_sin: jnp.array
    full_cos: jnp.array

    @classmethod
    def initialize(
        cls, batch_size, current_positions, config, max_total_length=0, use_kv=False
    ):
        head_dim = config["hidden_size"] // config["num_heads"]
        positions = jnp.arange(max_total_length).astype(jnp.int32)
        positions = jnp.broadcast_to(positions, (batch_size, max_total_length))
        sin, cos = build_rope(positions, head_dim, config["rope_base"])

        layers = [
            LayerCache(
                sin=sin,
                cos=cos,
                cached_lens=jnp.zeros((batch_size,)).astype(jnp.int32),
                positions=current_positions,
                keys=None,
                values=None,
            )
            for _ in range(config["num_layers"])
        ]
        return cls(
            layers=layers,
            use_kv=use_kv,
            full_sin=sin,
            full_cos=cos,
            full_positions=positions,
        )

    def roll(self):
        batch_indices = jnp.arange(self.full_positions.shape[0]).astype(jnp.int32)
        first_layer = self.layers[0]
        seq_lens = jnp.max(first_layer.positions, axis=-1).astype(jnp.int32) + 1

        full_positions = self.full_positions.at[batch_indices, seq_lens].set(seq_lens)

        if self.use_kv:
            cached_lens = seq_lens
            new_positions = full_positions[batch_indices, seq_lens][..., None]
            new_sin = self.full_sin[batch_indices, seq_lens][:, None, :]
            new_cos = self.full_cos[batch_indices, seq_lens][:, None, :]
        else:
            cached_lens = first_layer.cached_lens
            new_positions = full_positions
            new_sin = self.full_sin
            new_cos = self.full_cos

        new_layers = []
        for layer in self.layers:
            new_layers.append(
                layer.replace(
                    positions=new_positions,
                    cached_lens=cached_lens,
                    sin=new_sin,
                    cos=new_cos,
                )
            )

        return self.replace(layers=new_layers, full_positions=full_positions)
