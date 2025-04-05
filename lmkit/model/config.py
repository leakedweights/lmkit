from flax.core import FrozenDict


def extend_llama(base_config):
    return FrozenDict(
        {
            **base_config,
            "norm_eps": 1e-6,
            "norm_convert_w": False,
            "norm_w_bias": 0.0,
            "pre_ffn_norm": False,
            "post_ffn_norm": False,
        }
    )
