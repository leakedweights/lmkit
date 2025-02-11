import jax

def cfg_gemma2():
    return {
        # RMS Norm
        "norm_eps": 1e-6,
        "norm_convert_w": True,
        "norm_w_bias": 1.0,
        "pre_ffn_norm": True,
        "post_ffn_norm": True,

        # Model config
        "weight_tying": True,
        "apply_sliding_window": True,
        "act_fn": jax.nn.gelu,
    }

def cfg_llama():
    return {
        # RMS Norm
        "norm_eps": 1e-6,
        "norm_convert_w": False,
        "norm_w_bias": 0.0,
        "pre_ffn_norm": False,
        "post_ffn_norm": False,

        # Model config
        "weight_tying": False,
        "apply_sliding_window": False,
        "act_fn": jax.nn.silu,
    }