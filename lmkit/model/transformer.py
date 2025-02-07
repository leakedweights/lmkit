import jax
from functools import partial

def encode(inputs, embed_table):
    pass

def decode(inputs, embed_table):
    pass

@partial(jax.vmap, in_axes=(0,0,0,None,None))
def run_decoder(inputs, lengths, params, config):
    
    x = ...

    if config["weight_tying"]:
        outputs = decode(x, params["embed_table"])
    else:
        outputs = decode(x, params["lm_head"])

    return outputs