import json
import os

import huggingface_hub
import jax
import jax.numpy as jnp
import safetensors
import tokenizers
from einops import rearrange
from flax.core.frozen_dict import freeze
from tqdm import tqdm


def get_act_fn(act_str):
    return {"silu": jax.nn.silu, "gelu": jax.nn.gelu}[act_str]


def from_hf(repo, out_dir, token=None):
    if token is not None:
        huggingface_hub.login(token=token)

    call_params = {"repo_id": repo, "local_dir": out_dir}

    huggingface_hub.snapshot_download(
        **call_params, allow_patterns=["*.json", "*.safetensors", "*.model"]
    )


def gather_for_jax(st_dir):
    params = {}

    st_files = sorted(
        [
            os.path.join(st_dir, f)
            for f in os.listdir(st_dir)
            if f.endswith(".safetensors")
        ]
    )

    for fpath in tqdm(st_files, desc="Loading safetensors"):
        with safetensors.safe_open(fpath, framework="numpy") as st_file:
            for key in st_file.keys():
                tensor_np = st_file.get_tensor(key)
                tensor_jax = jnp.array(tensor_np)
                params[key] = tensor_jax

    frozen_params = freeze(params)

    return frozen_params


def params_to_lmkit(model_tensors):
    def rename_key(full_key):
        replaced = full_key.replace("model.norm", "model.out_norm")

        if replaced.startswith("model."):
            replaced = replaced[len("model.") :]

        replacement_rules = [
            ("self_attn", "attn"),
            ("attention", "attn"),
            ("mlp", "ffn"),
            ("feedforward", "ffn"),
            ("layernorm", "norm"),
            ("embed_tokens", "embed_table"),
        ]

        for old_str, new_str in replacement_rules:
            replaced = replaced.replace(old_str, new_str)

        return replaced

    new_dict = {}

    for full_key, tensor in model_tensors.items():
        if tensor.ndim >= 2 and "embed_tokens" not in full_key:
            tensor = rearrange(tensor, "... i j -> ... j i")

        key = rename_key(full_key)
        parts = key.split(".")

        current = new_dict
        for i, part in enumerate(parts[:-1]):
            if i == 1 and parts[0] == "layers" and part.isdigit():
                part = int(part)
            else:
                if isinstance(part, str) and part.endswith("_proj"):
                    part = "W_" + part[:-5]

            if part not in current:
                current[part] = {}
            current = current[part]

        last_part = parts[-1]
        if isinstance(last_part, str) and last_part.endswith("_proj"):
            last_part = "W_" + last_part[:-5]

        current[last_part] = tensor

    if "layers" in new_dict:
        layer_dict = new_dict["layers"]
        if isinstance(layer_dict, dict) and layer_dict:
            max_index = max(layer_dict.keys())
            layers_list = [None] * (max_index + 1)
            for idx, layer in layer_dict.items():
                layers_list[idx] = layer
            new_dict["layers"] = layers_list

    def collapse_leaf_dicts(d):
        if isinstance(d, dict):
            new_d = {k: collapse_leaf_dicts(v) for k, v in d.items()}
            if set(new_d.keys()) == {"weight"}:
                return new_d["weight"]
            return new_d
        elif isinstance(d, list):
            return [collapse_leaf_dicts(item) for item in d]
        else:
            return d

    new_dict = collapse_leaf_dicts(new_dict)
    return new_dict


def load_lmkit_config(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)

    keys_to_remove = ["architectures", "bos_token_id", "eos_token_id"]

    for key in keys_to_remove:
        del config[key]

    rename_map = {
        "attn_logit_softcapping": "attention_softcap",
        "final_logit_softcapping": "final_softcap",
        "rope_theta": "rope_base",
        "num_attention_heads": "num_heads",
        "num_key_value_heads": "num_kv_heads",
        "rms_norm_eps": "norm_eps",
        "torch_dtype": "precision",
        "hidden_act": "act_fn",
        "tie_word_embeddings": "io_tying",
        "num_hidden_layers": "num_layers",
    }

    new_config = {}
    for old_key, new_key in rename_map.items():
        if old_key in config:
            new_config[new_key] = config.pop(old_key)

    if "sliding_window" in config:
        window_val = config.pop("sliding_window")
        new_config["window_size"] = window_val
        if window_val is not None:
            new_config["apply_sliding_window"] = True

    for key, value in config.items():
        new_config[key] = value

    if "io_tying" not in new_config:
        new_config["io_tying"] = False

    new_config["act_fn"] = get_act_fn(new_config["act_fn"])

    return new_config


def load_lmkit_tokenizer(
    tokenizer_path, generation_config_path=None, pad_token="<|pad|>"
):
    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
    if generation_config_path is not None:
        with open(generation_config_path) as file:
            generation_config = json.load(file)

        bos = generation_config["bos_token_id"]
        eos = generation_config["bos_token_id"]
        tokenizer.bos_token_id = bos[0] if isinstance(bos, list) else bos
        tokenizer.eos_token_id = eos[0] if isinstance(eos, list) else eos

        tokenizer.add_special_tokens([pad_token])
        tokenizer.pad_token_id = tokenizer.token_to_id(pad_token)
        tokenizer.enable_padding(pad_id=tokenizer.pad_token_id, pad_token=pad_token)

    return tokenizer
