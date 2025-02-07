import os
import safetensors
import jax.numpy as jnp
from flax.core.frozen_dict import freeze
import huggingface_hub


def from_hf(repo, out_dir, api_key=None):
    if api_key is not None:
        huggingface_hub.login(token=api_key)

    call_params = {"repo_id": repo, "local_dir": out_dir}

    huggingface_hub.snapshot_download(
        **call_params,
        allow_patterns=["*.json", "*.safetensors", "*.model"]
    )


def assemble_for_jax(st_dir):
    params = {}

    st_files = sorted(
        [
            os.path.join(st_dir, f)
            for f in os.listdir(st_dir)
            if f.endswith(".safetensors")
        ]
    )

    for fpath in st_files:
        with safetensors.safe_open(fpath, framework="numpy") as st_file:
            for key in st_file.keys():
                tensor_np = st_file.get_tensor(key)
                tensor_jax = jnp.array(tensor_np)
                params[key] = tensor_jax

    frozen_params = freeze(params)

    return frozen_params


def gemma_to_lmkit(model_tensors):
    new_dict = {}

    for full_key, tensor in model_tensors.items():
        full_key = full_key.replace("model.norm", "model.out_norm")
        if full_key.startswith("model."):
            key = full_key[len("model.") :]
        else:
            key = full_key

        key = key.replace("self_attn", "attn")
        key = key.replace("attention", "attn")
        key = key.replace("mlp", "ffn")
        key = key.replace("feedforward", "ffn")
        key = key.replace("layernorm", "norm")
        key = key.replace("embed_tokens", "embed_table")

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
        if isinstance(layer_dict, dict):
            max_index = max(layer_dict.keys()) if layer_dict else -1
            layers_list = [None] * (max_index + 1)
            for idx, value in layer_dict.items():
                layers_list[idx] = value
            new_dict["layers"] = layers_list

    return new_dict