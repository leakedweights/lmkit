import json
import logging
import os
import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import optax
import wandb
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, Lowercase, Sequence, StripAccents
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tqdm.auto import tqdm

from ..model import caching, transformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def save_checkpoint(params, opt_state, step, checkpoint_dir, wandb_run_id=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pkl")
    checkpoint = {
        "params": jax.device_get(params),
        "opt_state": jax.device_get(opt_state),
        "step": step,
        "wandb_run_id": wandb_run_id,
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)
    logging.info(f"Checkpoint saved at step {step} to {checkpoint_path}")

    if wandb.run is not None:
        try:
            artifact = wandb.Artifact(
                f"checkpoint-{wandb.run.id}-{step}", type="model-checkpoint"
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact, aliases=[f"step_{step}", "latest"])
            logging.info(f"Logged checkpoint artifact to W&B for step {step}")
        except Exception as e:
            logging.error(f"Failed to log checkpoint artifact to W&B: {e}")


def load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        logging.warning(f"Checkpoint file not found: {checkpoint_path}")
        return None, None, 0, None

    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    logging.info(
        f"Loaded checkpoint from step {checkpoint.get('step', 0)}: {checkpoint_path}"
    )
    return (
        checkpoint["params"],
        checkpoint.get("opt_state"),
        checkpoint.get("step", 0),
        checkpoint.get("wandb_run_id"),
    )


def compute_metrics(logits, targets, mask):
    vocab_size = logits.shape[-1]

    valid_target_mask = (targets >= 0) & (targets < vocab_size)
    final_mask = mask & valid_target_mask
    final_mask = final_mask.astype(jnp.bool_)

    num_valid_tokens = jnp.maximum(jnp.sum(final_mask), 1)
    safe_targets = jnp.where(final_mask, targets, 0)

    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, safe_targets
    )

    masked_loss = per_token_loss * final_mask
    mean_loss = jnp.sum(masked_loss) / num_valid_tokens

    predictions = jnp.argmax(logits, axis=-1)
    correct_predictions = (predictions == targets) * final_mask
    accuracy = jnp.sum(correct_predictions) / num_valid_tokens

    metrics = {
        "loss": mean_loss,
        "accuracy": accuracy,
        "num_valid_tokens": num_valid_tokens,
    }
    return metrics


def log_weights_biases_norms(params, step, use_wandb=True):
    param_norms = jax.tree_util.tree_map(
        lambda x: jnp.linalg.norm(x.astype(jnp.float32)).item(), params
    )

    global_param_norm = optax.global_norm(params).item()
    logging.info(f"[Step {step}] Global Param Norm: {global_param_norm:.4f}")

    wandb_norms = {"norms/global_param_norm": global_param_norm}

    if "embed_table" in params:
        norm_val = param_norms["embed_table"]
        logging.info(f"[Step {step}] Norm(embed_table): {norm_val:.4f}")
        wandb_norms["norms/embed_table"] = norm_val

    if "layers" in params and len(params["layers"]) > 0:
        flat_params_with_path, _ = jax.tree_util.tree_flatten_with_path(
            params["layers"]
        )
        for key_path, leaf_norm in zip(
            flat_params_with_path, jax.tree_util.tree_leaves(param_norms["layers"])
        ):
            wandb_key = "norms/layers/" + "/".join(map(str, key_path))
            wandb_norms[wandb_key] = leaf_norm
            if (
                key_path[0] == 0
                and key_path[1] == "attn"
                and key_path[2].key in ["W_q", "W_k", "W_v", "W_o"]
            ):
                logging.info(
                    f"[Step {step}] Norm(layer0.attn.{key_path[2].key}): {leaf_norm:.4f}"
                )

    if use_wandb and wandb.run is not None:
        wandb.log(wandb_norms, step=step)


@partial(jax.jit, static_argnums=(3, 4))
def train_step(params, opt_state, batch, config, optimizer):
    def loss_fn(p):
        input_ids = batch["inputs"]
        targets = batch["targets"]
        positions = batch["positions"]
        batch_size, seq_len = input_ids.shape

        head_dim = config["hidden_size"] // config["num_heads"]

        sin, cos = caching.build_rope(
            positions=positions,
            head_dim=head_dim,
            base=config["rope_base"],
        )

        cache = caching.TransformerCache(
            use_kv=False,
            full_sin=sin,
            full_cos=cos,
            full_positions=positions,
            layers=[
                caching.LayerCache(
                    sin=sin,
                    cos=cos,
                    positions=positions,
                    cached_lens=jnp.zeros((batch_size,), dtype=jnp.int32),
                    keys=None,
                    values=None,
                )
                for _ in range(config["num_layers"])
            ],
        )

        logits, _ = transformer.run(input_ids, cache, p, config)
        mask = positions >= 0
        metrics = compute_metrics(logits, targets, mask)
        return metrics["loss"], metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    metrics["grad_norm"] = optax.global_norm(grads)

    return new_params, new_opt_state, metrics


def train_model(
    config,
    data_iterator,
    num_steps,
    learning_rate=1e-4,
    log_every=100,
    save_every=1000,
    checkpoint_dir="checkpoints",
    resume_from=None,
    seed=2002,
    use_wandb=True,
    wandb_project="jax-transformer-train",
    wandb_entity=None,
    wandb_run_name=None,
):
    logging.info("Starting training...")
    logging.info(f"Config: {config}")

    key = jax.random.PRNGKey(seed)
    model_key, init_key = jax.random.split(key)

    start_step = 0
    resumed_wandb_id = None
    if resume_from and os.path.exists(resume_from):
        logging.info(f"Attempting to resume from checkpoint: {resume_from}")
        params, opt_state, start_step, resumed_wandb_id = load_checkpoint(resume_from)
        if params is None:
            logging.warning("Checkpoint loading failed, initializing from scratch.")
            params = transformer.create(model_key, config)
            opt_state = None
            start_step = 0
            resumed_wandb_id = None
        else:
            logging.info(f"Resumed training from step {start_step}")
    else:
        if resume_from:
            logging.warning(
                f"Resume checkpoint not found: {resume_from}. Initializing from scratch."
            )
        else:
            logging.info("Initializing new model parameters.")
        params = transformer.create(model_key, config)
        opt_state = None

    optimizer = optax.adam(learning_rate=learning_rate, b1=0.9, b2=0.999, eps=1e-8)
    if opt_state is None:
        opt_state = optimizer.init(params)

    logging.info(f"Optimizer: Adam (lr={learning_rate})")
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logging.info(f"Total parameters: {total_params:,}")

    if use_wandb:
        resume_status = "allow" if resumed_wandb_id else None
        wandb_id = resumed_wandb_id

        try:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                config=config,
                resume=resume_status,
                id=wandb_id,
            )
            wandb.config.update(
                {
                    "learning_rate": learning_rate,
                    "num_steps": num_steps,
                    "seed": seed,
                    "total_parameters": total_params,
                    "start_step": start_step,
                }
            )
            logging.info(
                f"Wandb initialized. Run ID: {wandb.run.id}, Resumed: {wandb.run.resumed}"
            )
            current_wandb_id = wandb.run.id
        except Exception as e:
            logging.error(f"Failed to initialize wandb: {e}. Disabling wandb.")
            use_wandb = False
            current_wandb_id = None
    else:
        logging.info("Wandb logging is disabled.")
        current_wandb_id = None

    pbar = tqdm(
        range(start_step, num_steps),
        initial=start_step,
        total=num_steps,
        desc="Training",
    )
    try:
        for step in pbar:
            step_start_time = time.time()

            try:
                batch = next(data_iterator)
                batch = jax.tree_util.tree_map(jnp.asarray, batch)
            except StopIteration:
                logging.warning("Data iterator exhausted. Stopping training.")
                break

            params, opt_state, metrics = train_step(
                params, opt_state, batch, config, optimizer
            )

            metrics = jax.device_get(metrics)
            metrics = {k: v.item() for k, v in metrics.items()}
            step_time = time.time() - step_start_time
            tokens_per_sec = (
                metrics.get("num_valid_tokens", 0) / step_time if step_time > 0 else 0
            )

            if (step + 1) % log_every == 0:
                log_message = (
                    f"[Step {step + 1}/{num_steps}] "
                    f"Loss: {metrics['loss']:.4f}, "
                    f"Acc: {metrics.get('accuracy', -1):.3f}, "
                    f"Grad Norm: {metrics['grad_norm']:.4f}, "
                    f"Valid Tokens: {metrics['num_valid_tokens']:.0f}, "
                    f"Time/Step: {step_time:.3f}s, "
                    f"Tokens/Sec: {tokens_per_sec:.2f}"
                )
                logging.info(log_message)
                pbar.set_postfix(
                    loss=f"{metrics['loss']:.4f}",
                    acc=f"{metrics.get('accuracy', -1):.3f}",
                )

                if use_wandb and wandb.run is not None:
                    wandb_log_data = {
                        "train/loss": metrics["loss"],
                        "train/accuracy": metrics.get("accuracy", -1),
                        "train/grad_norm": metrics["grad_norm"],
                        "train/num_valid_tokens": metrics["num_valid_tokens"],
                        "perf/step_time_s": step_time,
                        "perf/tokens_per_sec": tokens_per_sec,
                        "progress/step": step + 1,
                        "progress/learning_rate": learning_rate,
                    }
                    wandb.log(wandb_log_data, step=step + 1)

                log_weights_biases_norms(params, step + 1, use_wandb=use_wandb)

            if (step + 1) % save_every == 0:
                save_checkpoint(
                    params, opt_state, step + 1, checkpoint_dir, current_wandb_id
                )

        logging.info("Training finished.")

        save_checkpoint(params, opt_state, num_steps, checkpoint_dir, current_wandb_id)

    finally:
        if use_wandb and wandb.run is not None:
            final_step = step + 1 if "step" in locals() else start_step
            if final_step > start_step:
                final_metrics = {f"final_{k}": v for k, v in metrics.items()}
                wandb.log(final_metrics, step=final_step)
                logging.info(f"Logged final metrics to W&B at step {final_step}")

            wandb.finish()
            logging.info("Wandb run finished.")

    return params, opt_state


def train_tokenizer(
    iterator,
    vocab_size,
    save_dir,
    generation_config,
    initial_alphabet=None,
    min_frequency=2,
    special_tokens=None,
    add_prefix_space=False,
    normalize=True,
):
    unk_token = generation_config.setdefault("unk_token", "<unk>")
    generation_config.setdefault("pad_token", "<pad>")
    bos_token = generation_config.setdefault("bos_token", "<bos>")
    eos_token = generation_config.setdefault("eos_token", "<eos>")

    default_special_tokens = [unk_token, bos_token, eos_token]

    if special_tokens is None:
        special_tokens = default_special_tokens
    else:
        existing_tokens = set(special_tokens)
        for token in default_special_tokens:
            if token not in existing_tokens:
                special_tokens.append(token)
        if unk_token not in special_tokens:
            special_tokens.insert(0, unk_token)

    tokenizer = Tokenizer(BPE(unk_token=unk_token))

    if normalize:
        tokenizer.normalizer = Sequence(
            [
                NFD(),
                Lowercase(),
                StripAccents(),
            ]
        )

    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=add_prefix_space)
    tokenizer.decoder = ByteLevelDecoder()

    if initial_alphabet is None:
        initial_alphabet = ByteLevel.alphabet()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=initial_alphabet,
        show_progress=True,
    )

    logging.info("Starting BPE tokenizer training...")
    logging.info(f"Vocab size: {vocab_size}, Min frequency: {min_frequency}")
    logging.info(f"Special tokens: {special_tokens}")
    logging.info(f"Saving tokenizer assets to directory: {save_dir}")

    tokenizer.train_from_iterator(iterator, trainer=trainer)

    logging.info("Training complete.")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Ensured directory exists: {save_dir}")

    tokenizer_file_path = os.path.join(save_dir, "tokenizer.json")
    generation_config_file_path = os.path.join(save_dir, "generation_config.json")

    tokenizer.save(tokenizer_file_path)
    logging.info(f"Tokenizer saved successfully to {tokenizer_file_path}")

    with open(generation_config_file_path, "w", encoding="utf-8") as f:
        json.dump(generation_config, f, indent=2, ensure_ascii=False)

    return tokenizer
