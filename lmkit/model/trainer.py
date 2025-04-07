import jax
import jax.numpy as jnp
import optax
import pickle
import os
import time
from tqdm.auto import tqdm
import logging
from functools import partial

from . import transformer
from . import caching

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Helper Functions ---

def save_checkpoint(params, opt_state, step, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pkl")
    checkpoint = {
        "params": jax.device_get(params),
        "opt_state": jax.device_get(opt_state),
        "step": step,
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)
    logging.info(f"Checkpoint saved at step {step} to {checkpoint_path}")


def load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        logging.warning(f"Checkpoint file not found: {checkpoint_path}")
        return None, None, 0

    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    logging.info(
        f"Loaded checkpoint from step {checkpoint.get('step', 0)}: {checkpoint_path}"
    )
    return checkpoint["params"], checkpoint["opt_state"], checkpoint["step"]


def compute_metrics(logits, targets, mask):
    mask = mask.astype(jnp.bool_)
    num_valid_tokens = jnp.maximum(jnp.sum(mask), 1)

    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    masked_loss = per_token_loss * mask
    mean_loss = jnp.sum(masked_loss) / num_valid_tokens

    # Accuracy Calculation (Optional but good to have)
    predictions = jnp.argmax(logits, axis=-1)
    correct_predictions = (predictions == targets) * mask
    accuracy = jnp.sum(correct_predictions) / num_valid_tokens

    metrics = {
        "loss": mean_loss,
        "accuracy": accuracy,
        "num_valid_tokens": num_valid_tokens,
    }
    return metrics


def log_weights_biases_norms(params, step):
    param_norms = jax.tree_util.tree_map(
        lambda x: jnp.linalg.norm(x.astype(jnp.float32)), params
    )

    global_param_norm = optax.global_norm(params)
    logging.info(f"[Step {step}] Global Param Norm: {global_param_norm:.4f}")

    if "embed_table" in params:
        logging.info(
            f"[Step {step}] Norm(embed_table): {param_norms['embed_table']:.4f}"
        )
    if "layers" in params and len(params["layers"]) > 0:
        layer0_attn = params["layers"][0].get("attn", {})
        for w_name in ["W_q", "W_k", "W_v", "W_o"]:
            if w_name in layer0_attn:
                logging.info(
                    f"[Step {step}] Norm(layer0.attn.{w_name}): {param_norms['layers'][0]['attn'][w_name]:.4f}"
                )

# --- Training Step Function ---


@partial(jax.jit, static_argnums=(3,))
def train_step(params, opt_state, batch, config, optimizer):

    # --- Loss Function Definition (inside train_step for closure) ---
    def loss_fn(p):
        input_ids = batch["input_ids"]
        targets = batch["target_ids"]
        positions = batch[
            "positions"
        ]
        batch_size, seq_len = input_ids.shape

        cache = caching.TransformerCache.initialize(
            batch_size=batch_size,
            current_positions=positions,
            config=config,
            max_total_length=seq_len,
            use_kv=False,
        )

        # Forward pass
        logits, _ = transformer.run(input_ids, cache, p, config)

        # Create mask: True for valid positions (>= 0), False for padding (-1)
        mask = positions >= 0

        # Compute loss (and optionally accuracy) using the mask
        metrics = compute_metrics(logits, targets, mask)
        return metrics["loss"], metrics  # Return loss and other metrics

    # --- Gradient Calculation and Parameter Update ---
    # Calculate loss, metrics, and gradients
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # Compute updates based on gradients
    updates, new_opt_state = optimizer.update(grads, opt_state, params)

    # Apply updates to parameters
    new_params = optax.apply_updates(params, updates)

    # Add gradient norm to metrics for logging
    metrics["grad_norm"] = optax.global_norm(grads)

    return new_params, new_opt_state, metrics


# --- Main Training Function ---

def train(
    config: dict,
    data_iterator: iter,  # Should yield dictionaries like {'input_ids': ..., 'target_ids': ..., 'positions': ...}
    num_steps: int,
    learning_rate: float = 1e-4,
    log_every: int = 100,
    save_every: int = 1000,
    checkpoint_dir: str = "checkpoints",
    resume_from: str = None,
    seed: int = 2002,
):
    logging.info("Starting training...")
    logging.info(f"Config: {config}")

    # --- Initialization ---
    key = jax.random.PRNGKey(seed)
    model_key, init_key = jax.random.split(key)

    # Initialize parameters or load from checkpoint
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        logging.info(f"Attempting to resume from checkpoint: {resume_from}")
        params, opt_state, start_step = load_checkpoint(resume_from)
        if params is None:  # Check if loading failed
            logging.warning("Checkpoint loading failed, initializing from scratch.")
            params = transformer.create(model_key, config)
            opt_state = None  # Will be initialized below
            start_step = 0
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
        opt_state = None  # Will be initialized below

    # Initialize Optimizer (Adam with standard settings)
    # You can customize b1, b2, eps if needed, but these are standard.
    optimizer = optax.adam(learning_rate=learning_rate, b1=0.9, b2=0.999, eps=1e-8)

    # Initialize optimizer state if not loaded from checkpoint
    if opt_state is None:
        opt_state = optimizer.init(params)

    logging.info(f"Optimizer: Adam (lr={learning_rate})")
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logging.info(f"Total parameters: {total_params:,}")

    # --- Training Loop ---
    pbar = tqdm(
        range(start_step, num_steps),
        initial=start_step,
        total=num_steps,
        desc="Training",
    )
    for step in pbar:
        start_time = time.time()

        try:
            batch = next(data_iterator)
            # Ensure batch arrays are JAX arrays (might already be)
            batch = jax.tree_util.tree_map(jnp.asarray, batch)
        except StopIteration:
            logging.warning("Data iterator exhausted. Stopping training.")
            break

        # Perform one training step
        params, opt_state, metrics = train_step(
            params, opt_state, batch, config, optimizer
        )

        # Ensure metrics are concrete values for logging (move from device)
        metrics = jax.device_get(metrics)
        step_time = time.time() - start_time

        # --- Logging ---
        if (step + 1) % log_every == 0:
            log_message = (
                f"[Step {step + 1}/{num_steps}] "
                f"Loss: {metrics['loss']:.4f}, "
                f"Acc: {metrics.get('accuracy', -1):.3f}, "  # Use .get in case accuracy isn't computed
                f"Grad Norm: {metrics['grad_norm']:.4f}, "
                f"Valid Tokens: {metrics['num_valid_tokens']:.0f}, "
                f"Time/Step: {step_time:.3f}s"
            )
            logging.info(log_message)
            pbar.set_postfix(
                loss=f"{metrics['loss']:.4f}", acc=f"{metrics.get('accuracy', -1):.3f}"
            )

            # Log weight/bias norms
            log_weights_biases_norms(params, step + 1)

        # --- Checkpointing ---
        if (step + 1) % save_every == 0:
            save_checkpoint(params, opt_state, step + 1, checkpoint_dir)

    logging.info("Training finished.")
    # Save final checkpoint
    save_checkpoint(params, opt_state, num_steps, checkpoint_dir)

    return params, opt_state
