from functools import partial

import jax
import jax.numpy as jnp
from jax import random
from tqdm.auto import tqdm

from . import transformer
from .caching import TransformerCache


@partial(jax.jit, static_argnums=(2, 3))
def sample_step(logits, random_key, top_p, temp):
    logits = logits / temp

    sorted_indices = jnp.argsort(logits, axis=-1)[..., ::-1]
    sorted_logits = jnp.sort(logits, axis=-1)[..., ::-1]

    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

    cutoff = cumulative_probs > top_p
    cutoff = jnp.pad(cutoff[..., :-1], ((0, 0), (1, 0)), constant_values=False)

    masked_logits = jnp.where(cutoff, -jnp.inf, sorted_logits)

    sampled_indices_in_sorted = jax.random.categorical(
        random_key, masked_logits, axis=-1
    )

    next_token = jnp.take_along_axis(
        sorted_indices,
        sampled_indices_in_sorted[..., None],
        axis=-1,
    ).squeeze(-1)

    return next_token


def generate(
    max_new_tokens,
    tokenizer,
    params,
    config,
    random_key,
    inputs=None,
    tokenized_inputs=None,
    temp=0.6,
    top_p=0.9,
    return_text=True,
    verbose=False,
):
    if inputs is None:
        if tokenized_inputs is None:
            raise ValueError("Either raw inputs or encoded inputs must be specified")

        batch_size = tokenized_inputs.shape[0]
        tokens = tokenized_inputs
    else:
        batch_size = len(inputs)
        encodings = tokenizer.encode_batch_fast(inputs)
        tokens = tokens = jnp.array([enc.ids for enc in encodings])

    tokens = jnp.concatenate(
        [tokens, tokenizer.pad_token_id * jnp.ones((batch_size, max_new_tokens))],
        axis=-1,
    ).astype(jnp.int32)
    positions = jnp.where(
        tokens != tokenizer.pad_token_id, jnp.arange(tokens.shape[-1]), -1
    )
    seq_lens = jnp.sum(positions >= 0, axis=-1)

    model_inputs = tokens

    cache = TransformerCache.initialize(
        batch_size=batch_size,
        current_positions=positions,
        config=config,
        max_total_length=jnp.max(seq_lens + max_new_tokens),
        use_kv=True,
    )

    step_iter = range(max_new_tokens)
    if verbose:
        step_iter = tqdm(step_iter)

    for step in step_iter:
        logits, cache = transformer.run(model_inputs, cache, params, config)
        next_token_logits = logits[jnp.arange(batch_size), seq_lens - 1, :]

        step_key = random.fold_in(random_key, step)
        next_tokens = sample_step(
            logits=next_token_logits, random_key=step_key, top_p=top_p, temp=temp
        )

        batch_indices = jnp.arange(batch_size).astype(jnp.int32)
        tokens = tokens.at[batch_indices, seq_lens].set(next_tokens)

        model_inputs = next_tokens[..., None] if cache.use_kv else tokens
        cache = cache.roll()
        seq_lens += 1

    if return_text:
        return tokenizer.decode_batch(
            jnp.where(tokens >= 0, tokens, tokenizer.pad_token_id)
        )
    return tokens
