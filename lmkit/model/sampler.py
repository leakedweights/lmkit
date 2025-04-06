import jax.numpy as jnp
from tqdm.auto import tqdm

from . import transformer
from .caching import TransformerCache

# TODO: replace greedy decoding with temperature sampling
# TODO: add break upon EOS token


def generate(
    inputs, max_new_tokens, tokenizer, params, config, return_text=True, verbose=False
):
    batch_size = len(inputs)

    encodings = tokenizer.encode_batch_fast(inputs)
    tokens = jnp.array([enc.ids for enc in encodings])
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

    for _ in step_iter:
        logits, cache = transformer.run(model_inputs, cache, params, config)
        next_token_logits = logits[jnp.arange(batch_size), seq_lens - 1, :]
        next_tokens = jnp.argmax(next_token_logits, axis=-1)

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
