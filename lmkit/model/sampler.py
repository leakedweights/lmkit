import jax
import jax.numpy as jnp
from jax import random
from flax.core import FrozenDict

from lmkit.model import transformer

def init_cache(config):
    return FrozenDict({}) # todo

def generate(key, start_sequences, params, config, tokenizer, max_steps, batch_size, temperature):
    max_len = max_steps + 1

    def continue_generation(carry):
        step, _, seq, *_ = carry
        not_at_max = step <= max_len
        all_finished = jnp.all(jnp.any(seq == tokenizer.eos_token_id, axis=1))
        return not_at_max & (~all_finished)

    def sample_fn(carry):
        step, key, inputs, lengths, cache = carry

        outputs, cache = transformer.run_decoder(inputs, lengths, cache, params, config)
        logits = outputs[:, step, :]

        gumbel_key = random.fold_in(key, step + 1)
        gumbel_noise = -jnp.log(-jnp.log(random.uniform(gumbel_key, logits.shape)))

        noisy_logits = logits + gumbel_noise
        if temperature > 0:
            noisy_logits = noisy_logits / temperature

        if temperature == 0:
            sample = jnp.argmax(noisy_logits, axis=-1)
        else:
            sample = random.categorical(key, noisy_logits, axis=-1)

        new_inputs = inputs.at[:, step + 1].set(sample)
        new_cache = cache

        return (step + 1, key, new_inputs, lengths+1, new_cache)

    step = 0
    cache = None
    start_lengths = jnp.array([len(seq) for seq in start_sequences])
    inputs = jnp.full((batch_size, max_len), tokenizer.pad_token_id)
    for i, seq in enumerate(start_sequences):
        seq_array = jnp.array(seq)
        seq_len = seq_array.shape[0]
        inputs = inputs.at[i, :seq_len].set(seq_array)


    loop_inputs = (step, key, inputs, start_lengths, cache)

    final_step, _, outputs, _, _ = jax.lax.while_loop(
        continue_generation, sample_fn, loop_inputs
    )

    return final_step, outputs
