import os

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm


def dir_line_iterator(directory, separator="\n", verbose=False):
    dataset_files = [os.path.join(directory, fname) for fname in os.listdir(directory)]

    iter = dataset_files if not verbose else tqdm(dataset_files, desc="Processed files")

    for file in iter:
        with open(file, "r") as file:
            lines = file.read().split(separator)
            for line in lines:
                yield line


def to_tfrecord(
    data_iter,
    out_dir,
    encode_fn,
    feature_name="data",
    size_limit=1024**3,
    base_fname="shard",
):
    os.makedirs(out_dir, exist_ok=True)
    shard = 0
    current_bytes = 0
    filename = os.path.join(out_dir, f"{base_fname}-{shard:05d}.tfrecord")
    writer = tf.io.TFRecordWriter(filename)

    iterator = tqdm(data_iter)

    for example in iterator:
        record_bytes = encode_fn(example)
        record_size = len(record_bytes)

        if current_bytes > 0 and current_bytes + record_size > size_limit:
            writer.close()
            shard += 1
            current_bytes = 0
            filename = os.path.join(out_dir, f"{base_fname}-{shard:05d}.tfrecord")
            writer = tf.io.TFRecordWriter(filename)  # Start new writer

        feature = {
            feature_name: tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[record_bytes])
            )
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        serialized_example = example_proto.SerializeToString()

        writer.write(serialized_example)
        current_bytes += len(serialized_example)

    writer.close()


def create_dataset(
    tfrecord_files,
    tokenizer,
    batch_size,
    feature_name="data",
    shuffle_buffer_size=10000,
    seed=2002,
    num_parallel_calls=tf.data.AUTOTUNE,
    prefetch_buffer_size=tf.data.AUTOTUNE,
):
    if not tfrecord_files:
        raise ValueError("tfrecord_files list cannot be empty.")

    if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have a valid 'pad_token_id' attribute.")

    pad_token_id = tokenizer.pad_token_id

    if len(tfrecord_files) > 1:
        dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=num_parallel_calls,
            deterministic=False,
        )
    else:
        dataset = tf.data.TFRecordDataset(
            tfrecord_files[0], num_parallel_reads=num_parallel_calls
        )

    feature_description = {feature_name: tf.io.FixedLenFeature([], tf.string)}

    def _parse_tfrecord(proto):
        parsed_features = tf.io.parse_single_example(proto, feature_description)
        return parsed_features[feature_name]

    dataset = dataset.map(_parse_tfrecord, num_parallel_calls=num_parallel_calls)
    dataset = dataset.filter(lambda x: tf.strings.length(x) > 0)
    if shuffle_buffer_size and shuffle_buffer_size > 0:
        dataset = dataset.shuffle(
            shuffle_buffer_size, seed=seed, reshuffle_each_iteration=True
        )

    def _tokenize_single_py(byte_string):
        text = byte_string.numpy().decode("utf-8")
        token_ids = tokenizer.encode(text).ids
        return np.array(token_ids, dtype=np.int32)

    dataset = dataset.map(
        lambda byte_string: tf.py_function(
            func=_tokenize_single_py,
            inp=[byte_string],
            Tout=tf.int32,
        ),
        num_parallel_calls=num_parallel_calls,
    )
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=[None],
        padding_values=tf.constant(pad_token_id, dtype=tf.int32),
        drop_remainder=True,
    )

    def _create_transformer_features(tokens):
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        current_len = tf.shape(inputs)[1]  # Length after slicing
        needs_padding = tf.cast(tf.math.mod(current_len, 2), dtype=tf.bool)
        paddings = tf.constant([[0, 0], [0, 1]], dtype=tf.int32)

        inputs = tf.cond(
            needs_padding,
            lambda: tf.pad(inputs, paddings, "CONSTANT", constant_values=pad_token_id),
            lambda: inputs,
        )
        targets = tf.cond(
            needs_padding,
            lambda: tf.pad(targets, paddings, "CONSTANT", constant_values=pad_token_id),
            lambda: targets,
        )

        final_len = tf.shape(inputs)[1]
        positions_base = tf.range(final_len, dtype=tf.int32)
        positions_base = tf.expand_dims(positions_base, 0)
        positions = tf.where(inputs != pad_token_id, positions_base, -1)

        return {"inputs": inputs, "targets": targets, "positions": positions}

    dataset = dataset.map(
        _create_transformer_features, num_parallel_calls=num_parallel_calls
    )

    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

    return dataset


def from_tfrecords_dir(tfrecords_dir, *args, **kwargs):
    tfrecord_files = tf.io.gfile.glob(os.path.join(tfrecords_dir, "*.tfrecord"))
    dataset = create_dataset(tfrecord_files=tfrecord_files, *args, **kwargs)

    return dataset
