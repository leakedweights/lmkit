import os
import grain.python as grain
import array_record.python.array_record_module as array_record

def to_arrayrecord(data_iter, out_dir, encode_fn, size_limit=1024**3, base_fname="shard"):
    os.makedirs(out_dir, exist_ok=True)
    shard = 0
    current_bytes = 0
    filename = f"{out_dir}/{base_fname}-{shard:05d}"
    writer = array_record.ArrayRecordWriter(filename, "group_size:1")
    
    for data in data_iter:
        record_bytes = encode_fn(data)
        record_size = len(record_bytes)
        if current_bytes + record_size > size_limit:
            writer.close()
            shard += 1
            current_bytes = 0
            filename = f"{out_dir}/{base_fname}-{shard:05d}"
            writer = array_record.ArrayRecordWriter(filename, "group_size:1")
        writer.write(record_bytes)
        current_bytes += record_size
        
    writer.close()


def grain_dataset_from(arrayrecord_dir, batch_size, map_fn=None, batch_map_fn=None, seed=2002):
    datasource = grain.ArrayRecordDataSource(
        [f"{arrayrecord_dir}/{record_path}" for record_path in os.listdir(arrayrecord_dir)]
    )

    dataset = (
        grain.MapDataset.source(datasource)
        .shuffle(seed=seed)
        .map(lambda x: map_fn(x) if map_fn else x)
        .batch(batch_size=batch_size)
        .map(lambda x: batch_map_fn(x) if batch_map_fn else x)
    )

    return dataset
