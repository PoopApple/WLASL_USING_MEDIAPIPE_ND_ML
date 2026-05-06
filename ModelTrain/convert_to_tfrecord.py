"""
Convert npz dataset to TFRecord shards for faster training.

Uses the ASLDataPipeline to get the train/val/test splits, then writes
each split into TFRecord shard files (~1000 samples per shard).

Usage:
    uv run convert_to_tfrecord.py
"""

import os
import numpy as np
import tensorflow as tf
from data_pipeline import ASLDataPipeline

# ══════════════════════════════════════════════════════════════════════
# CONFIG — edit these directly (keep in sync with train_v2.py)
# ══════════════════════════════════════════════════════════════════════

DATASET_DIR = "../ExtractLandmarks/dataset4.0/landmarks_npz"
NUM_WORDS = None
BATCH_SIZE = 64
VAL_SPLIT = 0.20
TEST_SPLIT = 0.00
SEED = 1234
SAMPLES_PER_SHARD = 1000

# ══════════════════════════════════════════════════════════════════════


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(data: np.ndarray, mask: np.ndarray, label: int) -> bytes:
    """Serialize a single sample to a tf.train.Example."""
    feature = {
        "data": _bytes_feature(data.astype(np.float32).tobytes()),
        "mask": _bytes_feature(mask.astype(np.bool_).tobytes()),
        "label": _int64_feature(label),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def write_shards(paths: np.ndarray, labels: np.ndarray, output_dir: str, split_name: str):
    """Write a split (train/val/test) into TFRecord shards."""
    os.makedirs(output_dir, exist_ok=True)

    num_samples = len(paths)
    num_shards = max(1, (num_samples + SAMPLES_PER_SHARD - 1) // SAMPLES_PER_SHARD)

    print(f"  {split_name}: {num_samples} samples → {num_shards} shards")

    written = 0
    for shard_idx in range(num_shards):
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:03d}.tfrecord")
        start = shard_idx * SAMPLES_PER_SHARD
        end = min(start + SAMPLES_PER_SHARD, num_samples)

        with tf.io.TFRecordWriter(shard_path) as writer:
            for i in range(start, end):
                npz = np.load(paths[i])
                data = npz["data"]   # (64, 64, 4) float32
                mask = npz["mask"]   # (64,) bool
                label = int(labels[i])

                writer.write(serialize_example(data, mask, label))
                written += 1

                if written % 2000 == 0:
                    print(f"    written {written}/{num_samples}")

    print(f"  {split_name} done: {written} samples written")


def main():
    output_dir = f"./dataset4.0/{NUM_WORDS or 'all'}words"

    # Use the same pipeline to get identical splits
    pipeline = ASLDataPipeline(
        dataset_dir=DATASET_DIR,
        num_words=NUM_WORDS,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        seed=SEED,
        output_dir=output_dir,
    )

    tfrecord_dir = os.path.join(output_dir, "tfrecords")
    print(f"\nWriting TFRecords to: {tfrecord_dir}\n")

    for split_name in ["train", "val", "test"]:
        paths, labels = pipeline.splits[split_name]
        if len(paths) == 0:
            print(f"  {split_name}: skipped (empty)")
            continue
        split_dir = os.path.join(tfrecord_dir, split_name)
        write_shards(paths, labels, split_dir, split_name)

    print(f"\nDone! TFRecords saved to {tfrecord_dir}")
    print(f"Label mapping: {output_dir}/word_to_ind_{NUM_WORDS or 'all'}.json")


if __name__ == "__main__":
    main()
