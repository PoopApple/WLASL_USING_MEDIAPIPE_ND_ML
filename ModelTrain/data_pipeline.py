"""
tf.data input pipeline for ASL landmark dataset.

Loads individual .npz files lazily from disk (no need to fit entire dataset in RAM).
Supports configurable number of words (top-N by sample count).
Handles train/val/test splitting with stratification.

Usage:
    from data_pipeline import ASLDataPipeline

    pipeline = ASLDataPipeline(
        dataset_dir="../ExtractLandmarks/dataset3.0/landmarks_npz",
        num_words=500,
        batch_size=64,
        val_split=0.15,
        test_split=0.05,
        seed=1234,
    )
    train_ds, val_ds, test_ds = pipeline.get_datasets()
    word_to_ind = pipeline.word_to_ind
    num_classes = pipeline.num_classes
"""

import os
import json
import numpy as np
import tensorflow as tf
from collections import defaultdict
from sklearn.model_selection import train_test_split


MAX_FRAMES = 64   # 95.4% of trimmed segments fit in 64f; median=37f (was 128)
NUM_FEATURES = 64
FEATURE_DIM = 4


class ASLDataPipeline:
    def __init__(
        self,
        dataset_dir: str,
        num_words: int | None = None,
        batch_size: int = 64,
        val_split: float = 0.15,
        test_split: float = 0.05,
        seed: int = 1234,
        output_dir: str = "./dataset4.0",
    ):
        """
        Args:
            dataset_dir:  Path to the landmarks_npz/ folder with word subfolders.
            num_words:    Number of top words (by sample count) to include.
                          None = use all words.
            batch_size:   Batch size for training.
            val_split:    Fraction of data for validation.
            test_split:   Fraction of data for test.
            seed:         Random seed for reproducibility.
            output_dir:   Where to save word_to_ind.json and split manifests.
        """
        self.dataset_dir = dataset_dir
        self.num_words = num_words
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Build file manifest
        self._build_manifest()

    # ──────────────────────────────────────────────────────────────────────
    # Manifest building
    # ──────────────────────────────────────────────────────────────────────

    def _build_manifest(self):
        """Scan dataset_dir, select top-N words, build file paths + labels."""
        word_dirs = sorted(
            [
                d
                for d in os.listdir(self.dataset_dir)
                if os.path.isdir(os.path.join(self.dataset_dir, d))
            ]
        )

        # Count samples per word
        word_sample_counts = {}
        for word in word_dirs:
            word_path = os.path.join(self.dataset_dir, word)
            npz_files = [f for f in os.listdir(word_path) if f.endswith(".npz")]
            word_sample_counts[word] = len(npz_files)

        # Select top-N words by sample count (or all)
        sorted_words = sorted(
            word_sample_counts.items(), key=lambda x: x[1], reverse=True
        )

        if self.num_words is not None:
            selected_words = [w for w, _ in sorted_words[: self.num_words]]
        else:
            selected_words = [w for w, _ in sorted_words]

        selected_words.sort()  # Alphabetical for deterministic label assignment

        # Build label mapping
        self.word_to_ind = {word: idx for idx, word in enumerate(selected_words)}
        self.ind_to_word = {idx: word for word, idx in self.word_to_ind.items()}
        self.num_classes = len(selected_words)

        # Save label mapping
        map_path = os.path.join(
            self.output_dir,
            f"word_to_ind_{self.num_words or 'all'}.json",
        )
        with open(map_path, "w") as f:
            json.dump(self.word_to_ind, f, indent=2)
        print(f"Saved label mapping to {map_path}")

        # Build (file_path, label) list
        all_paths = []
        all_labels = []
        for word in selected_words:
            word_path = os.path.join(self.dataset_dir, word)
            for fname in sorted(os.listdir(word_path)):
                if fname.endswith(".npz"):
                    all_paths.append(os.path.join(word_path, fname))
                    all_labels.append(self.word_to_ind[word])

        self.all_paths = np.array(all_paths)
        self.all_labels = np.array(all_labels, dtype=np.int32)

        print(
            f"Selected {self.num_classes} words, "
            f"{len(self.all_paths)} total samples"
        )

        # Stratified split: first split off test, then split remainder into train/val
        if self.test_split > 0:
            paths_trainval, paths_test, labels_trainval, labels_test = (
                train_test_split(
                    self.all_paths,
                    self.all_labels,
                    test_size=self.test_split,
                    random_state=self.seed,
                    stratify=self.all_labels,
                )
            )
        else:
            paths_trainval = self.all_paths
            labels_trainval = self.all_labels
            paths_test = np.array([], dtype=str)
            labels_test = np.array([], dtype=np.int32)

        # Split trainval into train and val
        relative_val = self.val_split / (1 - self.test_split)
        paths_train, paths_val, labels_train, labels_val = train_test_split(
            paths_trainval,
            labels_trainval,
            test_size=relative_val,
            random_state=self.seed,
            stratify=labels_trainval,
        )

        self.splits = {
            "train": (paths_train, labels_train),
            "val": (paths_val, labels_val),
            "test": (paths_test, labels_test),
        }

        print(
            f"Split sizes — "
            f"train: {len(paths_train)}, "
            f"val: {len(paths_val)}, "
            f"test: {len(paths_test)}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # tf.data dataset creation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_npz(file_path_tensor, label_tensor):
        """Load a single .npz file using tf.numpy_function."""

        def _load_fn(path_bytes, label):
            path_str = path_bytes.decode("utf-8")
            # Use context manager to ensure file handles are closed promptly
            with np.load(path_str) as npz:
                data = npz["data"].astype(np.float32)  # (128, 64, 4)
                mask = npz["mask"].astype(np.bool_)  # (128,)
            return data, mask, label

        data, mask, label = tf.numpy_function(
            _load_fn,
            [file_path_tensor, label_tensor],
            [tf.float32, tf.bool, tf.int32],
        )

        # Set static shapes (critical for model compatibility)
        data.set_shape([MAX_FRAMES, NUM_FEATURES, FEATURE_DIM])
        mask.set_shape([MAX_FRAMES])
        label.set_shape([])

        return {"input_data": data, "input_mask": mask}, label

    def _make_dataset(self, paths, labels, shuffle: bool, repeat: bool = False):
        """Create a tf.data.Dataset from file paths and labels.

        Uses interleave to load multiple files concurrently, bypassing
        the GIL bottleneck of tf.numpy_function.
        """
        ds = tf.data.Dataset.from_tensor_slices(
            (paths.astype(str), labels.astype(np.int32))
        )

        if shuffle:
            ds = ds.shuffle(
                buffer_size=min(len(paths), 50000),
                seed=self.seed,
                reshuffle_each_iteration=True,
            )

        # Interleave: create mini-datasets of 1 element each, then
        # load cycle_length of them concurrently across threads.
        # This is the key trick to keep the GPU fed despite numpy_function.
        ds = ds.interleave(
            lambda path, label: tf.data.Dataset.from_tensors((path, label)).map(
                self._load_npz, num_parallel_calls=tf.data.AUTOTUNE
            ),
            cycle_length=8,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not shuffle,
        )

        ds = ds.batch(self.batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        if repeat:
            ds = ds.repeat()

        return ds

    # ──────────────────────────────────────────────────────────────────────
    # TFRecord-based loading (faster — no Python GIL)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_tfrecord(serialized):
        """Parse a single TFRecord example — pure TF ops, no Python."""
        feature_spec = {
            "data": tf.io.FixedLenFeature([], tf.string),
            "mask": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(serialized, feature_spec)

        data = tf.io.decode_raw(example["data"], tf.float32)
        data = tf.reshape(data, [MAX_FRAMES, NUM_FEATURES, FEATURE_DIM])

        # TFRecord stores raw bytes — decode as uint8 then cast to bool.
        mask = tf.io.decode_raw(example["mask"], tf.uint8)
        mask = tf.reshape(mask, [MAX_FRAMES])
        # mask = tf.cast(mask, tf.bool)

        label = tf.cast(example["label"], tf.int32)

        return {"input_data": data, "input_mask": mask}, label

    def _make_dataset_tfrecord(self, tfrecord_dir: str, shuffle: bool, repeat: bool = False, augment: bool = False):
        """Create a tf.data.Dataset from TFRecord shard files."""
        import glob

        shard_files = sorted(glob.glob(os.path.join(tfrecord_dir, "*.tfrecord")))
        if not shard_files:
            raise FileNotFoundError(f"No .tfrecord files found in {tfrecord_dir}")

        ds = tf.data.TFRecordDataset(
            shard_files,
            num_parallel_reads=tf.data.AUTOTUNE,
            buffer_size=10 * 1024 * 1024,  # 10 MB buffer to silence the unspecified warning
        )

        if shuffle:
            ds = ds.shuffle(
                buffer_size=2000,  # Lowered from 10000 to drastically save RAM
                reshuffle_each_iteration=True,
            )

        ds = ds.map(self._parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

        if augment:
            from augmentation import augment_sample
            ds = ds.map(augment_sample, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(self.batch_size, drop_remainder=False)
        ds = ds.prefetch(2)  # Lowered from AUTOTUNE to prevent RAM bloating

        if repeat:
            ds = ds.repeat()

        return ds

    def get_datasets(self, use_tfrecord: bool = False, augment: bool = False):
        """
        Returns:
            (train_ds, val_ds, test_ds) — each is a tf.data.Dataset
            yielding ({"input_data": ..., "input_mask": ...}, label)

        Args:
            use_tfrecord: If True, load from TFRecord shards instead of npz files.
            augment: If True, apply data augmentation to the TRAINING set only.
        """
        train_count = len(self.splits["train"][0])
        val_count = len(self.splits["val"][0])
        test_count = len(self.splits["test"][0])

        print(f"\n{'─'*50}")
        print(f"  Data loading:")
        print(f"    Source:        {'TFRecord shards' if use_tfrecord else '.npz files'}")
        print(f"    Train samples: {train_count}")
        print(f"    Val samples:   {val_count}")
        print(f"    Test samples:  {test_count}")
        print(f"    Augmentation:  {'ON (train only)' if augment else 'OFF'}")
        if augment:
            from augmentation import NOISE_STDDEV, SCALE_RANGE, FRAME_DROP_RATE, LANDMARK_DROP_RATE
            print(f"      Noise σ:       {NOISE_STDDEV}")
            print(f"      Scale range:   {SCALE_RANGE[0]}–{SCALE_RANGE[1]}")
            print(f"      Frame drop:    {FRAME_DROP_RATE*100:.0f}%")
            print(f"      Landmark drop: {LANDMARK_DROP_RATE*100:.0f}%")
        print(f"{'─'*50}\n")

        if use_tfrecord:
            tfrecord_base = os.path.join(self.output_dir, "tfrecords")
            train_ds = self._make_dataset_tfrecord(
                os.path.join(tfrecord_base, "train"), shuffle=True, repeat=True, augment=augment
            )
            val_ds = self._make_dataset_tfrecord(
                os.path.join(tfrecord_base, "val"), shuffle=False
            )
            test_dir = os.path.join(tfrecord_base, "test")
            if os.path.isdir(test_dir) and os.listdir(test_dir):
                test_ds = self._make_dataset_tfrecord(test_dir, shuffle=False)
            else:
                test_ds = None
            return train_ds, val_ds, test_ds

        # Fallback: npz loading
        train_paths, train_labels = self.splits["train"]
        val_paths, val_labels = self.splits["val"]
        test_paths, test_labels = self.splits["test"]

        train_ds = self._make_dataset(train_paths, train_labels, shuffle=True, repeat=True)

        if augment:
            from augmentation import augment_sample
            train_ds = train_ds.map(augment_sample, num_parallel_calls=tf.data.AUTOTUNE)

        val_ds = self._make_dataset(val_paths, val_labels, shuffle=False)

        if len(test_paths) > 0:
            test_ds = self._make_dataset(test_paths, test_labels, shuffle=False)
        else:
            test_ds = None

        return train_ds, val_ds, test_ds

    def get_steps_per_epoch(self):
        """Returns (train_steps, val_steps) for use with model.fit()."""
        train_steps = int(np.ceil(len(self.splits["train"][0]) / self.batch_size))
        val_steps = int(np.ceil(len(self.splits["val"][0]) / self.batch_size))
        return train_steps, val_steps

    def get_class_weights(self):
        """Compute class weights to handle class imbalance."""
        train_labels = self.splits["train"][1]
        counts = np.bincount(train_labels, minlength=self.num_classes)
        total = counts.sum()
        # Smooth the weights with square root so the tail isn't overly punishing
        weights = np.sqrt(total / (self.num_classes * counts.astype(np.float64) + 1e-8))
        # Normalize so mean weight = 1
        weights = weights / weights.mean()
        # Clip max weight to 5.0 to prevent gradient explosions on extremely rare classes
        weights = np.clip(weights, 0.1, 5.0)
        return {i: w for i, w in enumerate(weights)}


# ──────────────────────────────────────────────────────────────────────────
# Quick test / sanity check
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = ASLDataPipeline(
        dataset_dir="../ExtractLandmarks/dataset3.0/landmarks_npz",
        num_words=500,
        batch_size=64,
        val_split=0.15,
        test_split=0.05,
        seed=1234,
    )

    train_ds, val_ds, test_ds = pipeline.get_datasets()
    train_steps, val_steps = pipeline.get_steps_per_epoch()

    print(f"\nNum classes: {pipeline.num_classes}")
    print(f"Train steps/epoch: {train_steps}")
    print(f"Val steps/epoch: {val_steps}")

    # Verify one batch
    for batch_inputs, batch_labels in train_ds.take(1):
        print(f"\nBatch data shape:  {batch_inputs['input_data'].shape}")
        print(f"Batch mask shape:  {batch_inputs['input_mask'].shape}")
        print(f"Batch label shape: {batch_labels.shape}")
        print(f"Label sample:      {batch_labels[:5].numpy()}")
        print(f"Mask sample (first 10 of first sample): "
              f"{batch_inputs['input_mask'][0, :10].numpy()}")

    # Class weights
    cw = pipeline.get_class_weights()
    weights_arr = np.array(list(cw.values()))
    print(f"\nClass weights — min: {weights_arr.min():.3f}, "
          f"max: {weights_arr.max():.3f}, mean: {weights_arr.mean():.3f}")
