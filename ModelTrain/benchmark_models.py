"""
Quick viability check for all models in the zoo.
Builds each model, counts params, runs a timed forward + backward pass.
No training needed — just a sanity check.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import numpy as np
import tensorflow as tf
from model import MODEL_REGISTRY, build_model, MAX_FRAMES

NUM_CLASSES = 500
BATCH_SIZE = 64

def benchmark_one(name: str):
    """Build, forward pass, backward pass, report."""
    print(f"\n{'='*60}")
    print(f"  {name.upper()}")
    print(f"{'='*60}")

    # Build
    try:
        model = build_model(name, num_classes=NUM_CLASSES)
    except Exception as e:
        print(f"  ❌ BUILD FAILED: {e}")
        return None

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Count params
    total_params = model.count_params()
    trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable = total_params - trainable

    print(f"  Total params:     {total_params:>10,}")
    print(f"  Trainable:        {trainable:>10,}")
    print(f"  Non-trainable:    {non_trainable:>10,}")

    # Dummy data
    dummy_data = np.random.randn(BATCH_SIZE, MAX_FRAMES, 64, 4).astype(np.float32)
    dummy_mask = np.ones((BATCH_SIZE, MAX_FRAMES), dtype=bool)
    dummy_mask[:, -20:] = False  # simulate some padding
    dummy_labels = np.random.randint(0, NUM_CLASSES, size=(BATCH_SIZE,)).astype(np.int32)

    inputs = {"input_data": dummy_data, "input_mask": dummy_mask}

    # Warm-up (first pass is always slow due to graph tracing)
    try:
        model.predict(inputs, batch_size=BATCH_SIZE, verbose=0)
    except Exception as e:
        print(f"  ❌ FORWARD PASS FAILED: {e}")
        del model
        tf.keras.backend.clear_session()
        return None

    # Timed forward pass (inference)
    t0 = time.perf_counter()
    for _ in range(3):
        model.predict(inputs, batch_size=BATCH_SIZE, verbose=0)
    fwd_time = (time.perf_counter() - t0) / 3

    # Timed training step (forward + backward + weight update)
    t0 = time.perf_counter()
    for _ in range(3):
        model.train_on_batch(inputs, dummy_labels)
    train_time = (time.perf_counter() - t0) / 3

    print(f"  Forward pass:     {fwd_time*1000:>10.1f} ms  (batch={BATCH_SIZE})")
    print(f"  Train step:       {train_time*1000:>10.1f} ms  (batch={BATCH_SIZE})")
    print(f"  Per-sample fwd:   {fwd_time/BATCH_SIZE*1000:>10.2f} ms")

    # Output shape check
    out = model.predict(inputs, batch_size=BATCH_SIZE, verbose=0)
    print(f"  Output shape:     {out.shape}  (expected: ({BATCH_SIZE}, {NUM_CLASSES}))")
    print(f"  Output sum≈1?     {out[0].sum():.4f}  (softmax check)")

    result = {
        "name": name,
        "params": total_params,
        "trainable": trainable,
        "fwd_ms": fwd_time * 1000,
        "train_ms": train_time * 1000,
        "status": "✅",
    }

    # Cleanup
    del model
    tf.keras.backend.clear_session()

    return result


def main():
    # GPU setup
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    results = []
    for name in MODEL_REGISTRY:
        r = benchmark_one(name)
        if r:
            results.append(r)

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY — All Models ({NUM_CLASSES} classes, batch={BATCH_SIZE})")
    print(f"{'='*80}")
    print(f"  {'Model':<15} {'Params':>10} {'Fwd (ms)':>10} {'Train (ms)':>12} {'Status':>8}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")
    for r in results:
        print(
            f"  {r['name']:<15} {r['params']:>10,} {r['fwd_ms']:>10.1f} "
            f"{r['train_ms']:>12.1f} {r['status']:>8}"
        )
    print()


if __name__ == "__main__":
    main()
