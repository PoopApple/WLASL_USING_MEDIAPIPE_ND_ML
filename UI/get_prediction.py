"""
get_prediction.py
=================
Mode-aware inference:

  real_time    — 1 model.predict() call, no TTA flip, <30ms target.
                 Returns {"mode": "real_time", "top": [[word, pct], ...]}

  video_testing — TTA (orig + flipped) on 3 temporal slices (full, first-half,
                  second-half).  Runs offline so speed is not a concern.
                  Returns {"mode": "video_testing", "slices": {label: [[word, pct], ...]}}
"""

import tensorflow as tf
import numpy as np
import os
import json

TARGET_FRAMES = 64   # Must match MAX_FRAMES in ModelTrain/model.py
TOPK          = 10

_INDEX_TO_WORD: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
def load_prediction_model(model_path: str):
    """
    Load a .keras model for inference only.

    compile=False skips restoring the optimizer/loss (avoids LabelSmoothingSparseCCE
    / AdamW deserialization errors).  The model is only used for forward-pass
    prediction so this is always safe.
    """
    global _INDEX_TO_WORD

    model_dir = os.path.dirname(model_path)
    # Try both possible JSON filenames
    for fname in ("word_to_ind_all.json", "word_to_ind.json", "word_to_ind_500.json"):
        json_path = os.path.join(model_dir, fname)
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    word_to_ind = json.load(f)
                _INDEX_TO_WORD = {v: k for k, v in word_to_ind.items()}
                print(f"[get_prediction] Loaded {len(_INDEX_TO_WORD)}-word map from {json_path}")
            except Exception as e:
                print(f"[get_prediction] Error loading label map: {e}")
            break
    else:
        print(f"[get_prediction] WARNING: no word_to_ind JSON found in {model_dir}")

    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"[get_prediction] Model loaded: {os.path.basename(model_path)}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
def _top_k_raw(predictions) -> list:
    """
    Return top-K predictions as [[word, confidence_pct], ...] sorted by confidence.
    """
    idx    = np.argpartition(predictions, -TOPK, axis=1)[0, -TOPK:]
    idx    = idx[np.argsort(predictions[0, idx])][::-1]
    return [
        [_INDEX_TO_WORD.get(int(i), f"cls_{i}"), round(float(predictions[0, i]) * 100, 1)]
        for i in idx
    ]


def _predict_one(model, ARR, mask):
    """Single forward pass.  ARR: (1,64,64,4), mask: (1,64) bool."""
    return model.predict({"input_data": ARR, "input_mask": mask}, verbose=0)


def _predict_tta(model, ARR, mask):
    """TTA: average of original + horizontally flipped."""
    from preprocess import flip_processed_arr
    preds_orig = _predict_one(model, ARR, mask)
    arr_flip   = np.expand_dims(flip_processed_arr(ARR[0]), 0)
    preds_flip = _predict_one(model, arr_flip, mask)
    return (preds_orig + preds_flip) / 2.0


def _make_slice_input(arr_padded, predefined_mask, start: int, end: int):
    """
    Slice [start:end] from (TARGET_FRAMES, 64, 4) and re-pad to full size.
    The mask for padded positions stays False.
    """
    n    = end - start
    ARR  = np.zeros((TARGET_FRAMES, 64, 4), dtype=np.float32)
    mask = np.zeros(TARGET_FRAMES, dtype=bool)
    ARR[:n]  = arr_padded[start:end]
    mask[:n] = predefined_mask[start:end]
    return np.expand_dims(ARR, 0), np.expand_dims(mask, 0)


# ─────────────────────────────────────────────────────────────────────────────
def run_inference(model, sequence, mode: str = "real_time", print_preds: bool = False) -> str | None:
    """
    Run inference and return a JSON string.

    Parameters
    ----------
    model    : loaded Keras model
    sequence : tuple of arrays/info, typically (arr_padded, mask, frame_start, frame_end)
    mode     : "real_time" | "video_testing"
    """
    if not (isinstance(sequence, tuple) and len(sequence) in (2, 4)):
        print("[get_prediction] run_inference: unexpected sequence format, skipping.")
        return None

    if len(sequence) == 4:
        arr_padded, predefined_mask, frame_start, frame_end = sequence
    else:
        arr_padded, predefined_mask = sequence
        frame_start, frame_end = None, None

    # ── Slices for both modes ─────────────────────────────────────────────────
    T, H = TARGET_FRAMES, TARGET_FRAMES // 2
    if mode == "real_time":
        slices = [("Full", 0, T)]
    else:
        slices = [
            ("Full",         0, T),
            ("First Half",   0, H),
            ("Second Half",  H, T),
        ]
        
    result_slices = {}
    if print_preds:
        print(f"\n--- Prediction ({mode}, {TARGET_FRAMES} frames) ---")

    for label, s, e in slices:
        ARR, mask = _make_slice_input(arr_padded, predefined_mask, s, e)
        
        # CuDNN doesn't support completely empty sequence masks (all False)
        if not np.any(mask):
            print(f"[get_prediction] Slice '{label}' has no active frames, skipping.")
            result_slices[label] = []
            continue
            
        try:
            if mode == "real_time":
                # No TTA for real-time to save latency
                preds = _predict_one(model, ARR, mask)
            else:
                # TTA for video testing
                preds = _predict_tta(model, ARR, mask)
        except Exception as ex:
            print(f"[get_prediction] Slice '{label}' error: {ex}")
            result_slices[label] = []
            continue
        top = _top_k_raw(preds)
        result_slices[label] = top
        
        # ALWAYS print the predictions to the console for monitoring
        print(f"\n--- Output ({label}) ---")
        for rank_idx, (w, p) in enumerate(top):
            print(f" #{rank_idx+1:<2} | {w:<15} | {p:.1f}%")

    res = {"mode": mode, "slices": result_slices}
    if frame_start is not None and frame_end is not None:
        res["frame_range"] = [int(frame_start), int(frame_end)]
    return json.dumps(res)
