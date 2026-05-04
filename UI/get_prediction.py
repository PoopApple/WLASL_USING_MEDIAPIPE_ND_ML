from datetime import datetime

import tensorflow as tf

import numpy as np

import os
import json

_INDEX_TO_WORD = {}

def load_prediction_model(model_path):
    """
    Load your .keras model here.
    This will be called ONCE by the InferenceWorker thread.
    """
    global _INDEX_TO_WORD
    
    model_dir = os.path.dirname(model_path)
    json_path = os.path.join(model_dir, "word_to_ind_all.json")
    
    try:
        with open(json_path, "r") as f:
            word_to_ind = json.load(f)
            _INDEX_TO_WORD = {v: k for k, v in word_to_ind.items()}
            print(len(_INDEX_TO_WORD))
    except Exception as e:
        print(f"Error loading JSON map: {e}")

    return tf.keras.models.load_model(model_path)


TOPK = 5


def _predict_arr_tta(model, ARR, mask):
    """
    Run inference on ARR (1, 128, 64, 4) with TTA (Test-Time Augmentation).
    Predicts on original + horizontally flipped, then averages the softmax outputs.
    
    Returns averaged predictions array of shape (1, num_classes).
    """
    from preprocess import flip_processed_arr

    # Original prediction
    preds_orig = model.predict({"input_data": ARR, "input_mask": mask}, verbose=0)

    # Flipped prediction
    arr_flipped = flip_processed_arr(ARR[0])  # (128, 64, 4)
    ARR_flip = np.expand_dims(arr_flipped, axis=0)
    preds_flip = model.predict({"input_data": ARR_flip, "input_mask": mask}, verbose=0)

    # Average softmax outputs
    return (preds_orig + preds_flip) / 2.0


def _top_k_from_predictions(predictions, include_confidence, print_label="", print_preds=False):
    """Rank top-k from averaged predictions array."""
    idx = np.argpartition(predictions, -TOPK, axis=1)[0, -TOPK:]
    idx = idx[np.argsort(predictions[0, idx])][::-1]
    values = predictions[0, idx]
    top_k_list = []
    for i, v in zip(idx, values):
        word = _INDEX_TO_WORD.get(int(i), "Unknown")
        if print_preds:
            print(f"  {print_label} {word}: {float(v)*100:.1f}%")
        if include_confidence:
            top_k_list.append(f"{word} ({float(v)*100:.1f}%)")
        else:
            top_k_list.append(word)
    return top_k_list


def run_inference(model, sequence, print_preds=False, include_confidence=False):
    # sequence is either:
    #   - a tuple (arr_padded, mask) of shape (128,64,4) and (128,) from normalise_lm_arr_temporally
    #   - a plain list of processed frames (legacy real_time without normalization)

    if isinstance(sequence, tuple) and len(sequence) == 2:
        arr_padded, predefined_mask = sequence

        def predict_slice(start, end):
            sliced_arr = arr_padded[start:end]
            sliced_mask = predefined_mask[start:end]
            n = end - start
            ARR = np.zeros(shape=(128, 64, 4), dtype=np.float32)
            mask = np.zeros(shape=128, dtype=bool)
            ARR[:n] = sliced_arr
            mask[:n] = sliced_mask
            ARR = np.expand_dims(ARR, axis=0)
            mask = np.expand_dims(mask, axis=0)
            try:
                predictions = _predict_arr_tta(model, ARR, mask)
                return _top_k_from_predictions(
                    predictions,
                    include_confidence,
                    print_label=f"[{start},{end-1}]",
                    print_preds=print_preds,
                )
            except Exception as e:
                print(f"Prediction Error [{start},{end-1}]: {e}")
                return []

        if print_preds:
            print(f"\n--- Prediction (TTA) ---")

        final_output = [
            predict_slice(0, 128),
            predict_slice(62, 128),
            predict_slice(0, 62),
            predict_slice(32, 94),
            predict_slice(19, 109),
        ]
        return json.dumps(final_output)

    # Legacy path: raw list of frames (should not normally be reached anymore)
    if print_preds:
        print(f"\n--- New Prediction Cycle ({len(sequence)} frames) ---")

    def predict_slice_legacy(start, end):
        slice_seq = sequence[start:end]
        slice_len = len(slice_seq)
        if slice_len == 0:
            return []
        mask = np.zeros(shape=128, dtype=bool)
        mask[:slice_len] = 1
        ARR = np.zeros(shape=(128, 64, 4), dtype=np.float32)
        ARR[:slice_len] = slice_seq
        ARR = np.expand_dims(ARR, axis=0)
        mask = np.expand_dims(mask, axis=0)
        try:
            predictions = _predict_arr_tta(model, ARR, mask)
            return _top_k_from_predictions(predictions, include_confidence,
                                           print_label=f"({start},{end-1})", print_preds=print_preds)
        except Exception as e:
            print(f"Prediction Error for ({start},{end-1}): {e}")
            return []

    final_output = [
        predict_slice_legacy(0, 128),
        predict_slice_legacy(62, 128),
        predict_slice_legacy(0, 62),
        predict_slice_legacy(32, 94),
        predict_slice_legacy(19, 109),
    ]
    return json.dumps(final_output)

