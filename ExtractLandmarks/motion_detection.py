"""
motion_detection.py
===================
Reusable ASL motion-energy segmentation module.

Usage:
    from MotionTesting.motion_detection import get_final_extraction, DEFAULTS

    frames_raw = np.load("some.npy")[1:]   # skip metadata row
    result = get_final_extraction(frames_raw)
    if result:
        start, end = result          # slice frames_raw[start:end] for normalisation
"""

import numpy as np

# ─────────────────────────────────────────────
# Default tuned parameters (finalised 2026-05-04)
# ─────────────────────────────────────────────
DEFAULTS = dict(
    start_thresh = 0.013,   # energy to enter SIGNING state
    end_thresh   = 0.008,   # energy to stay in SIGNING (below → COOLDOWN)
    cooldown     = 15,      # frames of low energy before segment fires
    min_frames   = 10,      # minimum segment length to keep
    method       = "max_hand",  # "mean" | "rms" | "top_k" | "max_hand"
    top_k        = 12,      # only used when method="top_k"
    tip_weight   = 1.5,     # extra weight for finger tip landmarks
    head_pad     = 5,       # frames before first hand detection to include
    vis_thresh   = 0.05,    # min visibility to count a hand as present
)

# ─────────────────────────────────────────────
# Landmark index groups (within the 75-point raw array)
#   0-32   : Pose (33 landmarks)
#   33-53  : Left Hand (21 landmarks)
#   54-74  : Right Hand (21 landmarks)
# ─────────────────────────────────────────────

# Landmarks used in energy calculation
KEY_INDICES = [15, 16, 17, 18, 19, 20, 21, 22] + list(range(33, 75))

# Positions of each hand group WITHIN the KEY_INDICES slice
# Left:  wrist(15), pose tips(17,19,21), left hand joints(33-53) → indices 0,2,4,8..28
_LEFT_POS  = [0, 2, 4] + list(range(8, 29))   # 24 points
# Right: wrist(16), pose tips(18,20,22), right hand joints(54-74) → indices 1,3,5,29..49
_RIGHT_POS = [1, 3, 5] + list(range(29, 50))  # 24 points

# Finger tip landmark indices (raw array coords) for extra weighting
FINGER_TIP_INDICES = set([17, 18, 19, 20, 21, 22, 37, 41, 45, 49, 53, 58, 62, 66, 70, 74])


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _make_weights(tip_weight: float) -> np.ndarray:
    """Build a per-landmark weight vector for KEY_INDICES."""
    w = np.ones(len(KEY_INDICES), dtype=np.float32)
    if tip_weight != 1.0:
        for j, idx in enumerate(KEY_INDICES):
            if idx in FINGER_TIP_INDICES:
                w[j] = tip_weight
    return w


def _hand_present(frame: np.ndarray, hand_slice: slice, vis_thresh: float = 0.05) -> bool:
    """Return True if ANY landmark in hand_slice has visibility > vis_thresh."""
    return float(np.max(frame[hand_slice, 3])) > vis_thresh


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def compute_motion_per_frame(
    frames_raw: np.ndarray,
    method: str = "max_hand",
    top_k: int = 12,
    tip_weight: float = 1.5,
    vis_thresh: float = 0.05,
) -> np.ndarray:
    """
    Compute per-frame motion energy for a raw landmark array.

    Parameters
    ----------
    frames_raw : np.ndarray, shape (T, 75, 4)
        Raw landmark array (metadata row already stripped).
    method : str
        'mean'     — average displacement of all key landmarks.
        'rms'      — root mean square displacement.
        'top_k'    — mean of the K most active points.
        'max_hand' — compute per hand independently, return the MAX.
                     Best for single-hand signs (idle hand doesn't dilute).
    top_k : int
        Only used when method='top_k'.
    tip_weight : float
        Multiplier for finger-tip landmark displacements.
    vis_thresh : float
        Minimum visibility to count a hand as present for a given frame.
        If a hand is absent in either the current or previous frame, its
        landmarks are excluded to prevent zero-jump artefacts.

    Returns
    -------
    energies : np.ndarray, shape (T,)
        Per-frame motion energy.
    """
    weights    = _make_weights(tip_weight)
    energies   = np.zeros(len(frames_raw), dtype=np.float32)
    prev_frame = None

    for i, frame in enumerate(frames_raw):
        if prev_frame is None:
            prev_frame = frame
            continue

        # A hand must be visible in BOTH frames for a valid displacement.
        left_present  = (_hand_present(frame,      slice(33, 54), vis_thresh) and
                         _hand_present(prev_frame, slice(33, 54), vis_thresh))
        right_present = (_hand_present(frame,      slice(54, 75), vis_thresh) and
                         _hand_present(prev_frame, slice(54, 75), vis_thresh))

        if not left_present and not right_present:
            # No hands → no meaningful motion; avoid zero-jump artefact
            prev_frame = frame
            continue

        curr_pos = frame[KEY_INDICES, :2]
        prev_pos = prev_frame[KEY_INDICES, :2]
        diffs    = np.linalg.norm(curr_pos - prev_pos, axis=1) * weights

        # Zero out absent-hand contributions
        if not left_present:
            diffs[_LEFT_POS] = 0.0
        if not right_present:
            diffs[_RIGHT_POS] = 0.0

        if method == "mean":
            energy = float(np.mean(diffs))
        elif method == "rms":
            energy = float(np.sqrt(np.mean(np.square(diffs))))
        elif method == "max_hand":
            left_e  = float(np.mean(diffs[_LEFT_POS]))  if left_present  else 0.0
            right_e = float(np.mean(diffs[_RIGHT_POS])) if right_present else 0.0
            energy  = max(left_e, right_e)
        else:  # top_k
            energy = float(np.mean(np.sort(diffs)[-top_k:]))

        energies[i] = energy
        prev_frame   = frame

    return energies


def find_motion_segments(
    energies: np.ndarray,
    start_thresh: float,
    end_thresh: float,
    cooldown_frames: int,
    min_sign_frames: int,
) -> list:
    """
    IDLE → SIGNING → COOLDOWN state machine.

    Returns a list of (start_idx, end_idx) tuples.
    """
    segments      = []
    state         = "idle"
    sign_start    = 0
    cooldown_count = 0
    i             = 0

    for i, e in enumerate(energies):
        if state == "idle":
            if e >= start_thresh:
                state      = "signing"
                sign_start = i

        elif state == "signing":
            if e < end_thresh:
                state         = "cooldown"
                cooldown_count = 1
            # else: still signing, do nothing

        elif state == "cooldown":
            if e >= start_thresh:
                state         = "signing"
                cooldown_count = 0
            else:
                cooldown_count += 1
                if cooldown_count >= cooldown_frames:
                    end_idx = i - cooldown_count + 1
                    if end_idx - sign_start >= min_sign_frames:
                        segments.append((sign_start, end_idx))
                    state         = "idle"
                    sign_start    = 0
                    cooldown_count = 0

    # Handle open segment at end of video
    if state in ("signing", "cooldown") and i - sign_start >= min_sign_frames:
        segments.append((sign_start, i + 1))

    return segments


def find_longest_segment(segments: list):
    """Return the longest (start, end) tuple or None."""
    if not segments:
        return None
    return max(segments, key=lambda s: s[1] - s[0])


def get_final_extraction(
    frames_raw: np.ndarray,
    start_thresh: float  = DEFAULTS["start_thresh"],
    end_thresh: float    = DEFAULTS["end_thresh"],
    cooldown: int        = DEFAULTS["cooldown"],
    min_frames: int      = DEFAULTS["min_frames"],
    method: str          = DEFAULTS["method"],
    top_k: int           = DEFAULTS["top_k"],
    tip_weight: float    = DEFAULTS["tip_weight"],
    head_pad: int        = DEFAULTS["head_pad"],
    vis_thresh: float    = DEFAULTS["vis_thresh"],
):
    """
    Full pipeline: energy → segments → longest → head-padded final range.

    Parameters
    ----------
    frames_raw : np.ndarray, shape (T, 75, 4)
        Raw landmark frames (metadata row already stripped).
    All other params: see DEFAULTS.

    Returns
    -------
    (start, end) : tuple[int, int] | None
        The final frame range to extract, or None if no valid segment found.
        Use frames_raw[start:end] for downstream normalisation.
    """
    energies    = compute_motion_per_frame(frames_raw, method, top_k, tip_weight, vis_thresh)
    segments    = find_motion_segments(energies, start_thresh, end_thresh, cooldown, min_frames)
    longest_seg = find_longest_segment(segments)

    if longest_seg is None:
        return None

    ls, le = longest_seg

    # Find the first frame inside the segment where hands are actually present,
    # then step back head_pad frames to capture hand setup.
    first_hand_frame = ls
    for fi in range(ls, le):
        if (_hand_present(frames_raw[fi], slice(33, 54), vis_thresh) or
                _hand_present(frames_raw[fi], slice(54, 75), vis_thresh)):
            first_hand_frame = fi
            break

    final_start = max(0, first_hand_frame - head_pad)
    return (final_start, le)
