"""
MotionTesting/motion_threshold_viz.py
======================================
Visualise motion-energy segmentation on raw .npy landmark files + their .mp4 videos.

Colour coding:
  🔵 Blue  border = frame is above start_thresh (motion detected)
  🟢 Green border = frame is inside the LONGEST consecutive motion window
  🔴 Red   border = frame is idle / redacted

Usage:
  python motion_threshold_viz.py \
    --videos_dir  /path/to/videos/ \
    --landmarks_dir /path/to/dataset3.0/landmarks/ \
    --n 30 \
    --start_thresh 0.015 \
    --end_thresh   0.008

Controls:
  Any key  → advance to next video
  q / ESC  → quit
"""

import os
import random
import sys

import cv2
import numpy as np


# ─────────────────────────────────────────────
# Motion energy (ported from UI/worker.py)
# ─────────────────────────────────────────────

# Landmark index groups within the full 75-point array
#   15, 16     = wrists (pose)
#   17-22      = pinky, index, thumb tips (pose)
#   33-53      = left hand joints (21 points)
#   54-74      = right hand joints (21 points)
KEY_INDICES = [15, 16, 17, 18, 19, 20, 21, 22] + list(range(33, 75))

# Positions of each hand group WITHIN the KEY_INDICES slice
# Left:  wrist(15), pose tips(17,19,21), + left hand joints(33-53)
_LEFT_POS  = [0, 2, 4] + list(range(8, 29))   # 24 points
# Right: wrist(16), pose tips(18,20,22), + right hand joints(54-74)
_RIGHT_POS = [1, 3, 5] + list(range(29, 50))  # 24 points

# Dedicated indices for finger tips (Pose + Hand models)
FINGER_TIP_INDICES = set([17, 18, 19, 20, 21, 22, 37, 41, 45, 49, 53, 58, 62, 66, 70, 74])

# Build precomputed weight mask (updated before each call via _make_weights)
def _make_weights(tip_weight):
    w = np.ones(len(KEY_INDICES), dtype=np.float32)
    if tip_weight != 1.0:
        for j, idx in enumerate(KEY_INDICES):
            if idx in FINGER_TIP_INDICES:
                w[j] = tip_weight
    return w


def _hand_present(frame, hand_slice, vis_thresh=0.05):
    """Return True if at least one landmark in the hand slice has visibility > vis_thresh.
    Checks both the dedicated hand model AND pose tips for that hand side.
    """
    return float(np.max(frame[hand_slice, 3])) > vis_thresh


def compute_motion_per_frame(frames_raw, method="max_hand", top_k=5, tip_weight=1.0,
                              vis_thresh=0.05):
    """
    Given raw frame data shape (T, 75, 4), return per-frame motion energy array (T,).

    Methods:
      'mean'     : Average displacement of all key landmarks.
      'rms'      : Root Mean Square -- magnifies large displacements.
      'top_k'    : Average of the K most active points.
      'max_hand' : Compute energy per hand separately, take the MAX.
                   Best choice for single-hand signs.

    tip_weight : Extra multiplier for finger tip landmarks.
    vis_thresh : If a hand's max landmark visibility is below this, the hand is
                 considered absent. Absent-hand landmarks are excluded from energy,
                 preventing zero-jump artifacts when hands enter/exit the frame.
    """
    weights = _make_weights(tip_weight)
    energies = np.zeros(len(frames_raw), dtype=np.float32)
    prev_frame = None
    for i, frame in enumerate(frames_raw):
        if prev_frame is None:
            prev_frame = frame
            continue

        # ── Determine hand presence in BOTH frames of the transition ──
        # We need a hand to be present in BOTH frames to compute valid displacement.
        # If it was absent in the previous frame (zeros) but present now, the jump
        # is from 0→real which is not real motion.
        left_present  = (_hand_present(frame,       slice(33, 54), vis_thresh) and
                         _hand_present(prev_frame,  slice(33, 54), vis_thresh))
        right_present = (_hand_present(frame,       slice(54, 75), vis_thresh) and
                         _hand_present(prev_frame,  slice(54, 75), vis_thresh))

        if not left_present and not right_present:
            # No hands in either frame: no meaningful motion, avoid zero-jump artifact
            energies[i] = 0.0
            prev_frame = frame
            continue

        curr_pos = frame[KEY_INDICES, :2]
        prev_pos = prev_frame[KEY_INDICES, :2]

        # Per-landmark displacement, scaled by tip weights
        diffs = np.linalg.norm(curr_pos - prev_pos, axis=1) * weights

        # Zero out absent-hand landmarks so they don't contaminate the score
        if not left_present:
            diffs[_LEFT_POS] = 0.0
        if not right_present:
            diffs[_RIGHT_POS] = 0.0

        if method == "mean":
            energy = float(np.mean(diffs))
        elif method == "rms":
            energy = float(np.sqrt(np.mean(np.square(diffs))))
        elif method == "max_hand":
            # Score each hand independently; take the winner
            left_e  = float(np.mean(diffs[_LEFT_POS]))  if left_present  else 0.0
            right_e = float(np.mean(diffs[_RIGHT_POS])) if right_present else 0.0
            energy  = max(left_e, right_e)
        else:  # top_k
            top_diffs = np.sort(diffs)[-top_k:]
            energy = float(np.mean(top_diffs))

        energies[i] = energy
        prev_frame = frame
    return energies


def trim_tail_armsweep(frames_raw, seg_end, wrist_drop_thresh=0.015, lookback=10):
    """
    Walk backwards from seg_end and drop frames where EITHER wrist is
    moving sharply downward (y increasing fast) -- the camera-off arm sweep.
    Uses np.any() so a single-arm shutoff is also caught.
    If the video cuts off automatically (no arm drop), returns seg_end unchanged.
    Returns a new (possibly earlier) end index.
    """
    end = seg_end
    for i in range(seg_end - 1, max(seg_end - lookback - 1, 0), -1):
        curr_y = frames_raw[i,     [15, 16], 1]
        prev_y = frames_raw[i - 1, [15, 16], 1]
        drop   = curr_y - prev_y   # positive = moving down in image
        if np.any(drop > wrist_drop_thresh):   # either wrist drops
            end = i
        else:
            break
    return end


def find_motion_segments(energies, start_thresh, end_thresh, cooldown_frames, min_sign_frames):
    """
    Run the IDLE → SIGNING → COOLDOWN state machine (same as UI/worker.py).
    Returns a list of (start_idx, end_idx) tuples for each fired segment.
    """
    segments = []
    state = "idle"
    sign_start = 0
    cooldown_count = 0

    for i, e in enumerate(energies):
        if state == "idle":
            if e >= start_thresh:
                state = "signing"
                sign_start = i
                cooldown_count = 0

        elif state == "signing":
            if e < end_thresh:
                state = "cooldown"
                cooldown_count = 1
            elif i - sign_start >= 128:
                # buffer full → fire
                segments.append((sign_start, i))
                state = "idle"
                sign_start = 0

        elif state == "cooldown":
            if e >= start_thresh:
                # resumed signing
                state = "signing"
                cooldown_count = 0
            else:
                cooldown_count += 1
                if cooldown_count >= cooldown_frames:
                    # fire
                    end_idx = i - cooldown_count + 1
                    if end_idx - sign_start >= min_sign_frames:
                        segments.append((sign_start, end_idx))
                    state = "idle"
                    sign_start = 0
                    cooldown_count = 0

    # Handle segment still open at end of video
    if state in ("signing", "cooldown") and i - sign_start >= min_sign_frames:
        segments.append((sign_start, i + 1))

    return segments


def find_longest_segment(segments):
    """Return the longest (start, end) segment, or None."""
    if not segments:
        return None
    return max(segments, key=lambda s: s[1] - s[0])


def load_npy(npy_path):
    """
    Load raw .npy file.
    Shape: (total_frames+1, 75, 4) — first row is metadata [total_frames, fps, ...].
    Returns (frames_raw, fps) where frames_raw has shape (T, 75, 4).
    """
    arr = np.load(npy_path)
    total_frames = int(arr[0, 0, 0])
    fps = float(arr[0, 1, 0])
    frames_raw = arr[1:total_frames + 1]   # shape (T, 75, 4)
    return frames_raw, fps, total_frames


# ─────────────────────────────────────────────
# Pair matching: video ↔ npy
# ─────────────────────────────────────────────

def collect_pairs(videos_dir, landmarks_dir):
    """
    Find all (video_path, npy_path) pairs.

    Video filename:    WORD_fileid-WORD LABEL.mp4
    NPY path:          landmarks/WORD/WORD_fileid-WORD LABEL.npy

    Strategy: build a stem→npy_path dict from all .npy files,
    then match each .mp4 by its stem.
    """
    stem_to_npy = {}
    for word in os.listdir(landmarks_dir):
        word_dir = os.path.join(landmarks_dir, word)
        if not os.path.isdir(word_dir):
            continue
        for fname in os.listdir(word_dir):
            if fname.endswith(".npy"):
                stem = fname[:-4]
                stem_to_npy[stem] = os.path.join(word_dir, fname)

    pairs = []
    for fname in os.listdir(videos_dir):
        if not fname.lower().endswith(".mp4"):
            continue
        stem = fname[:-4]
        if stem in stem_to_npy:
            pairs.append((os.path.join(videos_dir, fname), stem_to_npy[stem]))

    return pairs


# ─────────────────────────────────────────────
# Overlay helpers
# ─────────────────────────────────────────────

COLOR_BLUE   = (255,  80,  20)   # BGR — blue border (motion detected)
COLOR_GREEN  = ( 30, 220,  30)   # BGR — green border (longest segment)
COLOR_RED    = ( 20,  20, 220)   # BGR — red border (idle / redacted)
COLOR_YELLOW = (  0, 215, 255)   # BGR — yellow border (arm-sweep / camera-off)
BORDER_THICKNESS = 18            # px


def detect_armsweep_mask(frames_raw, wrist_drop_thresh=0.015):
    """
    Return a boolean array (T,) where True means EITHER wrist is dropping
    sharply downward on that frame -- the camera-off arm sweep.
    """
    mask = np.zeros(len(frames_raw), dtype=bool)
    for i in range(1, len(frames_raw)):
        curr_y = frames_raw[i,     [15, 16], 1]
        prev_y = frames_raw[i - 1, [15, 16], 1]
        drop   = curr_y - prev_y
        if np.any(drop > wrist_drop_thresh):
            mask[i] = True
    return mask

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(frame, border_color, energy, frame_idx, total_frames,
                 state_label, segment_info, longest_seg, method):
    """Draw the border + HUD text on a copy of frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    t = BORDER_THICKNESS
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), border_color, t * 2)

    # HUD lines
    lines = [
        f"Frame: {frame_idx}/{total_frames}",
        f"Energy: {energy:.4f}",
        f"State: {state_label}",
        f"Method: {method}",
    ]
    if longest_seg:
        ls, le = longest_seg
        lines.append(f"Longest: [{ls}:{le}] ({le-ls} frames)")
    if segment_info:
        lines.append(f"Segments: {len(segment_info)}")

    y = 30
    for line in lines:
        cv2.putText(out, line, (10, y), FONT, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), FONT, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26

    return out


# ─────────────────────────────────────────────
# Main visualisation loop
# ─────────────────────────────────────────────

def visualise_video(video_path, npy_path, start_thresh, end_thresh,
                    cooldown_frames, min_sign_frames, method="top_k", top_k=5, 
                    tip_weight=1.0, head_pad=5, window_name="Motion Viz"):
    """Show one video with motion-energy colour overlay. Returns (continue, speed_factor)."""

    # ── Load .npy ──
    try:
        frames_raw, npy_fps, total_npy_frames = load_npy(npy_path)
    except Exception as e:
        print(f"[SKIP] Cannot load npy {npy_path}: {e}")
        return True

    # ── Compute energies & segments offline ──
    energies = compute_motion_per_frame(frames_raw, method=method, top_k=top_k, tip_weight=tip_weight)

    segments = find_motion_segments(
        energies, start_thresh, end_thresh, cooldown_frames, min_sign_frames
    )
    longest_seg = find_longest_segment(segments)

    # "Tail" = frames after the original longest segment end.
    # Yellow will ONLY appear here, not mid-sign.
    tail_start = longest_seg[1] if longest_seg else len(frames_raw)

    # ── Compute final extraction segment ──
    # Walk forward inside the longest segment to find the first frame where
    # hands are actually present, then pad back by HEAD_PAD frames.
    if longest_seg:
        ls, le = longest_seg
        first_hand_frame = ls   # default: segment start
        for fi in range(ls, le):
            if _hand_present(frames_raw[fi], slice(33, 54)) or \
               _hand_present(frames_raw[fi], slice(54, 75)):
                first_hand_frame = fi
                break
        final_start = max(0, first_hand_frame - head_pad)
        final_seg   = (final_start, le)
    else:
        final_seg = None

    # Precompute per-frame armsweep detection mask (wrist drop in tail only)
    armsweep_mask = detect_armsweep_mask(frames_raw)

    # Build per-frame label: 'idle' | 'motion' | 'final_keep'
    labels = np.full(len(frames_raw), "idle", dtype=object)
    for (s, e) in segments:
        labels[s:e] = "motion"
    if final_seg:                          # green = FINAL extraction range
        fs, fe = final_seg
        labels[fs:fe] = "longest"

    # ── Open video ──
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[SKIP] Cannot open video {video_path}")
        return True

    vid_fps   = cap.get(cv2.CAP_PROP_FPS) or npy_fps or 30
    vid_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Persistent speed factor for this video session
    speed_factor = 1.0

    print(f"\n{'='*60}")
    print(f"Video : {os.path.basename(video_path)}")
    print(f"NPY   : {os.path.basename(npy_path)}")
    print(f"NFrame: vid={vid_total}  npy={total_npy_frames}")
    print(f"Segs  : {segments}")
    print(f"Longest segment  : {longest_seg}")
    print(f"Final extraction : {final_seg}  (head_pad={head_pad})")
    print(f"start_thresh={start_thresh}  end_thresh={end_thresh}")
    print("Controls:")
    print("  [ / ]     → Slow down / Speed up")
    print("  Any other → Next video")
    print("  q / ESC   → Quit")

    vid_name = os.path.basename(video_path)
    frame_idx = 0
    while True:
        # Calculate delay based on current speed factor
        delay_ms = max(1, int(1000 / (vid_fps * speed_factor)))
        
        ret, frame = cap.read()
        if not ret:
            # Loop back for easy re-inspection
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            continue

        # Map video frame_idx → npy frame_idx (they may differ in count)
        npy_idx = min(frame_idx, len(frames_raw) - 1)
        energy  = energies[npy_idx]
        label   = labels[npy_idx]

        # Detect if both hands are absent (visibility near zero → ghost zeros)
        left_vis  = float(np.mean(frames_raw[npy_idx, 33:54, 3]))
        right_vis = float(np.mean(frames_raw[npy_idx, 54:75, 3]))
        no_hands  = left_vis < 0.05 and right_vis < 0.05

        # Arm-sweep only counts if we are PAST the end of the longest segment
        in_tail     = npy_idx >= tail_start
        is_armsweep = in_tail and bool(armsweep_mask[npy_idx])

        if label == "longest":
            border_color = COLOR_GREEN
            state_label  = "KEEP (longest)"
        elif label == "motion":
            border_color = COLOR_BLUE
            state_label  = "MOTION"
        else:
            border_color = COLOR_RED
            state_label  = "IDLE (redact)"

        # Override with yellow ONLY in the tail after the segment
        if is_armsweep:
            border_color = COLOR_YELLOW
            state_label  = "ARM-SWEEP (camera-off)"

        hand_str = f"L:{left_vis:.2f} R:{right_vis:.2f}" + (" !! NO HANDS" if no_hands else "")

        # Prepare HUD lines
        hud_lines = [
            vid_name[:42],
            f"Frame: {frame_idx}/{vid_total}",
            f"Energy: {energy:.4f}",
            f"State: {state_label}",
            f"Hands: {hand_str}",
            f"Method: {method}  TipW:{tip_weight:.1f}x  Spd:{speed_factor:.1f}x",
        ]

        # Draw overlay
        out = frame.copy()
        h_orig, w_orig = out.shape[:2]
        t = BORDER_THICKNESS
        cv2.rectangle(out, (0, 0), (w_orig - 1, h_orig - 1), border_color, t * 2)

        y = 30
        for line in hud_lines:
            cv2.putText(out, line, (10, y), FONT, 0.60, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, line, (10, y), FONT, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
            y += 24
            
        if final_seg:
            fs, fe = final_seg
            f_line = f"FINAL: [{fs}:{fe}] ({fe-fs} f)  (pad={head_pad})"
            cv2.putText(out, f_line, (10, y), FONT, 0.65, (0, 150, 0), 3, cv2.LINE_AA)
            cv2.putText(out, f_line, (10, y), FONT, 0.65, (0, 255, 80), 1, cv2.LINE_AA)

        # Resize to fit screen comfortably
        h, w = out.shape[:2]
        max_w, max_h = 900, 600
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            out = cv2.resize(out, (int(w * scale), int(h * scale)))

        cv2.imshow(window_name, out)
        key = cv2.waitKey(delay_ms) & 0xFF

        if key == ord("q") or key == 27:   # ESC
            cap.release()
            return False, speed_factor      # signal caller to quit
        elif key == ord("["):               # Slow down
            speed_factor = max(0.1, speed_factor - 0.1)
        elif key == ord("]"):               # Speed up
            speed_factor = min(5.0, speed_factor + 0.1)
        elif key != 255:                    # any other key → next video
            break

        frame_idx += 1

    cap.release()
    return True, speed_factor   # continue


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main(videos_dir, landmarks_dir, n, start_thresh, end_thresh, cooldown, min_frames, method, top_k, tip_weight, head_pad, seed):
    if not os.path.isdir(videos_dir):
        print(f"[ERROR] videos_dir not found: {videos_dir}")
        sys.exit(1)
    if not os.path.isdir(landmarks_dir):
        print(f"[ERROR] landmarks_dir not found: {landmarks_dir}")
        sys.exit(1)

    print("Collecting video/npy pairs...")
    pairs = collect_pairs(videos_dir, landmarks_dir)
    print(f"  Found {len(pairs)} matched pairs.")

    if not pairs:
        print("[ERROR] No matching pairs found. Check that filenames match between videos and npy dirs.")
        sys.exit(1)

    random.seed(seed)
    sample = random.sample(pairs, min(n, len(pairs)))
    print(f"  Sampled {len(sample)} pairs to visualise.")
    print(f"\nThresholds: start={start_thresh}  end={end_thresh}  "
          f"cooldown={cooldown}  min_frames={min_frames}  method={method}  "
          f"top_k={top_k}  tip_weight={tip_weight}  head_pad={head_pad}\n")

    window_name = "Motion Viz — q=quit, any key=next"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for i, (vid_path, npy_path) in enumerate(sample):
        print(f"[{i+1}/{len(sample)}]", end=" ")
        keep_going, _ = visualise_video(
            vid_path, npy_path,
            start_thresh, end_thresh,
            cooldown, min_frames,
            method=method,
            top_k=top_k,
            tip_weight=tip_weight,
            head_pad=head_pad,
            window_name=window_name
        )
        if not keep_going:
            print("Quitting.")
            break

    cv2.destroyAllWindows()
    print("\nDone. Adjust START_THRESH / END_THRESH / TOP_K / TIP_WEIGHT based on what you saw.")


if __name__ == "__main__":
    # ── CONFIGURATION ──
    VIDEOS_DIR     = "/run/media/aryan/PoopiDrive/projects_linux/microsoft asl citizen/ASL_Citizen/videos/"
    LANDMARKS_DIR  = "/run/media/aryan/PoopiDrive/projects_linux/microsoft asl citizen/landmarks/"
    
    N_SAMPLES      = 500
    START_THRESH   = 0.013
    END_THRESH     = 0.008
    COOLDOWN       = 15
    MIN_FRAMES     = 10
    
    METHOD         = "max_hand"  # "mean", "rms", "top_k", or "max_hand"
    TOP_K          = 12          # Only used if METHOD is "top_k"
    TIP_WEIGHT     = 1.5         # Weight multiplier for finger tips (1.0 = no boost)
    HEAD_PAD       = 5           # Frames before first hand detection to include in final extraction
    
    SEED           = 42
    # ──────────────────

    main(
        VIDEOS_DIR, 
        LANDMARKS_DIR, 
        N_SAMPLES, 
        START_THRESH, 
        END_THRESH, 
        COOLDOWN, 
        MIN_FRAMES, 
        METHOD,
        TOP_K,
        TIP_WEIGHT,
        HEAD_PAD,
        SEED
    )
