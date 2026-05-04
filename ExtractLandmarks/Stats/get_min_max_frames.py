import os
import cv2
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

path = "/run/media/aryan/PoopiDrive/projects_linux/microsoft asl citizen/ASL_Citizen/videos/"

# ── how many threads to use ──────────────────────────────────────────────────
# cv2.VideoCapture is I/O-bound and releases the GIL, so threads work great.
# Rule of thumb: 4–8× logical cores for network/disk I/O.
NUM_WORKERS = min(50, (os.cpu_count() or 4) * 4)
NUM_WORKERS = 128

def get_video_meta(vid_path: str):
    """Return (fps, frame_count) for a single video, or None on error."""
    try:
        cap = cv2.VideoCapture(vid_path)
        fps    = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return fps, frames
    except Exception:
        return None


def collect_video_paths(directory: str):
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".mp4")
    ]


def getallstats(label: str, data: list):
    print(f"\n── {label} ──")
    print(f"  Count           : {len(data)}")
    print(f"  Min             : {min(data):.4f}")
    print(f"  Max             : {max(data):.4f}")
    print(f"  Mean            : {statistics.mean(data):.4f}")
    print(f"  Median          : {statistics.median(data):.4f}")
    try:
        print(f"  Mode            : {statistics.mode(data):.4f}")
    except statistics.StatisticsError as e:
        print(f"  Mode            : error – {e}")
    print(f"  Std Dev (sample): {statistics.stdev(data):.4f}")
    print(f"  Std Dev (pop)   : {statistics.pstdev(data):.4f}")
    print(f"  Variance (samp) : {statistics.variance(data):.4f}")
    print(f"  Variance (pop)  : {statistics.pvariance(data):.4f}")
    print(f"  Harmonic Mean   : {statistics.harmonic_mean(data):.4f}")


def main():
    video_paths = collect_video_paths(path)
    total = len(video_paths)
    print(f"Found {total} .mp4 files – processing with {NUM_WORKERS} threads …")

    fps_list    = []
    frames_list = []
    errors      = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(get_video_meta, p): p for p in video_paths}
        with tqdm(total=total, unit="vid", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    errors += 1
                else:
                    fps, frames = result
                    fps_list.append(fps)
                    frames_list.append(frames)
                pbar.update(1)

    print(f"\nDone. Processed: {len(fps_list)}  Errors: {errors}")

    getallstats("FPS",    fps_list)
    getallstats("FRAMES", frames_list)


if __name__ == "__main__":
    main()
