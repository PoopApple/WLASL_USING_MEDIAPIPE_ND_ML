"""
main.py
=======
Application controller for the ASL Recognition UI.

Connects VideoWorker signals to the UI and renders structured prediction
results in mode-appropriate formats.

Real-time display
─────────────────
  Prominent ranked table: rank | word | confidence bar.
  Updates every time a prediction arrives (<30ms expected latency).

Video-testing display
─────────────────────
  3-column table (Full / First Half / Second Half) with top-5 per column.
  Ground-truth word (parsed from filename) highlighted in green.

Motion energy bar
─────────────────
  Horizontal chart showing per-frame motion energy with:
    - Blue fill for energy levels
    - Green vertical markers for segment boundaries
    - Orange dashed threshold line
  Rendered as a QPixmap via numpy/cv2.
"""

import sys
import json
import cv2
import os
import random

import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from ui_main import MainUI, SettingsDialog
from worker import VideoWorker


# ── Confidence bar helpers ────────────────────────────────────────────────────
def _conf_bar_html(pct: float, width: int = 80) -> str:
    """Inline HTML mini-bar for the confidence column."""
    filled = int(pct / 100 * width)
    color  = "#13cf38" if pct >= 60 else "#e0a016" if pct >= 30 else "#555"
    return (
        f"<span style='display:inline-block;width:{filled}px;height:10px;"
        f"background:{color};border-radius:3px;'></span>"
        f"<span style='margin-left:4px;font-size:11px;color:#aaa;'>{pct:.1f}%</span>"
    )


def _render_rt_table(top: list, frame_range=None) -> str:
    """
    Render a styled HTML table for real-time top-5 predictions.
    top: [[word, confidence_pct], ...]
    """
    range_text = f"<div style='color:#888; font-size:12px; margin-bottom: 4px; text-align: center;'>Frames Analyzed: {frame_range[0]} - {frame_range[1]}</div>" if frame_range else ""
    html = (
        f"{range_text}"
        "<table width='100%' cellspacing='0' cellpadding='8' "
        "style='border-collapse:collapse;font-family:sans-serif;'>"
    )
    for i, (word, pct) in enumerate(top[:5]):
        bg   = "#1e1e30"
        size = "14px"
        html += (
            f"<tr style='background:{bg};'>"
            f"<td style='color:#f0f0f0;font-size:{size};font-weight:"
            f"normal;padding-left:12px;'>{word}</td>"
            f"<td align='right' style='padding-right:8px;'>{_conf_bar_html(pct)}</td>"
            f"</tr>"
        )
    html += "</table>"
    return html


def _render_video_table(slices: dict, ground_truth: str | None, frame_range=None) -> str:
    """
    Render a 3-column prediction table for video testing.
    slices: {"Full": [[word,pct],...], "First Half": [...], "Second Half": [...]}
    """
    range_text = f"<div style='color:#888; font-size:12px; margin-bottom: 4px; text-align: center;'>Frames Analyzed: {frame_range[0]} - {frame_range[1]}</div>" if frame_range else ""
    slice_keys = ["Full", "First Half", "Second Half"]
    html = (
        "<div align='center'>"
        f"{range_text}"
        "<table width='100%' border='0' cellspacing='4' cellpadding='6' "
        "style='border-collapse:separate;font-family:sans-serif;'>"
        "<tr style='background:#333;color:#fff;'>"
    )
    for key in slice_keys:
        html += f"<th style='border-radius:4px 4px 0 0;padding:8px;'>{key}</th>"
    html += "</tr>"

    max_rows = max((len(slices.get(k, [])) for k in slice_keys), default=0)
    for row_i in range(max(max_rows, 1)):
        html += "<tr>"
        for key in slice_keys:
            items = slices.get(key, [])
            bg    = "#1e2a1e"
            if row_i < len(items):
                word, pct = items[row_i]
                is_gt = (ground_truth and word.lower().strip() == ground_truth)
                bg    = "#1e4a1e" if is_gt else ("#2a2a2a" if row_i % 2 else "#242424")
                gt_mark = " ✓" if is_gt else ""
                html += (
                    f"<td style='background:{bg};color:#eee;border-radius:3px;"
                    f"padding:6px 8px;'>"
                    f"<b>{word}{gt_mark}</b><br>"
                    f"<span style='color:#aaa;font-size:11px;'>{pct:.1f}%</span>"
                    f"</td>"
                )
            else:
                html += f"<td style='background:{bg};'></td>"
        html += "</tr>"

    html += "</table></div>"
    return html


def _render_motion_energy_pixmap(data: dict, width: int = 700, height: int = 60) -> QPixmap | None:
    """
    Draw a motion energy chart as a QPixmap.
    data keys: energies (list), segments (list of [s,e]), best_seg ([s,e]|None),
               total_frames (int), start_thresh (float).
    """
    try:
        energies     = np.array(data["energies"], dtype=np.float32)
        segments     = data.get("segments", [])
        best_seg     = data.get("best_seg")
        total_frames = max(data.get("total_frames", len(energies)), 1)
        start_thresh = data.get("start_thresh", 0.013)

        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (28, 28, 28)

        max_e = max(float(np.max(energies)), 1e-6)

        # Draw energy bars
        for i, e in enumerate(energies):
            x1 = int(i / total_frames * width)
            x2 = max(x1 + 1, int((i + 1) / total_frames * width))
            bh = int(float(e) / max_e * (height - 4))
            cv2.rectangle(img, (x1, height - bh), (x2, height - 2), (80, 160, 240), -1)

        # Threshold line (orange dashed)
        ty = height - max(1, int(start_thresh / max_e * (height - 4)))
        for x in range(0, width, 8):
            cv2.line(img, (x, ty), (min(x + 4, width), ty), (224, 160, 22), 1)

        # All segments (semi-transparent green border)
        for seg in segments:
            s, e = seg if isinstance(seg, (list, tuple)) else (seg[0], seg[1])
            x1   = int(s / total_frames * width)
            x2   = int(e / total_frames * width)
            cv2.rectangle(img, (x1, 2), (x2, height - 2), (40, 160, 40), 1)

        # Best segment (bright green)
        if best_seg:
            s, e = best_seg
            x1   = int(s / total_frames * width)
            x2   = int(e / total_frames * width)
            cv2.rectangle(img, (x1, 2), (x2, height - 2), (19, 207, 56), 2)

        # Convert to QPixmap
        rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg     = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)
    except Exception as ex:
        print(f"[main] motion energy render error: {ex}")
        return None


def load_config():
    with open("config.json") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
class App(MainUI):
    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.worker = VideoWorker(self.config["camera_index"], self.config)

        # Navigation
        self.nav_video_btn.clicked.connect(lambda: self.switch_mode(1))
        self.nav_realtime_btn.clicked.connect(lambda: self.switch_mode(2))
        self.nav_settings_btn.clicked.connect(self.open_settings)

        # Video Testing controls
        self.video_back_btn.clicked.connect(lambda: self.switch_mode(0))
        self.video_choose_btn.clicked.connect(self.choose_video)
        self.video_random_btn.clicked.connect(self.random_video)

        # Real-Time controls
        self.rt_back_btn.clicked.connect(lambda: self.switch_mode(0))
        self.rt_camera_select.currentTextChanged.connect(self.on_camera_changed)
        self.rt_start_btn.clicked.connect(self.start)
        self.rt_stop_btn.clicked.connect(self.stop)

        # Initial camera
        self.rt_camera_select.setCurrentText(f"Camera {self.config['camera_index']}")

        # Worker signals
        self.worker.frame_signal.connect(self.update_frame)
        self.worker.text_signal.connect(self.update_text)
        self.worker.motion_energy_signal.connect(self.update_motion_energy)
        self.worker.start()

    # ── Mode switching ────────────────────────────────────────────────────────
    def switch_mode(self, index: int):
        self.stacked_widget.setCurrentIndex(index)
        self.stop()
        if index == 1:
            self.worker.set_mode("video_testing")
            self.video_name_label.setText("No video selected")
            self.motion_energy_label.setText("Run a video to see motion energy")
            self.worker.change_source(None)
        elif index == 2:
            self.worker.set_mode("real_time")
            cam_idx = self.config.get("camera_index", 0)
            self.rt_camera_select.setCurrentText(f"Camera {cam_idx}")
            self.rt_name_label.setText(f"Camera {cam_idx}")
            self.worker.change_source(cam_idx)

    # ── Video file controls ───────────────────────────────────────────────────
    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov *.webm)")
        if path:
            self._load_video(path)

    def random_video(self):
        folder = self.config.get("random_video_folder")
        if not folder or not os.path.exists(folder):
            QMessageBox.critical(self, "Error", "Invalid 'random_video_folder' in config.")
            return
        videos = [f for f in os.listdir(folder)
                  if f.endswith((".mp4", ".avi", ".mkv", ".mov", ".webm"))]
        if not videos:
            QMessageBox.critical(self, "Error", "No videos found in random_video_folder.")
            return
        self._load_video(os.path.join(folder, random.choice(videos)))

    def _load_video(self, path: str):
        self.video_name_label.setText(f"▶  {os.path.basename(path)}")
        self.motion_energy_label.setText("Processing video…")
        self.video_output_label.setText("Analysing sign segment…")
        self.config["video_file"] = path
        with open("config.json", "w") as f:
            json.dump(self.config, f, indent=4)
        self.worker.change_source(path)

    # ── Camera control ────────────────────────────────────────────────────────
    def on_camera_changed(self, text: str):
        if text.startswith("Camera"):
            idx = int(text.split(" ")[1])
            self.config["camera_index"] = idx
            with open("config.json", "w") as f:
                json.dump(self.config, f, indent=4)
            self.rt_name_label.setText(f"Camera {idx}")
            self.worker.change_source(idx)

    # ── Settings ──────────────────────────────────────────────────────────────
    def open_settings(self):
        dialog = SettingsDialog(self.config, self)
        if dialog.exec_() == QDialog.Accepted:
            self.worker.draw_landmarks_flag             = self.config.get("draw_landmarks", True)
            self.worker.motion_check_interval           = self.config.get("motion_check_interval", 5)
            self.worker.periodic_fallback_interval      = self.config.get("periodic_fallback_interval", 30)
            self.worker.inference_worker.update_model_path(self.config.get("prediction_model", ""))

    # ── Processing control ────────────────────────────────────────────────────
    def start(self):
        self.worker.start_processing()

    def stop(self):
        self.worker.stop_processing()

    # ── Signal handlers ───────────────────────────────────────────────────────
    def update_frame(self, frame):
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg     = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap   = QPixmap.fromImage(qimg)

        idx = self.stacked_widget.currentIndex()
        if idx == 1:
            lbl = self.video_feed_label
        elif idx == 2:
            lbl = self.rt_feed_label
        else:
            return

        lbl.setPixmap(pixmap.scaled(lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_text(self, text: str):
        idx = self.stacked_widget.currentIndex()
        if idx == 1:
            output_lbl = self.video_output_label
        elif idx == 2:
            output_lbl = self.rt_output_label
        else:
            return

        try:
            data = json.loads(text)
            mode = data.get("mode", "real_time")
            frame_range = data.get("frame_range")

            if mode == "real_time":
                # Use the new slices format for real-time mode too. Real-time is fast enough for just the Full slice top results.
                slices = data.get("slices", {})
                top = slices.get("Full", [])
                html = _render_rt_table(top, frame_range)
                output_lbl.setText(html)

            elif mode == "video_testing":
                slices = data.get("slices", {})
                gt     = None
                if isinstance(self.worker.source, str):
                    fname = os.path.basename(self.worker.source)
                    # Filename format: WORD_id-LABEL.mp4
                    gt = fname.split("_")[0].strip().lower() if "_" in fname else None
                html = _render_video_table(slices, gt, frame_range)
                output_lbl.setText(html)

            elif mode == "error":
                output_lbl.setText(f"<span style='color:#e66;'>{data.get('message','Error')}</span>")

        except Exception:
            # Fallback: plain text (e.g. "Processing…" interim messages)
            output_lbl.setText(text)

    def update_motion_energy(self, data: dict):
        """Render the motion energy bar in the active screen."""
        idx = self.stacked_widget.currentIndex()
        if idx == 1:
            lbl = self.motion_energy_label
        elif idx == 2:
            lbl = self.rt_motion_energy_label
        else:
            return

        pmap = _render_motion_energy_pixmap(data, width=lbl.width() or 700, height=60)
        if pmap:
            lbl.setPixmap(pmap)
            # Summary text as tooltip
            segs = data.get("segments", [])
            best = data.get("best_seg")
            tip  = f"{len(segs)} segment(s) found."
            if best:
                tip += f"  Best: frames {best[0]}–{best[1]} ({best[1]-best[0]} frames)"
            lbl.setToolTip(tip)

    def closeEvent(self, event):
        if hasattr(self, "worker") and self.worker:
            self.worker.stop()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app    = QApplication(sys.argv)
    window = App()
    window.resize(1000, 780)
    window.show()
    sys.exit(app.exec_())