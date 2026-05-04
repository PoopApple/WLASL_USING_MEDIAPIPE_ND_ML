import sys
import json
import cv2
import os
import random

from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from ui_main import MainUI, SettingsDialog
from worker import VideoWorker


def load_config():
    with open("config.json") as f:
        return json.load(f)


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
        # Real Time controls
        self.rt_back_btn.clicked.connect(lambda: self.switch_mode(0))
        self.rt_camera_select.currentTextChanged.connect(self.on_camera_changed)
        self.rt_start_btn.clicked.connect(self.start)
        self.rt_stop_btn.clicked.connect(self.stop)
        
        # Set initial camera dropdown based on config
        self.rt_camera_select.setCurrentText(f"Camera {self.config['camera_index']}")
        
        self.worker.frame_signal.connect(self.update_frame)
        self.worker.text_signal.connect(self.update_text)
        self.worker.start()

    def switch_mode(self, index):
        self.stacked_widget.setCurrentIndex(index)
        self.stop()
        
        if index == 1:
            self.worker.set_mode("video_testing")
            self.video_name_label.setText("No video selected")
            self.worker.change_source(None) # Make it wait for user input
        elif index == 2:
            self.worker.set_mode("real_time")
            cam_idx = self.config.get("camera_index", 0)
            self.rt_camera_select.setCurrentText(f"Camera {cam_idx}")
            self.rt_name_label.setText(f"Playing: Camera {cam_idx}")
            self.worker.change_source(cam_idx)

    def choose_video(self):
        video_file, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if video_file:
            self.video_name_label.setText(f"Playing: {os.path.basename(video_file)}")
            self.config["video_file"] = video_file
            with open("config.json", "w") as f:
                json.dump(self.config, f, indent=4)
            self.worker.change_source(video_file)

    def random_video(self):
        folder = self.config.get("random_video_folder")
        if not folder or not os.path.exists(folder):
            QMessageBox.critical(self, "Error", "Invalid or missing 'random_video_folder' in config!")
            return
        
        videos = [f for f in os.listdir(folder) if f.endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm'))]
        if not videos:
            QMessageBox.critical(self, "Error", "No valid videos found in random_video_folder!")
            return
            
        chosen = random.choice(videos)
        self.video_name_label.setText(f"Playing: {chosen}")
        self.worker.change_source(os.path.join(folder, chosen))

    def on_camera_changed(self, text):
        if text.startswith("Camera"):
            idx = int(text.split(" ")[1])
            self.config["camera_index"] = idx
            
            with open("config.json", "w") as f:
                json.dump(self.config, f, indent=4)
                
            self.rt_name_label.setText(f"Playing: Camera {idx}")
            self.worker.change_source(idx)

    def open_settings(self):
        dialog = SettingsDialog(self.config, self)
        if dialog.exec_() == QDialog.Accepted:
            self.worker.draw_landmarks_flag = self.config.get("draw_landmarks", True)
            self.worker.interval = self.config.get("inference_interval", 10)
            self.worker.inference_worker.update_model_path(self.config.get("prediction_model", ""))

    def start(self):
        if self.worker:
            self.worker.start_processing()

    def stop(self):
        if self.worker:
            self.worker.stop_processing()

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)

        if self.stacked_widget.currentIndex() == 1:
            active_label = self.video_feed_label
        elif self.stacked_widget.currentIndex() == 2:
            active_label = self.rt_feed_label
        else:
            return

        scaled = pixmap.scaled(
            active_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        active_label.setPixmap(scaled)

    def update_text(self, text):
        if self.stacked_widget.currentIndex() == 1:
            active_label = self.video_output_label
        elif self.stacked_widget.currentIndex() == 2:
            active_label = self.rt_output_label
        else:
            return

        try:
            data = json.loads(text)

            ground_truth = None
            if self.stacked_widget.currentIndex() == 1 and isinstance(self.worker.source, str):
                import os
                filename = os.path.basename(self.worker.source)
                if "_" in filename:
                    ground_truth = filename.split("_")[0].strip().lower()

            titles = ["(0, 127)", "(62, 127)", "(0, 61)", "(32, 93)", "(19, 108)"]

            html = "<div align='center'><table width='100%' border='1' cellspacing='0' cellpadding='4' style='border-collapse: collapse;'>"
            html += "<tr style='background-color: #333; color: white;'>"
            for title in titles:
                html += f"<th>{title}</th>"
            html += "</tr>"

            for i in range(5):
                html += "<tr>"
                for seq_idx in range(5):
                    seq = data[seq_idx] if seq_idx < len(data) else []
                    
                    if seq and i < len(seq):
                        cell_text = seq[i]
                        pred_word = cell_text.split(" (")[0].strip().lower()
                        if ground_truth and pred_word == ground_truth:
                            html += f"<td align='center' style='background-color: rgba(19, 207, 56, 0.5);'><b>{cell_text}</b></td>"
                        else:
                            html += f"<td align='center'><b>{cell_text}</b></td>"
                    elif i == 0 and not seq:
                        html += "<td align='center'><i>Wait...</i></td>"
                    else:
                        html += "<td></td>"
                html += "</tr>"

            html += "</table></div>"
            active_label.setText(html)
        except Exception:
            # Fallback if it's not JSON (e.g. "Prediction Error" or initial states)
            active_label.setText(text)

    def closeEvent(self, event):
        if hasattr(self, 'worker') and self.worker:
            self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = App()
    window.resize(900, 700)
    window.show()

    sys.exit(app.exec_())