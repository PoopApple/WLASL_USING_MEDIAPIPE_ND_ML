from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import json
import os


class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.config = config

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.pose_model_select = QComboBox()
        self.pose_model_select.addItems(["lite", "full", "heavy"])
        self.pose_model_select.setCurrentText(self.config.get("pose_model", "lite"))

        form_layout.addRow("Pose Model:", self.pose_model_select)

        self.inference_interval_spinbox = QSpinBox()
        self.inference_interval_spinbox.setRange(1, 1000)
        self.inference_interval_spinbox.setValue(self.config.get("inference_interval", 10))
        form_layout.addRow("Inference Interval (frames):", self.inference_interval_spinbox)

        self.draw_landmarks_checkbox = QCheckBox("Draw Landmarks")
        self.draw_landmarks_checkbox.setChecked(self.config.get("draw_landmarks", True))
        form_layout.addRow(self.draw_landmarks_checkbox)

        video_file_layout = QHBoxLayout()
        self.video_file_input = QLineEdit(self.config.get("video_file", ""))
        self.video_file_btn = QPushButton("Browse")
        self.video_file_btn.clicked.connect(self.browse_video)
        video_file_layout.addWidget(self.video_file_input)
        video_file_layout.addWidget(self.video_file_btn)

        form_layout.addRow("Video File:", video_file_layout)

        pred_model_layout = QHBoxLayout()
        self.pred_model_input = QLineEdit(self.config.get("prediction_model", ""))
        self.pred_model_btn = QPushButton("Browse")
        self.pred_model_btn.clicked.connect(self.browse_pred_model)
        pred_model_layout.addWidget(self.pred_model_input)
        pred_model_layout.addWidget(self.pred_model_btn)

        form_layout.addRow("Prediction Model:", pred_model_layout)

        layout.addLayout(form_layout)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self.setLayout(layout)

    def browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video")
        if file_path:
            self.video_file_input.setText(file_path)

    def browse_pred_model(self):
        os.makedirs("ASL_Models", exist_ok=True)
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Prediction Model", "ASL_Models", "Keras Models (*.keras)")
        if file_path:
            self.pred_model_input.setText(file_path)

    def accept(self):
        self.config["pose_model"] = self.pose_model_select.currentText()
        self.config["video_file"] = self.video_file_input.text()
        self.config["prediction_model"] = self.pred_model_input.text()
        self.config["inference_interval"] = self.inference_interval_spinbox.value()
        self.config["draw_landmarks"] = self.draw_landmarks_checkbox.isChecked()
        with open("config.json", "w") as f:
            json.dump(self.config, f, indent=4)
        super().accept()


class MainUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Vision App")
        self.stacked_widget = QStackedWidget()

        # --- Screen 0: Home Menu ---
        self.home_widget = QWidget()
        home_layout = QVBoxLayout()
        home_layout.setAlignment(Qt.AlignCenter)
        
        title_label = QLabel("Vision App")
        title_font = title_label.font()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        
        self.nav_video_btn = QPushButton("Video Testing")
        self.nav_video_btn.setMinimumHeight(50)
        self.nav_realtime_btn = QPushButton("Real Time")
        self.nav_realtime_btn.setMinimumHeight(50)
        self.nav_settings_btn = QPushButton("Settings")
        self.nav_settings_btn.setMinimumHeight(50)
        
        home_layout.addWidget(title_label)
        home_layout.addSpacing(30)
        home_layout.addWidget(self.nav_video_btn)
        home_layout.addWidget(self.nav_realtime_btn)
        home_layout.addWidget(self.nav_settings_btn)
        self.home_widget.setLayout(home_layout)

        # --- Screen 1: Video Testing ---
        self.video_widget = QWidget()
        video_layout = QVBoxLayout()
        
        video_top_layout = QHBoxLayout()
        self.video_back_btn = QPushButton("Back")
        self.video_choose_btn = QPushButton("Choose Video")
        self.video_random_btn = QPushButton("Random Video")
        video_top_layout.addWidget(self.video_back_btn)
        video_top_layout.addWidget(self.video_choose_btn)
        video_top_layout.addWidget(self.video_random_btn)
        
        self.video_feed_label = QLabel()
        self.video_feed_label.setAlignment(Qt.AlignCenter)
        self.video_name_label = QLabel("")
        self.video_name_label.setAlignment(Qt.AlignCenter)
        self.video_name_label.setStyleSheet("font-weight: bold; color: #777;")
        self.video_output_label = QLabel("Output...")
        self.video_output_label.setAlignment(Qt.AlignCenter)
        
        video_layout.addLayout(video_top_layout)
        video_layout.addWidget(self.video_feed_label)
        video_layout.addWidget(self.video_name_label)
        video_layout.addWidget(self.video_output_label)
        self.video_widget.setLayout(video_layout)

        # --- Screen 2: Real Time ---
        self.rt_widget = QWidget()
        rt_layout = QVBoxLayout()
        
        rt_top_layout = QHBoxLayout()
        self.rt_back_btn = QPushButton("Back")
        self.rt_camera_select = QComboBox()
        self.rt_camera_select.addItems(["Camera 0", "Camera 1", "Camera 2"])
        rt_top_layout.addWidget(self.rt_back_btn)
        rt_top_layout.addWidget(self.rt_camera_select)
        
        self.rt_feed_label = QLabel()
        self.rt_feed_label.setAlignment(Qt.AlignCenter)
        self.rt_name_label = QLabel("")
        self.rt_name_label.setAlignment(Qt.AlignCenter)
        self.rt_name_label.setStyleSheet("font-weight: bold; color: #777;")
        self.rt_output_label = QLabel("Output...")
        self.rt_output_label.setAlignment(Qt.AlignCenter)
        
        rt_ctrl_layout = QHBoxLayout()
        self.rt_start_btn = QPushButton("Start")
        self.rt_stop_btn = QPushButton("Stop")
        rt_ctrl_layout.addWidget(self.rt_start_btn)
        rt_ctrl_layout.addWidget(self.rt_stop_btn)
        
        rt_layout.addLayout(rt_top_layout)
        rt_layout.addWidget(self.rt_feed_label)
        rt_layout.addWidget(self.rt_name_label)
        rt_layout.addWidget(self.rt_output_label)
        rt_layout.addLayout(rt_ctrl_layout)
        self.rt_widget.setLayout(rt_layout)

        self.stacked_widget.addWidget(self.home_widget)   # 0
        self.stacked_widget.addWidget(self.video_widget)  # 1
        self.stacked_widget.addWidget(self.rt_widget)     # 2

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stacked_widget)
        self.setLayout(main_layout)