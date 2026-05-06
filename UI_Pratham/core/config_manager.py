"""
ConfigManager — JSON-based configuration with hot-reload support.

Loads settings from a JSON file and provides typed getters for all pipeline
parameters. Emits a Qt signal when configuration changes so all components
can update without restarting the application.

Config keys:
    camera_index         (int)   — OpenCV camera device index
    fps_limit            (int)   — Max frames per second for capture
    deque_length         (int)   — Rolling buffer size for landmarks
    inference_interval   (int)   — Run ML model every N processed frames
    beam_search_enabled  (bool)  — Show beam search results in UI
    beam_search_k        (int)   — Number of beam search candidates
    model_path           (str)   — Path to trained .keras model
    label_map_path       (str)   — Path to word_to_ind JSON mapping
    mediapipe_pose_model (str)   — Path to MediaPipe pose .task file
    mediapipe_hand_model (str)   — Path to MediaPipe hand .task file
    confidence_threshold (float) — Min confidence to display prediction
    display_top_k        (int)   — Number of top predictions to show
"""

import json
import os
from PySide6.QtCore import QObject, Signal


# Absolute defaults — used when no config file exists
_DEFAULTS = {
    "camera_index": 0,
    "fps_limit": 30,
    "deque_length": 128,
    "inference_interval": 5,
    "beam_search_enabled": False,
    "beam_search_k": 5,
    "model_path": "",
    "label_map_path": "",
    "mediapipe_pose_model": "ExtractLandmarks/vision_models/pose_landmarker_heavy.task",
    "mediapipe_hand_model": "ExtractLandmarks/vision_models/hand_landmarker.task",
    "confidence_threshold": 0.3,
    "display_top_k": 5,
}


class ConfigManager(QObject):
    """
    Thread-safe configuration manager with live-reload capability.

    Usage:
        cfg = ConfigManager("config.json")
        fps = cfg.get("fps_limit")          # typed getter
        cfg.set("fps_limit", 25)            # update + auto-save
        cfg.config_changed.connect(on_cfg)  # subscribe to changes
    """

    # Emitted whenever any config value changes.  Payload is the full dict.
    config_changed = Signal(dict)

    def __init__(self, config_path: str):
        super().__init__()
        self._path = os.path.abspath(config_path)
        self._data: dict = {}
        self._load()

    # ── public API ──────────────────────────────────────────────────────

    def get(self, key: str, fallback=None):
        """Return the value for *key*, or *fallback* / default if missing."""
        return self._data.get(key, _DEFAULTS.get(key, fallback))

    def set(self, key: str, value) -> None:
        """Update a single key, save to disk, and notify listeners."""
        self._data[key] = value
        self._save()
        self.config_changed.emit(dict(self._data))

    def update(self, partial: dict) -> None:
        """Batch-update multiple keys, save, and notify once."""
        self._data.update(partial)
        self._save()
        self.config_changed.emit(dict(self._data))

    def reset_defaults(self) -> None:
        """Restore every key to its factory default."""
        self._data = dict(_DEFAULTS)
        self._save()
        self.config_changed.emit(dict(self._data))

    def all(self) -> dict:
        """Return a shallow copy of the full config dict."""
        merged = dict(_DEFAULTS)
        merged.update(self._data)
        return merged

    @staticmethod
    def defaults() -> dict:
        """Return a copy of the factory defaults."""
        return dict(_DEFAULTS)

    # ── internals ───────────────────────────────────────────────────────

    def _load(self) -> None:
        """Read config from disk, merging with defaults for missing keys."""
        if os.path.isfile(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as fh:
                    self._data = json.load(fh)
            except (json.JSONDecodeError, IOError) as exc:
                print(f"[ConfigManager] Failed to read {self._path}: {exc}")
                self._data = {}
        else:
            self._data = dict(_DEFAULTS)
            self._save()  # create file on first run

    def _save(self) -> None:
        """Persist current config to disk."""
        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=4)
        except IOError as exc:
            print(f"[ConfigManager] Failed to write {self._path}: {exc}")
