"""
Threading utilities — thread-safe data structures and helpers.

Provides a thin wrapper around collections.deque that is safe for the
producer/consumer pattern used in the pipeline (camera → mediapipe → deque,
inference thread reads snapshot).

CPython's GIL guarantees that deque.append() and deque.copy() are atomic
for single operations, but we add an explicit lock for the snapshot method
which reads the full buffer to prevent tearing during resize (when config
changes deque_length at runtime).
"""

import threading
import collections
import numpy as np


class SafeDeque:
    """
    A thread-safe fixed-length deque for landmark frames.

    Typical usage in the pipeline:
        buf = SafeDeque(maxlen=128)
        buf.append(frame_landmarks)          # called from mediapipe thread
        snapshot = buf.snapshot()             # called from inference thread
        buf.resize(64)                       # called from config hot-reload
    """

    def __init__(self, maxlen: int = 128):
        self._lock = threading.Lock()
        self._deque: collections.deque = collections.deque(maxlen=maxlen)

    def append(self, item: np.ndarray) -> None:
        """Push a single landmark frame (64, 4) into the buffer."""
        with self._lock:
            self._deque.append(item)

    def snapshot(self) -> list[np.ndarray]:
        """Return a shallow copy of the current buffer contents."""
        with self._lock:
            return list(self._deque)

    def clear(self) -> None:
        """Remove all items."""
        with self._lock:
            self._deque.clear()

    def resize(self, new_maxlen: int) -> None:
        """
        Change the maximum buffer length.

        If the new length is smaller, the oldest frames are silently
        discarded — matching the behavior of deque(maxlen=N).
        """
        with self._lock:
            old_items = list(self._deque)
            self._deque = collections.deque(old_items, maxlen=new_maxlen)

    def __len__(self) -> int:
        with self._lock:
            return len(self._deque)

    @property
    def maxlen(self) -> int:
        return self._deque.maxlen
