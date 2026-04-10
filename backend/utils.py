"""
Utility functions for AirCanvas Pro.
Smoothing, color management, and frame helpers.
"""

import cv2
import numpy as np
from collections import deque


class SmoothingFilter:
    """Moving average filter for smoothing coordinate jitter."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.x_buf = deque(maxlen=window_size)
        self.y_buf = deque(maxlen=window_size)

    def update(self, x: int, y: int) -> tuple:
        self.x_buf.append(x)
        self.y_buf.append(y)
        return int(np.mean(self.x_buf)), int(np.mean(self.y_buf))

    def reset(self):
        self.x_buf.clear()
        self.y_buf.clear()


class Debouncer:
    """Prevents rapid repeated triggers within a cooldown period."""

    def __init__(self, cooldown_frames: int = 15):
        self.cooldown = cooldown_frames
        self._counter = 0

    def trigger(self) -> bool:
        """Returns True if action should fire, False if still in cooldown."""
        if self._counter == 0:
            self._counter = self.cooldown
            return True
        return False

    def tick(self):
        """Call once per frame to decrement cooldown."""
        if self._counter > 0:
            self._counter -= 1


# Pen color palette (name → BGR)
PEN_COLORS = {
    "red":    (0,   0,   220),
    "blue":   (220, 80,  0),
    "green":  (0,   200, 60),
    "yellow": (0,   220, 220),
    "white":  (255, 255, 255),
    "eraser": None,  # handled separately
}

# Default drawing settings
DEFAULT_PEN_THICKNESS = 8
ERASER_THICKNESS = 40
CURSOR_RADIUS = 12
PINCH_THRESHOLD = 45   # pixels — index-to-middle distance for pinch
HOVER_THRESHOLD = 60   # pixels — fingertip to UI element center


def encode_frame(frame: np.ndarray, quality: int = 80) -> bytes:
    """JPEG-encode a frame and return raw bytes."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def decode_frame(data: bytes) -> np.ndarray:
    """Decode JPEG bytes into a BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def overlay_canvas(frame: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    """
    Merge drawing canvas onto webcam frame.
    Canvas pixels that are non-black are blended over the frame.
    """
    # Create mask where canvas has content
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Black out canvas area on frame, then add canvas
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    return cv2.add(frame_bg, canvas_fg)


def draw_cursor(frame: np.ndarray, x: int, y: int, color: tuple, mode: str):
    """Draw a fingertip cursor indicator on the frame."""
    if x is None or y is None:
        return
    if mode == "drawing":
        cv2.circle(frame, (x, y), CURSOR_RADIUS, color, -1)
        cv2.circle(frame, (x, y), CURSOR_RADIUS + 2, (255, 255, 255), 1)
    elif mode == "selection":
        cv2.circle(frame, (x, y), CURSOR_RADIUS, (200, 200, 200), 2)
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
    elif mode == "pinch":
        cv2.circle(frame, (x, y), CURSOR_RADIUS, (0, 255, 255), -1)
    else:
        cv2.circle(frame, (x, y), CURSOR_RADIUS, (150, 150, 150), 1)
