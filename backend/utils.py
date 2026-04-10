"""
AirCanvas Pro — Utilities
Drawing engine, smoothing, eraser, color management, frame helpers.
"""

import cv2
import numpy as np
from collections import deque
import math


# ── Color palette (name → BGR) ─────────────────────────────
PEN_COLORS = {
    "red":    (0,   0,   220),
    "blue":   (220, 80,  0),
    "green":  (0,   200, 60),
    "yellow": (0,   220, 220),
    "white":  (255, 255, 255),
    "eraser": None,   # handled by DrawingEngine
}

# ── Constants ──────────────────────────────────────────────
DEFAULT_PEN_THICKNESS  = 8
CURSOR_RADIUS          = 10
PINCH_THRESHOLD        = 45    # px — index-to-middle distance
HOVER_THRESHOLD        = 60    # px — fingertip to dock button center

# Eraser defaults
ERASER_W_DEFAULT       = 60    # rectangle half-width
ERASER_H_DEFAULT       = 60    # rectangle half-height
ERASER_W_MIN           = 20
ERASER_W_MAX           = 120
ERASER_H_MIN           = 20
ERASER_H_MAX           = 120

# Smoothing
SMOOTHING_FACTOR       = 5     # higher = smoother but more lag (4–7 recommended)
JUMP_THRESHOLD         = 80    # px — ignore jumps larger than this


# ── Interpolation smoother ─────────────────────────────────

class InterpolationSmoother:
    """
    Exponential moving average smoother.
    new = prev + (current - prev) / factor
    Prevents jitter while keeping responsiveness.
    """

    def __init__(self, factor: int = SMOOTHING_FACTOR):
        self.factor = max(1, factor)
        self._sx: float | None = None
        self._sy: float | None = None

    def update(self, x: int, y: int) -> tuple[int, int]:
        if self._sx is None:
            self._sx, self._sy = float(x), float(y)
        else:
            self._sx += (x - self._sx) / self.factor
            self._sy += (y - self._sy) / self.factor
        return int(self._sx), int(self._sy)

    def reset(self):
        self._sx = None
        self._sy = None

    @property
    def has_value(self) -> bool:
        return self._sx is not None


# ── Debouncer ──────────────────────────────────────────────

class Debouncer:
    """Prevents rapid repeated triggers within a cooldown window."""

    def __init__(self, cooldown_frames: int = 18):
        self.cooldown = cooldown_frames
        self._counter = 0

    def trigger(self) -> bool:
        if self._counter == 0:
            self._counter = self.cooldown
            return True
        return False

    def tick(self):
        if self._counter > 0:
            self._counter -= 1


# ── Drawing engine ─────────────────────────────────────────

class DrawingEngine:
    """
    Manages the drawing canvas with smooth strokes and a
    rectangular eraser whose size can be set dynamically.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.width  = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Stroke state
        self._prev_x: int | None = None
        self._prev_y: int | None = None

        # Pen settings
        self.pen_color: tuple = PEN_COLORS["red"]
        self.pen_thickness: int = DEFAULT_PEN_THICKNESS

        # Eraser settings (rectangular)
        self.eraser_w: int = ERASER_W_DEFAULT
        self.eraser_h: int = ERASER_H_DEFAULT

    # ── Canvas management ──────────────────────────────────

    def resize(self, w: int, h: int):
        if w != self.width or h != self.height:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
            self.width, self.height = w, h

    def clear(self):
        self.canvas[:] = 0

    def reset_stroke(self):
        """Call when lifting finger or switching modes."""
        self._prev_x = None
        self._prev_y = None

    # ── Drawing ────────────────────────────────────────────

    def draw_stroke(self, x: int, y: int):
        """
        Draw a continuous line from the previous point to (x, y).
        Applies a jump-distance guard to avoid sudden long strokes.
        """
        if self._prev_x is not None and self._prev_y is not None:
            dist = math.hypot(x - self._prev_x, y - self._prev_y)
            if dist < JUMP_THRESHOLD:
                cv2.line(
                    self.canvas,
                    (self._prev_x, self._prev_y),
                    (x, y),
                    self.pen_color,
                    self.pen_thickness,
                    lineType=cv2.LINE_AA,
                )
        self._prev_x, self._prev_y = x, y

    # ── Erasing ────────────────────────────────────────────

    def erase_rect(self, cx: int, cy: int):
        """
        Erase a rectangle centred on (cx, cy) by filling with black.
        Also erases along the stroke path for smooth erasing.
        """
        x1 = max(0, cx - self.eraser_w)
        y1 = max(0, cy - self.eraser_h)
        x2 = min(self.width,  cx + self.eraser_w)
        y2 = min(self.height, cy + self.eraser_h)
        cv2.rectangle(self.canvas, (x1, y1), (x2, y2), (0, 0, 0), -1)

        # Fill along stroke path so fast movement doesn't leave gaps
        if self._prev_x is not None and self._prev_y is not None:
            dist = math.hypot(cx - self._prev_x, cy - self._prev_y)
            if dist < JUMP_THRESHOLD and dist > 1:
                steps = max(int(dist / 4), 1)
                for i in range(1, steps):
                    t = i / steps
                    ix = int(self._prev_x + (cx - self._prev_x) * t)
                    iy = int(self._prev_y + (cy - self._prev_y) * t)
                    ex1 = max(0, ix - self.eraser_w)
                    ey1 = max(0, iy - self.eraser_h)
                    ex2 = min(self.width,  ix + self.eraser_w)
                    ey2 = min(self.height, iy + self.eraser_h)
                    cv2.rectangle(self.canvas, (ex1, ey1), (ex2, ey2), (0, 0, 0), -1)

        self._prev_x, self._prev_y = cx, cy

    def set_eraser_size_from_pinch(self, pinch_dist: float):
        """
        Map thumb-index pinch distance to eraser size dynamically.
        Larger spread → bigger eraser.
        """
        # Map pinch_dist [20, 150] → eraser half-size [20, 120]
        size = int(np.interp(pinch_dist, [20, 150], [ERASER_W_MIN, ERASER_W_MAX]))
        self.eraser_w = size
        self.eraser_h = size

    def adjust_eraser_size(self, delta: int):
        """Keyboard control: pass +10 or -10."""
        self.eraser_w = int(np.clip(self.eraser_w + delta, ERASER_W_MIN, ERASER_W_MAX))
        self.eraser_h = int(np.clip(self.eraser_h + delta, ERASER_H_MIN, ERASER_H_MAX))

    # ── Eraser preview ─────────────────────────────────────

    def draw_eraser_preview(self, frame: np.ndarray, cx: int, cy: int):
        """
        Draw a dashed rectangle preview on the OUTPUT frame (not canvas).
        Shows the eraser area before the user commits to erasing.
        """
        x1 = max(0, cx - self.eraser_w)
        y1 = max(0, cy - self.eraser_h)
        x2 = min(frame.shape[1] - 1, cx + self.eraser_w)
        y2 = min(frame.shape[0] - 1, cy + self.eraser_h)

        # Outer white border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        # Inner dark border for contrast
        cv2.rectangle(frame, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (40, 40, 40), 1)

        # Size label
        label = f"{self.eraser_w * 2}x{self.eraser_h * 2}"
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


# ── Frame helpers ──────────────────────────────────────────

def encode_frame(frame: np.ndarray, quality: int = 78) -> bytes:
    """JPEG-encode a BGR frame."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def decode_frame(data: bytes) -> np.ndarray | None:
    """Decode JPEG bytes to BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def overlay_canvas(frame: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    """
    Composite drawing canvas over webcam frame.
    Only non-black canvas pixels are blended on top.
    """
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    return cv2.add(bg, fg)


def draw_cursor(frame: np.ndarray, x: int, y: int,
                color: tuple, mode: str):
    """Render a fingertip cursor indicator appropriate for the current mode."""
    if x is None or y is None:
        return

    if mode == "drawing":
        # Filled dot in pen colour with white ring
        cv2.circle(frame, (x, y), CURSOR_RADIUS, color, -1)
        cv2.circle(frame, (x, y), CURSOR_RADIUS + 2, (255, 255, 255), 1)

    elif mode == "selection":
        # Hollow ring + centre dot
        cv2.circle(frame, (x, y), CURSOR_RADIUS + 4, (200, 200, 200), 2)
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

    elif mode == "pinch":
        # Cyan filled dot
        cv2.circle(frame, (x, y), CURSOR_RADIUS, (0, 230, 230), -1)
        cv2.circle(frame, (x, y), CURSOR_RADIUS + 2, (255, 255, 255), 1)

    elif mode == "erasing":
        # Small crosshair at centre
        cv2.line(frame, (x - 8, y), (x + 8, y), (255, 255, 255), 1)
        cv2.line(frame, (x, y - 8), (x, y + 8), (255, 255, 255), 1)

    else:
        cv2.circle(frame, (x, y), CURSOR_RADIUS, (140, 140, 140), 1)
