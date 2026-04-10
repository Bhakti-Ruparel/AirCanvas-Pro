"""
Hand tracking module — MediaPipe 0.10.x Tasks API.
Provides landmark detection, finger state, and distance utilities.
"""

import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers import landmark as mp_landmark

# Download the hand landmarker model on first use
import urllib.request, pathlib, os

MODEL_PATH = pathlib.Path(__file__).parent / "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def _ensure_model():
    if not MODEL_PATH.exists():
        print(f"[HandTracker] Downloading model to {MODEL_PATH} …")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[HandTracker] Model downloaded.")

_ensure_model()


class HandTracker:
    """
    Wraps MediaPipe HandLandmarker (Tasks API, 0.10.x+).
    Call find_hands() each frame, then use the helper methods.
    """

    # Landmark indices (same as classic API)
    WRIST       = 0
    THUMB_TIP   = 4
    INDEX_MCP   = 5
    INDEX_TIP   = 8
    MIDDLE_MCP  = 9
    MIDDLE_TIP  = 12
    RING_TIP    = 16
    PINKY_TIP   = 20

    def __init__(self, max_hands: int = 1,
                 detection_confidence: float = 0.6,
                 tracking_confidence: float = 0.6):

        base_opts = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,   # per-frame (sync)
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(opts)
        self._result   = None
        self.landmark_list: list = []   # [[id, x_px, y_px], …]

    # ------------------------------------------------------------------
    def find_hands(self, frame: np.ndarray, draw: bool = False) -> np.ndarray:
        """
        Run detection on a BGR numpy frame.
        Populates internal result; optionally draws landmarks.
        Returns the (possibly annotated) frame.
        """
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._result = self._detector.detect(mp_image)

        if draw and self._result.hand_landmarks:
            frame = self._draw_landmarks(frame)

        return frame

    def _draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        import cv2
        h, w = frame.shape[:2]
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
            (0,17),
        ]
        for hand_lms in self._result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
            for a, b in connections:
                cv2.line(frame, pts[a], pts[b], (0, 200, 100), 1)
            for x, y in pts:
                cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
        return frame

    # ------------------------------------------------------------------
    def find_landmarks(self, frame: np.ndarray, hand_no: int = 0) -> list:
        """
        Extract pixel-space landmarks for hand `hand_no`.
        Returns [[id, x, y], …] or [] if no hand detected.
        """
        self.landmark_list = []
        if not self._result or not self._result.hand_landmarks:
            return self.landmark_list
        if hand_no >= len(self._result.hand_landmarks):
            return self.landmark_list

        h, w = frame.shape[:2]
        for idx, lm in enumerate(self._result.hand_landmarks[hand_no]):
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            self.landmark_list.append([idx, cx, cy])

        return self.landmark_list

    # ------------------------------------------------------------------
    def fingers_up(self) -> list:
        """
        Returns [thumb, index, middle, ring, pinky] as booleans.
        True = finger extended.
        """
        fingers = [False] * 5
        lm = self.landmark_list
        if len(lm) < 21:
            return fingers

        # Thumb: tip x vs IP joint x (works for mirrored/front-facing cam)
        fingers[0] = lm[self.THUMB_TIP][1] < lm[self.THUMB_TIP - 1][1]

        # Fingers 1-4: tip y < MCP y  (y increases downward)
        tip_ids = [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        mcp_ids = [self.INDEX_MCP, self.MIDDLE_MCP, 13, 17]
        for i, (tip, mcp) in enumerate(zip(tip_ids, mcp_ids)):
            fingers[i + 1] = lm[tip][2] < lm[mcp][2]

        return fingers

    # ------------------------------------------------------------------
    def find_distance(self, p1: int, p2: int) -> tuple:
        """
        Euclidean distance between landmarks p1 and p2.
        Returns (distance, mid_x, mid_y).
        """
        if len(self.landmark_list) < 21:
            return 0, 0, 0
        x1, y1 = self.landmark_list[p1][1], self.landmark_list[p1][2]
        x2, y2 = self.landmark_list[p2][1], self.landmark_list[p2][2]
        dist = math.hypot(x2 - x1, y2 - y1)
        return dist, (x1 + x2) // 2, (y1 + y2) // 2

    # ------------------------------------------------------------------
    def get_landmark(self, idx: int) -> tuple:
        """Return (x, y) for landmark idx, or (None, None)."""
        if len(self.landmark_list) > idx:
            return self.landmark_list[idx][1], self.landmark_list[idx][2]
        return None, None

    def hand_detected(self) -> bool:
        return bool(self._result and self._result.hand_landmarks)
