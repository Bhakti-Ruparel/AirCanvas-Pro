"""
AirCanvas Pro — FastAPI Backend
Handles WebSocket frame streaming, hand tracking, gesture recognition,
and drawing canvas state.
"""

import asyncio
import base64
import json
import logging
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from hand_tracking import HandTracker
from utils import (
    SmoothingFilter,
    Debouncer,
    PEN_COLORS,
    DEFAULT_PEN_THICKNESS,
    ERASER_THICKNESS,
    PINCH_THRESHOLD,
    HOVER_THRESHOLD,
    encode_frame,
    decode_frame,
    overlay_canvas,
    draw_cursor,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aircanvas")

app = FastAPI(title="AirCanvas Pro", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Serve frontend static files
# ---------------------------------------------------------------------------
import pathlib

FRONTEND_DIR = pathlib.Path(__file__).parent.parent / "frontend"

if FRONTEND_DIR.exists():
    @app.get("/")
    async def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/style.css")
    async def serve_css():
        return FileResponse(str(FRONTEND_DIR / "style.css"), media_type="text/css")

    @app.get("/script.js")
    async def serve_js():
        return FileResponse(str(FRONTEND_DIR / "script.js"), media_type="application/javascript")


# ---------------------------------------------------------------------------
# Per-connection drawing session state
# ---------------------------------------------------------------------------

class DrawingSession:
    """Holds all mutable state for one WebSocket client session."""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.tracker = HandTracker(max_hands=1)
        self.smoother = SmoothingFilter(window_size=5)
        self.debouncer = Debouncer(cooldown_frames=20)

        # Drawing state
        self.prev_x: Optional[int] = None
        self.prev_y: Optional[int] = None
        self.current_color: str = "red"
        self.mode: str = "idle"   # idle | drawing | selection | pinch | paused
        self.brush_thickness: int = DEFAULT_PEN_THICKNESS

        # UI dock button regions (populated per frame based on frame size)
        self.dock_buttons: list = []

        # FPS tracking
        self._last_time = time.time()
        self.fps: float = 0.0

    def reset_canvas(self):
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def resize_canvas(self, w: int, h: int):
        if w != self.width or h != self.height:
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
            self.width, self.height = w, h

    def tick_fps(self):
        now = time.time()
        elapsed = now - self._last_time
        self.fps = 1.0 / elapsed if elapsed > 0 else 0
        self._last_time = now

    def build_dock_buttons(self) -> list:
        """
        Compute dock button hit-areas based on current frame dimensions.
        Returns list of dicts: {name, color_key, cx, cy, radius}
        """
        colors = ["red", "blue", "green", "yellow", "white", "eraser"]
        n = len(colors)
        btn_r = 28
        spacing = 72
        total_w = (n - 1) * spacing
        start_x = (self.width - total_w) // 2
        cy = self.height - 55

        buttons = []
        for i, name in enumerate(colors):
            cx = start_x + i * spacing
            buttons.append({"name": name, "cx": cx, "cy": cy, "radius": btn_r})
        self.dock_buttons = buttons
        return buttons

    def check_hover(self, fx: int, fy: int) -> Optional[str]:
        """Return button name if fingertip is hovering over it."""
        for btn in self.dock_buttons:
            dist = np.hypot(fx - btn["cx"], fy - btn["cy"])
            if dist < HOVER_THRESHOLD:
                return btn["name"]
        return None


# ---------------------------------------------------------------------------
# Frame processing pipeline
# ---------------------------------------------------------------------------

def process_frame(session: DrawingSession, raw_bytes: bytes, flip: bool = True) -> dict:
    """
    Full pipeline: decode → detect → gesture → draw → encode.
    Returns dict with processed JPEG (base64) and metadata.
    """
    frame = decode_frame(raw_bytes)
    if frame is None:
        return {"error": "bad_frame"}

    # Flip horizontally (mirror) so it feels natural
    if flip:
        frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    session.resize_canvas(w, h)
    session.tick_fps()
    session.debouncer.tick()

    # --- Hand detection ---
    frame = session.tracker.find_hands(frame, draw=False)
    landmarks = session.tracker.find_landmarks(frame)

    gesture = "none"
    hovered_btn = None
    ix, iy = None, None  # index fingertip

    if landmarks:
        fingers = session.tracker.fingers_up()
        # [thumb, index, middle, ring, pinky]
        index_up  = fingers[1]
        middle_up = fingers[2]
        ring_up   = fingers[3]
        pinky_up  = fingers[4]

        raw_ix, raw_iy = session.tracker.get_landmark(8)   # index tip
        ix, iy = session.smoother.update(raw_ix, raw_iy)

        # Pinch detection (index + middle close together)
        dist, _, _ = session.tracker.find_distance(8, 12)
        is_pinch = dist < PINCH_THRESHOLD

        # Gesture classification
        all_down = not any(fingers[1:])
        only_index = index_up and not middle_up and not ring_up and not pinky_up

        if all_down:
            gesture = "paused"
            session.mode = "paused"
            session.prev_x, session.prev_y = None, None

        elif is_pinch and index_up and middle_up:
            gesture = "pinch"
            session.mode = "pinch"
            session.prev_x, session.prev_y = None, None

            # Check dock interaction
            session.build_dock_buttons()
            hovered_btn = session.check_hover(ix, iy)
            if hovered_btn and session.debouncer.trigger():
                session.current_color = hovered_btn
                if hovered_btn == "eraser":
                    session.brush_thickness = ERASER_THICKNESS
                else:
                    session.brush_thickness = DEFAULT_PEN_THICKNESS

        elif index_up and middle_up and not is_pinch:
            gesture = "selection"
            session.mode = "selection"
            session.prev_x, session.prev_y = None, None

            # Hover highlight (no click yet)
            session.build_dock_buttons()
            hovered_btn = session.check_hover(ix, iy)

        elif only_index:
            gesture = "drawing"
            session.mode = "drawing"

            if session.prev_x is not None and session.prev_y is not None:
                if session.current_color == "eraser":
                    cv2.line(session.canvas, (session.prev_x, session.prev_y),
                             (ix, iy), (0, 0, 0), ERASER_THICKNESS)
                else:
                    color_bgr = PEN_COLORS[session.current_color]
                    cv2.line(session.canvas, (session.prev_x, session.prev_y),
                             (ix, iy), color_bgr, session.brush_thickness)

            session.prev_x, session.prev_y = ix, iy

        else:
            gesture = "idle"
            session.mode = "idle"
            session.prev_x, session.prev_y = None, None

    else:
        session.smoother.reset()
        session.prev_x, session.prev_y = None, None
        session.mode = "idle"

    # --- Compose output frame ---
    output = overlay_canvas(frame, session.canvas)

    # Draw dock buttons overlay
    session.build_dock_buttons()
    _draw_dock(output, session.dock_buttons, session.current_color, hovered_btn)

    # Draw cursor
    if ix is not None:
        color_bgr = PEN_COLORS.get(session.current_color, (255, 255, 255)) or (255, 255, 255)
        draw_cursor(output, ix, iy, color_bgr, session.mode)

    # FPS overlay
    cv2.putText(output, f"FPS: {session.fps:.0f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # Mode label
    mode_colors = {
        "drawing": (0, 255, 100),
        "selection": (200, 200, 0),
        "pinch": (0, 255, 255),
        "paused": (100, 100, 100),
        "idle": (180, 180, 180),
    }
    label_color = mode_colors.get(session.mode, (180, 180, 180))
    cv2.putText(output, session.mode.upper(), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 1, cv2.LINE_AA)

    encoded = encode_frame(output, quality=75)
    b64 = base64.b64encode(encoded).decode("utf-8")

    return {
        "frame": b64,
        "gesture": gesture,
        "mode": session.mode,
        "color": session.current_color,
        "fps": round(session.fps, 1),
        "hovered": hovered_btn,
    }


def _draw_dock(frame: np.ndarray, buttons: list, active_color: str, hovered: Optional[str]):
    """Render the pen dock onto the frame."""
    if not buttons:
        return

    h, w = frame.shape[:2]
    # Semi-transparent dock background
    btn_r = buttons[0]["radius"]
    pad = 18
    xs = [b["cx"] for b in buttons]
    dock_x1 = min(xs) - btn_r - pad
    dock_x2 = max(xs) + btn_r + pad
    dock_y1 = buttons[0]["cy"] - btn_r - pad
    dock_y2 = min(h - 1, buttons[0]["cy"] + btn_r + pad)

    overlay = frame.copy()
    cv2.rectangle(overlay, (dock_x1, dock_y1), (dock_x2, dock_y2),
                  (30, 30, 40), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (dock_x1, dock_y1), (dock_x2, dock_y2),
                  (80, 80, 100), 1)

    color_map = {
        "red":    (0,   0,   220),
        "blue":   (220, 80,  0),
        "green":  (0,   200, 60),
        "yellow": (0,   220, 220),
        "white":  (240, 240, 240),
        "eraser": (120, 120, 120),
    }

    for btn in buttons:
        name = btn["name"]
        cx, cy = btn["cx"], btn["cy"]
        r = btn["radius"]
        bgr = color_map.get(name, (200, 200, 200))

        is_active = (name == active_color)
        is_hovered = (name == hovered)

        # Lift active button
        if is_active:
            cy -= 10
            r += 4

        # Draw shadow
        cv2.circle(frame, (cx + 2, cy + 3), r, (10, 10, 10), -1)
        # Fill
        cv2.circle(frame, (cx, cy), r, bgr, -1)
        # Border
        border_color = (255, 255, 255) if is_active else (160, 160, 160)
        border_thick = 3 if is_active else 1
        cv2.circle(frame, (cx, cy), r, border_color, border_thick)

        # Hover ring
        if is_hovered and not is_active:
            cv2.circle(frame, (cx, cy), r + 5, (255, 255, 255), 1)

        # Eraser icon (X)
        if name == "eraser":
            off = r // 2
            cv2.line(frame, (cx - off, cy - off), (cx + off, cy + off), (255, 255, 255), 2)
            cv2.line(frame, (cx + off, cy - off), (cx - off, cy + off), (255, 255, 255), 2)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = DrawingSession()
    logger.info("Client connected")

    try:
        while True:
            # Receive message (JSON with base64 frame + optional commands)
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            # Handle control commands
            if msg.get("type") == "clear":
                session.reset_canvas()
                await websocket.send_text(json.dumps({"type": "cleared"}))
                continue

            if msg.get("type") == "set_color":
                color = msg.get("color", "red")
                if color in PEN_COLORS:
                    session.current_color = color
                    session.brush_thickness = ERASER_THICKNESS if color == "eraser" else DEFAULT_PEN_THICKNESS
                continue

            if msg.get("type") == "get_canvas":
                # Return raw canvas as PNG for saving
                _, buf = cv2.imencode(".png", session.canvas)
                b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                await websocket.send_text(json.dumps({"type": "canvas_data", "data": b64}))
                continue

            # Normal frame processing
            if msg.get("type") == "frame":
                frame_b64 = msg.get("data", "")
                if not frame_b64:
                    continue
                frame_bytes = base64.b64decode(frame_b64)

                # Run in thread pool to avoid blocking event loop
                result = await asyncio.get_event_loop().run_in_executor(
                    None, process_frame, session, frame_bytes, True
                )
                result["type"] = "frame"
                await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Session error: {e}", exc_info=True)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# REST fallback endpoint
# ---------------------------------------------------------------------------

from fastapi import UploadFile, File
from fastapi.responses import JSONResponse

_rest_sessions: dict = {}

@app.post("/process-frame")
async def process_frame_rest(file: UploadFile = File(...), session_id: str = "default"):
    """Fallback REST endpoint for single-frame processing."""
    if session_id not in _rest_sessions:
        _rest_sessions[session_id] = DrawingSession()
    session = _rest_sessions[session_id]

    raw = await file.read()
    result = await asyncio.get_event_loop().run_in_executor(
        None, process_frame, session, raw, True
    )
    return JSONResponse(result)


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in _rest_sessions:
        _rest_sessions[session_id].reset_canvas()
    return {"status": "cleared"}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "AirCanvas Pro"}
