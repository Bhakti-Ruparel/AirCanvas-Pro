"""
AirCanvas Pro — FastAPI Backend
Professional drawing engine with smooth strokes, rectangular eraser,
dynamic eraser sizing, and clean mode switching.
"""

import asyncio
import base64
import json
import logging
import math
import pathlib
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from hand_tracking import HandTracker
from utils import (
    InterpolationSmoother,
    Debouncer,
    DrawingEngine,
    PEN_COLORS,
    DEFAULT_PEN_THICKNESS,
    PINCH_THRESHOLD,
    HOVER_THRESHOLD,
    encode_frame,
    decode_frame,
    overlay_canvas,
    draw_cursor,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aircanvas")

app = FastAPI(title="AirCanvas Pro", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static file serving ────────────────────────────────────
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


# ── Drawing session ────────────────────────────────────────

class DrawingSession:
    """
    All per-connection state: hand tracker, drawing engine,
    smoothers, debouncers, and mode.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.tracker  = HandTracker(max_hands=1)
        self.engine   = DrawingEngine(width, height)
        self.smoother = InterpolationSmoother(factor=5)
        self.debouncer = Debouncer(cooldown_frames=20)

        # Mode: idle | drawing | erasing | selection | pinch | paused
        self.mode: str = "idle"
        self.current_color: str = "red"

        # Dock buttons (rebuilt each frame)
        self.dock_buttons: list = []

        # FPS
        self._last_t = time.time()
        self.fps: float = 0.0

    # ── Helpers ────────────────────────────────────────────

    def tick_fps(self):
        now = time.time()
        dt = now - self._last_t
        self.fps = 1.0 / dt if dt > 0 else 0.0
        self._last_t = now

    def resize(self, w: int, h: int):
        self.engine.resize(w, h)

    def reset_canvas(self):
        self.engine.clear()

    def build_dock_buttons(self, w: int, h: int) -> list:
        colors  = ["red", "blue", "green", "yellow", "white", "eraser"]
        btn_r   = 28
        spacing = 72
        total_w = (len(colors) - 1) * spacing
        start_x = (w - total_w) // 2
        cy      = h - 55
        self.dock_buttons = [
            {"name": c, "cx": start_x + i * spacing, "cy": cy, "radius": btn_r}
            for i, c in enumerate(colors)
        ]
        return self.dock_buttons

    def check_hover(self, fx: int, fy: int) -> Optional[str]:
        for btn in self.dock_buttons:
            if math.hypot(fx - btn["cx"], fy - btn["cy"]) < HOVER_THRESHOLD:
                return btn["name"]
        return None


# ── Frame processing pipeline ──────────────────────────────

def process_frame(session: DrawingSession,
                  raw_bytes: bytes,
                  flip: bool = True) -> dict:
    """
    Decode → detect → classify gesture → update drawing engine → compose output.
    Returns dict with base64 JPEG + metadata.
    """
    frame = decode_frame(raw_bytes)
    if frame is None:
        return {"error": "bad_frame"}

    if flip:
        frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    session.resize(w, h)
    session.tick_fps()
    session.debouncer.tick()

    # ── Hand detection ─────────────────────────────────────
    frame = session.tracker.find_hands(frame, draw=False)
    landmarks = session.tracker.find_landmarks(frame)

    gesture    = "none"
    hovered    = None
    ix, iy     = None, None   # smoothed index fingertip

    if landmarks:
        fingers = session.tracker.fingers_up()
        # fingers = [thumb, index, middle, ring, pinky]
        thumb_up  = fingers[0]
        index_up  = fingers[1]
        middle_up = fingers[2]
        ring_up   = fingers[3]
        pinky_up  = fingers[4]

        # Raw index tip → smooth
        raw_ix, raw_iy = session.tracker.get_landmark(8)
        ix, iy = session.smoother.update(raw_ix, raw_iy)

        # Distances
        idx_mid_dist, _, _ = session.tracker.find_distance(8, 12)   # pinch
        thumb_idx_dist, _, _ = session.tracker.find_distance(4, 8)  # eraser size

        is_pinch    = idx_mid_dist < PINCH_THRESHOLD
        all_down    = not any(fingers[1:])
        only_index  = index_up and not middle_up and not ring_up and not pinky_up

        # ── Gesture classification ─────────────────────────

        if all_down:
            # ✊ Pause — lift finger to resume
            gesture = "paused"
            session.mode = "paused"
            session.engine.reset_stroke()
            session.smoother.reset()

        elif is_pinch and index_up and middle_up:
            # 🤏 Pinch — click dock button
            gesture = "pinch"
            session.mode = "pinch"
            session.engine.reset_stroke()

            session.build_dock_buttons(w, h)
            hovered = session.check_hover(ix, iy)
            if hovered and session.debouncer.trigger():
                session.current_color = hovered
                if hovered == "eraser":
                    session.engine.pen_color = None   # eraser mode
                else:
                    session.engine.pen_color = PEN_COLORS[hovered]
                    session.engine.pen_thickness = DEFAULT_PEN_THICKNESS

        elif index_up and middle_up and not is_pinch:
            # ✌️ Selection — hover over dock, no drawing
            gesture = "selection"
            session.mode = "selection"
            session.engine.reset_stroke()

            session.build_dock_buttons(w, h)
            hovered = session.check_hover(ix, iy)

        elif only_index:
            # ☝️ Draw or erase depending on selected tool
            if session.current_color == "eraser":
                gesture = "erasing"
                session.mode = "erasing"

                # Dynamic eraser size from thumb-index spread
                session.engine.set_eraser_size_from_pinch(thumb_idx_dist)
                session.engine.erase_rect(ix, iy)

            else:
                gesture = "drawing"
                session.mode = "drawing"
                session.engine.draw_stroke(ix, iy)

        else:
            gesture = "idle"
            session.mode = "idle"
            session.engine.reset_stroke()

    else:
        # No hand detected
        session.smoother.reset()
        session.engine.reset_stroke()
        session.mode = "idle"

    # ── Compose output ─────────────────────────────────────
    output = overlay_canvas(frame, session.engine.canvas)

    # Dock overlay
    session.build_dock_buttons(w, h)
    _draw_dock(output, session.dock_buttons, session.current_color, hovered)

    # Eraser preview rectangle (shown in selection AND erasing modes)
    if ix is not None and session.current_color == "eraser" and session.mode in ("erasing", "selection", "idle"):
        session.engine.draw_eraser_preview(output, ix, iy)

    # Cursor
    if ix is not None:
        pen_bgr = PEN_COLORS.get(session.current_color) or (255, 255, 255)
        draw_cursor(output, ix, iy, pen_bgr, session.mode)

    # HUD — FPS + mode
    _draw_hud(output, session.fps, session.mode, session.engine.eraser_w)

    encoded = encode_frame(output, quality=76)
    b64 = base64.b64encode(encoded).decode("utf-8")

    return {
        "frame":   b64,
        "gesture": gesture,
        "mode":    session.mode,
        "color":   session.current_color,
        "fps":     round(session.fps, 1),
        "hovered": hovered,
    }


# ── HUD ────────────────────────────────────────────────────

def _draw_hud(frame: np.ndarray, fps: float, mode: str, eraser_size: int):
    mode_color = {
        "drawing":   (0,   230, 100),
        "erasing":   (0,   180, 255),
        "selection": (220, 200, 0),
        "pinch":     (0,   230, 230),
        "paused":    (120, 120, 120),
        "idle":      (170, 170, 170),
    }.get(mode, (170, 170, 170))

    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, mode.upper(), (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1, cv2.LINE_AA)

    if mode == "erasing":
        cv2.putText(frame, f"Eraser: {eraser_size * 2}px", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 180, 255), 1, cv2.LINE_AA)


# ── Dock renderer ──────────────────────────────────────────

def _draw_dock(frame: np.ndarray, buttons: list,
               active_color: str, hovered: Optional[str]):
    if not buttons:
        return

    r   = buttons[0]["radius"]
    pad = 18
    xs  = [b["cx"] for b in buttons]
    x1  = min(xs) - r - pad
    x2  = max(xs) + r + pad
    y1  = buttons[0]["cy"] - r - pad
    y2  = min(frame.shape[0] - 1, buttons[0]["cy"] + r + pad)

    # Semi-transparent background pill
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (22, 24, 34), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 105), 1)

    COLOR_MAP = {
        "red":    (0,   0,   220),
        "blue":   (220, 80,  0),
        "green":  (0,   200, 60),
        "yellow": (0,   220, 220),
        "white":  (240, 240, 240),
        "eraser": (100, 100, 115),
    }

    for btn in buttons:
        name = btn["name"]
        cx, cy = btn["cx"], btn["cy"]
        br = btn["radius"]
        bgr = COLOR_MAP.get(name, (180, 180, 180))

        is_active  = (name == active_color)
        is_hovered = (name == hovered)

        if is_active:
            cy -= 10
            br += 4

        # Shadow
        cv2.circle(frame, (cx + 2, cy + 3), br, (8, 8, 8), -1)
        # Fill
        cv2.circle(frame, (cx, cy), br, bgr, -1)
        # Border
        border_col   = (255, 255, 255) if is_active else (150, 150, 150)
        border_thick = 3 if is_active else 1
        cv2.circle(frame, (cx, cy), br, border_col, border_thick)

        # Hover ring
        if is_hovered and not is_active:
            cv2.circle(frame, (cx, cy), br + 6, (255, 255, 255), 1)

        # Eraser X icon
        if name == "eraser":
            off = br // 2
            cv2.line(frame, (cx - off, cy - off), (cx + off, cy + off), (220, 220, 220), 2)
            cv2.line(frame, (cx + off, cy - off), (cx - off, cy + off), (220, 220, 220), 2)


# ── WebSocket endpoint ─────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = DrawingSession()
    logger.info("Client connected")

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            t   = msg.get("type")

            if t == "clear":
                session.reset_canvas()
                await websocket.send_text(json.dumps({"type": "cleared"}))

            elif t == "set_color":
                color = msg.get("color", "red")
                if color in PEN_COLORS:
                    session.current_color = color
                    session.engine.pen_color = PEN_COLORS[color]  # None for eraser

            elif t == "get_canvas":
                _, buf = cv2.imencode(".png", session.engine.canvas)
                b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                await websocket.send_text(json.dumps({"type": "canvas_data", "data": b64}))

            elif t == "eraser_size":
                # Frontend can send explicit size adjustments: {"type":"eraser_size","delta":10}
                delta = int(msg.get("delta", 0))
                session.engine.adjust_eraser_size(delta)

            elif t == "frame":
                data = msg.get("data", "")
                if not data:
                    continue
                raw_bytes = base64.b64decode(data)
                result = await asyncio.get_event_loop().run_in_executor(
                    None, process_frame, session, raw_bytes, True
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


# ── REST fallback ──────────────────────────────────────────

_rest_sessions: dict = {}

@app.post("/process-frame")
async def process_frame_rest(file: UploadFile = File(...),
                              session_id: str = "default"):
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
    return {"status": "ok", "service": "AirCanvas Pro v2"}
