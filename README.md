# AirCanvas Pro

Draw in the air using hand gestures via your webcam.

## Stack

- **Backend**: Python · FastAPI · OpenCV · MediaPipe
- **Frontend**: HTML · CSS · Vanilla JS
- **Transport**: WebSocket (real-time bidirectional)

## Project Structure

```
aircanvas-pro/
├── backend/
│   ├── main.py           # FastAPI app, WebSocket endpoint, frame pipeline
│   ├── hand_tracking.py  # MediaPipe hand tracker + gesture utilities
│   └── utils.py          # Smoothing, debounce, color map, frame helpers
├── frontend/
│   ├── index.html        # App shell + dock + instructions panel
│   ├── style.css         # Glassmorphism dark UI + animations
│   └── script.js         # Webcam capture, WS client, rendering
├── requirements.txt
└── README.md
```

## Setup & Run

### 1. Install dependencies

```bash
cd aircanvas-pro
pip install -r requirements.txt
```

> MediaPipe requires Python 3.8–3.11. Use a virtual environment:
> ```bash
> python -m venv venv
> source venv/bin/activate   # Windows: venv\Scripts\activate
> pip install -r requirements.txt
> ```

### 2. Start the backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open the frontend

Navigate to `http://localhost:8000` in Chrome.

The backend serves the frontend automatically from the `/frontend` directory.

---

## Gesture Reference

| Gesture | Action |
|---|---|
| ☝️ Index finger up | Drawing mode |
| ✌️ Index + Middle up | Selection / hover mode |
| 🤏 Pinch (index + middle close) | Click / select tool |
| ✊ All fingers down | Pause drawing |

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `1` | Red pen |
| `2` | Blue pen |
| `3` | Green pen |
| `4` | Yellow pen |
| `5` | White pen |
| `E` | Eraser |

## Deployment

### Backend (Render / Railway)

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Set `CORS` origins to your frontend domain in `main.py`.

### Frontend (Vercel / Netlify)

Deploy the `frontend/` folder as a static site.  
Update `CONFIG.WS_URL` in `script.js` to point to your deployed backend (`wss://your-backend.com/ws`).

> **Note**: Webcam access requires HTTPS in production. Both Vercel and Render provide HTTPS by default.

## Tips for Best Results

- Use in a well-lit environment
- Keep your hand within the camera frame
- Avoid fast jerky movements — the smoothing filter works best with steady motion
- Chrome is recommended for best WebSocket + webcam performance
