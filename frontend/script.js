/**
 * AirCanvas Pro — Frontend Controller
 * Webcam capture · WebSocket · frame rendering · UI
 */

'use strict';

console.log('✦ AirCanvas Pro script loaded');

// ── Config ─────────────────────────────────────────────────
const CONFIG = {
  // Auto-detect: if served from FastAPI on :8000 use that, else same host
  WS_URL: `ws://${location.hostname}:${location.port || 8000}/ws`,
  CAPTURE_FPS: 20,
  CAPTURE_W: 640,
  CAPTURE_H: 480,
  JPEG_QUALITY: 0.75,
  RECONNECT_DELAY: 3000,
  MAX_RECONNECTS: 6,
};

// ── DOM ────────────────────────────────────────────────────
const video         = document.getElementById('webcam');
const captureCanvas = document.getElementById('capture-canvas');
const displayCanvas = document.getElementById('display-canvas');
const placeholder   = document.getElementById('camera-placeholder');
const placeholderMsg= document.getElementById('placeholder-msg');
const statusDot     = document.getElementById('status-dot');
const statusText    = document.getElementById('status-text');
const statusMode    = document.getElementById('status-mode');
const statusFps     = document.getElementById('status-fps');
const statusColor   = document.getElementById('status-color');
const btnClear      = document.getElementById('btn-clear');
const btnSave       = document.getElementById('btn-save');
const btnHelp       = document.getElementById('btn-instructions');
const btnTheme      = document.getElementById('btn-theme');
const penDock       = document.getElementById('pen-dock');
const instrPanel    = document.getElementById('instructions-panel');
const instrBackdrop = document.getElementById('instructions-backdrop');
const closeInstr    = document.getElementById('close-instructions');
const toastBox      = document.getElementById('toast-container');

const captureCtx = captureCanvas.getContext('2d');
const displayCtx = displayCanvas.getContext('2d');

// ── State ──────────────────────────────────────────────────
let ws            = null;
let wsReady       = false;
let reconnects    = 0;
let captureTimer  = null;
let pendingFrame  = false;   // back-pressure: wait for ack before next send
let currentColor  = 'red';
let cameraReady   = false;

// ── Camera init ────────────────────────────────────────────

async function initCamera() {
  console.log('📷 Requesting camera…');

  // getUserMedia requires a secure context (localhost counts as secure)
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showPlaceholderError('Camera API not available. Use http://localhost:8000');
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width:     { ideal: CONFIG.CAPTURE_W },
        height:    { ideal: CONFIG.CAPTURE_H },
        facingMode: 'user',
        frameRate: { ideal: 30 },
      },
      audio: false,
    });

    video.srcObject = stream;

    // Wait for video to be ready
    video.onloadedmetadata = () => {
      video.play().then(() => {
        console.log('✅ Camera started');

        // Size canvases to actual video dimensions
        const vw = video.videoWidth  || CONFIG.CAPTURE_W;
        const vh = video.videoHeight || CONFIG.CAPTURE_H;

        captureCanvas.width  = vw;
        captureCanvas.height = vh;
        displayCanvas.width  = vw;
        displayCanvas.height = vh;

        cameraReady = true;
        hidePlaceholder();
        setStatus('connecting', 'Camera ready — connecting…');
        connectWS();
      });
    };

  } catch (err) {
    console.error('❌ Camera error:', err.name, err.message);

    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      showPlaceholderError('Camera permission denied. Click the camera icon in your browser address bar and allow access, then refresh.');
    } else if (err.name === 'NotFoundError') {
      showPlaceholderError('No camera found. Please connect a webcam and refresh.');
    } else {
      showPlaceholderError(`Camera error: ${err.message}`);
    }

    showToast('Camera access required — check browser permissions', 6000);
  }
}

function hidePlaceholder() {
  placeholder.classList.add('hidden');
}

function showPlaceholderError(msg) {
  placeholderMsg.textContent = msg;
  placeholder.classList.remove('hidden');
}

// ── WebSocket ──────────────────────────────────────────────

function connectWS() {
  console.log('🔌 Connecting WebSocket:', CONFIG.WS_URL);
  setStatus('connecting', 'Connecting to server…');

  try {
    ws = new WebSocket(CONFIG.WS_URL);
  } catch (e) {
    console.error('WS create failed:', e);
    scheduleReconnect();
    return;
  }

  ws.onopen = () => {
    console.log('✅ WebSocket connected');
    wsReady = true;
    reconnects = 0;
    setStatus('connected', 'Connected');
    showToast('Connected to AirCanvas Pro');
    startCapture();
  };

  ws.onmessage = (evt) => {
    try {
      handleMessage(JSON.parse(evt.data));
    } catch (e) {
      console.warn('WS parse error:', e);
    }
  };

  ws.onerror = () => {
    // onerror always fires before onclose — just log
    console.warn('⚠️ WebSocket error');
  };

  ws.onclose = (evt) => {
    console.log('🔌 WebSocket closed', evt.code);
    wsReady = false;
    pendingFrame = false;
    stopCapture();
    setStatus('disconnected', 'Disconnected');
    scheduleReconnect();
  };
}

function scheduleReconnect() {
  if (reconnects >= CONFIG.MAX_RECONNECTS) {
    setStatus('disconnected', 'Server unreachable — is the backend running?');
    showToast('Backend not found. Run: uvicorn main:app --port 8000', 7000);
    return;
  }
  reconnects++;
  const delay = Math.min(CONFIG.RECONNECT_DELAY * reconnects, 12000);
  setStatus('connecting', `Reconnecting in ${(delay / 1000).toFixed(0)}s… (${reconnects}/${CONFIG.MAX_RECONNECTS})`);
  setTimeout(connectWS, delay);
}

function handleMessage(msg) {
  switch (msg.type) {
    case 'frame':
      renderProcessedFrame(msg.frame);
      syncStatusBar(msg);
      pendingFrame = false;
      break;

    case 'cleared':
      showToast('Canvas cleared ✓');
      break;

    case 'canvas_data':
      triggerDownload(msg.data);
      break;

    case 'error':
      console.error('Server error:', msg.message);
      showToast(`Server: ${msg.message}`, 4000);
      pendingFrame = false;
      break;
  }
}

function wsSend(obj) {
  if (ws && wsReady && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
    return true;
  }
  return false;
}

// ── Capture loop ───────────────────────────────────────────

function startCapture() {
  if (captureTimer) return;
  const ms = Math.round(1000 / CONFIG.CAPTURE_FPS);
  captureTimer = setInterval(captureFrame, ms);
  console.log(`▶ Capture started @ ${CONFIG.CAPTURE_FPS} fps`);
}

function stopCapture() {
  if (captureTimer) {
    clearInterval(captureTimer);
    captureTimer = null;
  }
}

function captureFrame() {
  if (!wsReady || pendingFrame || !cameraReady) return;
  if (video.readyState < 2) return; // video not ready yet

  // Draw video frame onto capture canvas
  captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

  // Encode as JPEG blob → base64 → send
  captureCanvas.toBlob((blob) => {
    if (!blob) return;
    const reader = new FileReader();
    reader.onloadend = () => {
      const b64 = reader.result.split(',')[1];
      if (wsSend({ type: 'frame', data: b64 })) {
        pendingFrame = true;
      }
    };
    reader.readAsDataURL(blob);
  }, 'image/jpeg', CONFIG.JPEG_QUALITY);
}

// ── Frame rendering ────────────────────────────────────────

function renderProcessedFrame(b64) {
  const img = new Image();
  img.onload = () => {
    displayCtx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
    displayCtx.drawImage(img, 0, 0, displayCanvas.width, displayCanvas.height);
    // Make canvas visible on first frame
    if (!displayCanvas.classList.contains('visible')) {
      displayCanvas.classList.add('visible');
    }
  };
  img.src = `data:image/jpeg;base64,${b64}`;
}

// ── Status bar ─────────────────────────────────────────────

function setStatus(state, text) {
  statusDot.className = `status-dot ${state}`;
  statusText.textContent = text;
}

function syncStatusBar(msg) {
  if (msg.mode)  statusMode.textContent  = `Mode: ${msg.mode}`;
  if (msg.fps)   statusFps.textContent   = `FPS: ${msg.fps}`;
  if (msg.color) {
    statusColor.textContent = `Color: ${msg.color}`;
    if (msg.color !== currentColor) {
      setActiveColor(msg.color, false);
    }
  }
}

// ── Pen dock ───────────────────────────────────────────────

function setActiveColor(color, notify = true) {
  currentColor = color;
  document.querySelectorAll('.pen-btn').forEach((btn) => {
    const active = btn.dataset.color === color;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-pressed', String(active));
  });
  statusColor.textContent = `Color: ${color}`;
  if (notify) wsSend({ type: 'set_color', color });
}

penDock.addEventListener('click', (e) => {
  const btn = e.target.closest('.pen-btn');
  if (!btn || !btn.dataset.color) return;
  setActiveColor(btn.dataset.color);
  showToast(`${btn.dataset.color.charAt(0).toUpperCase() + btn.dataset.color.slice(1)} selected`);
});

// ── Keyboard shortcuts ─────────────────────────────────────

document.addEventListener('keydown', (e) => {
  // Don't fire when typing in inputs
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  const colorMap = {
    '1': 'red', '2': 'blue', '3': 'green',
    '4': 'yellow', '5': 'white',
    'e': 'eraser', 'E': 'eraser',
  };

  if (colorMap[e.key]) {
    setActiveColor(colorMap[e.key]);
    showToast(`${colorMap[e.key]} selected`);
    return;
  }

  // Eraser size: + / -
  if (e.key === '+' || e.key === '=') {
    wsSend({ type: 'eraser_size', delta: 10 });
    showToast('Eraser size +');
    return;
  }
  if (e.key === '-' || e.key === '_') {
    wsSend({ type: 'eraser_size', delta: -10 });
    showToast('Eraser size −');
    return;
  }

  if (e.key === 'Escape' && !instrPanel.hasAttribute('hidden')) {
    closeInstructions();
  }
});

// ── Navbar buttons ─────────────────────────────────────────

btnClear.addEventListener('click', () => {
  wsSend({ type: 'clear' });
  showToast('Canvas cleared');
});

btnSave.addEventListener('click', () => {
  if (!wsReady) {
    showToast('Not connected to server', 3000);
    return;
  }
  wsSend({ type: 'get_canvas' });
  showToast('Preparing download…');
});

btnHelp.addEventListener('click', openInstructions);
closeInstr.addEventListener('click', closeInstructions);
instrBackdrop.addEventListener('click', closeInstructions);

function openInstructions() {
  instrPanel.removeAttribute('hidden');
  instrBackdrop.removeAttribute('hidden');
}

function closeInstructions() {
  instrPanel.setAttribute('hidden', '');
  instrBackdrop.setAttribute('hidden', '');
}

// ── Theme toggle ───────────────────────────────────────────

btnTheme.addEventListener('click', () => {
  document.body.classList.toggle('light');
  const light = document.body.classList.contains('light');
  localStorage.setItem('aircanvas-theme', light ? 'light' : 'dark');
  showToast(light ? '☀️ Light mode' : '🌙 Dark mode');
});

// Restore saved theme on load
if (localStorage.getItem('aircanvas-theme') === 'light') {
  document.body.classList.add('light');
}

// ── Download helper ────────────────────────────────────────

function triggerDownload(b64png) {
  const a = document.createElement('a');
  a.href = `data:image/png;base64,${b64png}`;
  a.download = `aircanvas-${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.png`;
  a.click();
  showToast('Drawing saved ✓');
}

// ── Toast ──────────────────────────────────────────────────

function showToast(msg, duration = 2500) {
  const el = document.createElement('div');
  el.className = 'toast';
  el.textContent = msg;
  toastBox.appendChild(el);

  setTimeout(() => {
    el.classList.add('out');
    el.addEventListener('animationend', () => el.remove(), { once: true });
  }, duration);
}

// ── Boot ───────────────────────────────────────────────────

initCamera();
