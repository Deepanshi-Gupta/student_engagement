"""
Week 8 — FastAPI server (the display layer)
------------------------------------------------
Serves the dashboard and exposes the SessionStore over HTTP. This process
starts the EngagementMonitor (orchestrator) on startup and stops it on
shutdown — so `python run.py` is the single command that boots the whole
project.

Endpoints
  GET /                  the dashboard (static/dashboard.html)
  GET /api/state         latest signals + fusion score + observation + history (JSON)
  GET /api/report        session summary stats (JSON)
  GET /video             live annotated camera feed (MJPEG; live mode only)
  POST /api/session/stop stop the monitor threads

Mode is chosen by the MONITOR_MODE env var ("live" default, or "sim").
"""

import os
import time
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import (FileResponse, JSONResponse, StreamingResponse,
                               PlainTextResponse)

from session_store import SessionStore
from orchestrator import EngagementMonitor

HERE = os.path.dirname(os.path.abspath(__file__))
DASHBOARD = os.path.join(HERE, "static", "dashboard.html")

MODE = os.getenv("MONITOR_MODE", "live").lower()

# Created in the lifespan handler so they share one process.
store: SessionStore = None
monitor: EngagementMonitor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, monitor
    store = SessionStore(mode=MODE)
    monitor = EngagementMonitor(store, mode=MODE)
    monitor.start()
    print(f"[server] EngagementMonitor started in '{MODE}' mode.")
    try:
        yield
    finally:
        monitor.stop()
        print("[server] EngagementMonitor stopped.")


app = FastAPI(title="Engagement Monitor", lifespan=lifespan)


@app.get("/")
def index():
    if not os.path.exists(DASHBOARD):
        return PlainTextResponse("dashboard.html missing in static/", status_code=500)
    return FileResponse(DASHBOARD)


@app.get("/api/state")
def api_state():
    return JSONResponse(store.state_dict())


@app.get("/api/report")
def api_report():
    return JSONResponse(store.report_dict())


@app.post("/api/session/stop")
def api_stop():
    monitor.stop()
    return JSONResponse({"running": False})


def _mjpeg():
    """Yield the latest annotated frame as a multipart MJPEG stream.

    Exits when the monitor is stopped (e.g. POST /api/session/stop) so the
    stream doesn't outlive the thing producing frames. The hard Ctrl+C case
    is handled by uvicorn's timeout_graceful_shutdown in run.py — see note
    there for why a _stop check alone can't break the shutdown wait."""
    boundary = b"--frame"
    while monitor is None or not monitor._stop.is_set():
        frame = store.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        yield (boundary + b"\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.05)  # ~20 fps cap


@app.get("/video")
def video():
    return StreamingResponse(_mjpeg(),
                             media_type="multipart/x-mixed-replace; boundary=frame")
