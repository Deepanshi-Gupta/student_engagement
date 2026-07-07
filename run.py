"""
Single entry point for the whole project (Week 8).

    python run.py            # live: real camera + mic, dashboard at :8000
    python run.py --sim      # simulation: no hardware, synthetic signals
    python run.py --port 9000 --topic "thermodynamics"

Open http://127.0.0.1:8000 in your browser. Ctrl+C to stop.
"""

import os
import argparse


def main():
    p = argparse.ArgumentParser(description="Run the engagement-monitor dashboard.")
    p.add_argument("--sim", action="store_true",
                   help="run with synthetic signals, no camera/mic (great for demos)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--topic", default=None, help="lesson topic given to Gemini for context")
    args = p.parse_args()

    # These are read by server.py / orchestrator.py at startup.
    os.environ["MONITOR_MODE"] = "sim" if args.sim else "live"
    if args.topic:
        os.environ["MONITOR_TOPIC"] = args.topic

    import uvicorn
    print(f"Dashboard:  http://{args.host}:{args.port}   (mode: {os.environ['MONITOR_MODE']})")
    uvicorn.run(
        "server:app", host=args.host, port=args.port, reload=False,
        # The /video endpoint is an endless MJPEG stream. On Ctrl+C uvicorn
        # waits for in-flight responses to finish BEFORE running lifespan
        # shutdown (where monitor.stop() sets _stop) — so without a bound it
        # waits forever on that stream. Force-close lingering connections
        # after 5s so a single Ctrl+C always exits cleanly.
        timeout_graceful_shutdown=5,
    )


if __name__ == "__main__":
    main()
