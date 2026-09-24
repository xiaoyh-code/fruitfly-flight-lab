#!/usr/bin/env python3
"""Loopback-only, dependency-free FruitFly training dashboard server."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "outputs" / "training"


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "FruitFlyDashboard/1.0"

    def log_message(self, fmt, *args):
        # One-second camera/status polling should not fill the training log.
        if getattr(self, "path", "").startswith("/api/"):
            return
        super().log_message(fmt, *args)

    def _send(self, body: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, value, status=200):
        self._send(json.dumps(value, ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8", status)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        route = urlsplit(self.path).path
        if route in ("/", "/index.html"):
            self._send((ROOT / "dashboard" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif route in ("/api/status", "/api/history"):
            file = TRAINING / ("status.json" if route.endswith("status") else "history.json")
            fallback = {"state": "initializing", "message": "等候訓練程序連線…", "milestones": [], "log": []} if route.endswith("status") else []
            try:
                value = json.loads(file.read_text(encoding='utf-8'))
                self._json(value)
            except FileNotFoundError:
                self._json(fallback)
            except (json.JSONDecodeError, ValueError):
                self._json({"error": "Training snapshot is being updated."}, 503)
        elif route in ("/api/frame", "/api/preview"):
            file = TRAINING / ("live.jpg" if route.endswith("frame") else "preview.mp4")
            try:
                self._send(file.read_bytes(), "image/jpeg" if route.endswith("frame") else "video/mp4")
            except FileNotFoundError:
                self._json({"error": "No evaluation frame yet."}, 404)
        elif route == "/favicon.ico":
            self._send(b"", "image/x-icon", 204)
        else:
            self._json({"error": "Unknown endpoint."}, 404)

    def do_POST(self):
        if urlsplit(self.path).path != "/api/control":
            self._json({"error": "Unknown endpoint."}, 404)
            return
        # Reject cross-origin browser writes even though the listener is local.
        origin = self.headers.get("Origin")
        allowed = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
        if origin and origin not in allowed:
            self._json({"error": "Local dashboard origin required."}, 403)
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self._json({"error": "JSON body required."}, 415)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("Invalid body size.")
            request = json.loads(self.rfile.read(length))
            command = request.get("command")
            if command not in ("pause", "resume", "stop"):
                raise ValueError("Command must be pause, resume, or stop.")
        except (ValueError, AttributeError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)
            return
        TRAINING.mkdir(parents=True, exist_ok=True)
        payload = {"command": command, "timestamp": time.time()}
        descriptor, temp_path = tempfile.mkstemp(prefix=".control-", suffix=".json", dir=TRAINING)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(payload, stream)
            os.replace(temp_path, TRAINING / "control.json")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        self._json({"accepted": True, **payload}, 202)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    print(f"FruitFly live training: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
