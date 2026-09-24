#!/usr/bin/env python3
"""Loopback-only mission dashboard. Viewing never starts training."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/missions'
TASKS = {'landing', 'obstacles3', 'obstacles5', 'obstacles8', 'obstacle_landing', 'hall_avoidance'}
RUN_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z')


def read_json(path, fallback=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return fallback


def current_run():
    latest = read_json(OUTPUT / 'latest.json', {})
    ident = latest.get('run_id') if isinstance(latest, dict) else None
    if not isinstance(ident, str) or not RUN_ID.fullmatch(ident):
        return None, None
    folder = (OUTPUT / ident).resolve()
    if folder.parent != OUTPUT.resolve():
        return None, None
    return ident, folder


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if len(args) > 1 and str(args[1]).startswith(('4', '5')):
            super().log_message(fmt, *args)

    def send_headers(self, code, kind, size, extra=None):
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(size))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'same-origin')
        for key, value in (extra or {}).items():
            self.send_header(key, str(value))
        self.end_headers()

    def reply(self, code, value):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_headers(code, 'application/json; charset=utf-8', len(body))
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def file(self, path, kind, ranged=False):
        try:
            stream = path.open('rb')
        except OSError:
            return self.reply(404, {'error': 'File not available yet'})
        with stream:
            size = path.stat().st_size
            start, end, code = 0, size - 1, 200
            extra = {'Accept-Ranges': 'bytes'} if ranged else {}
            # BaseHTTPRequestHandler stores request headers on the instance.
            request_range = self.request_headers.get('Range') if ranged else None
            if request_range:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', request_range)
                if not match or not size or not any(match.groups()):
                    return self.reply(416, {'error': 'Invalid byte range'})
                a, b = match.groups()
                if a:
                    start = int(a)
                    end = min(int(b), size - 1) if b else size - 1
                else:
                    start = max(0, size - int(b))
                if start > end or start >= size:
                    return self.reply(416, {'error': 'Range outside file'})
                code = 206
                extra['Content-Range'] = f'bytes {start}-{end}/{size}'
            self.send_headers(code, kind, max(0, end - start + 1), extra)
            stream.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = stream.read(min(left, 1024 * 1024))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_GET(self):
        parsed = urlsplit(self.path)
        if parsed.path in ('/', '/missions'):
            return self.file(ROOT / 'dashboard/missions.html', 'text/html; charset=utf-8')
        if parsed.path == '/api/health':
            return self.reply(200, {'service': 'fruitfly-missions', 'training_started': False})
        ident, folder = current_run()
        if parsed.path == '/api/status':
            return self.reply(200, read_json(folder / 'status.json', {}) if folder else
                {'state': 'waiting', 'message': '未有任務紀錄；開啟呢個頁面唔會啟動訓練。', 'tasks': []})
        if parsed.path == '/api/history':
            return self.reply(200, read_json(folder / 'history.json', []) if folder else [])
        if parsed.path == '/api/frame' and folder:
            return self.file(folder / 'live.jpg', 'image/jpeg')
        if parsed.path == '/api/observer-meta' and folder:
            return self.file(folder / 'observer-preview.json', 'application/json; charset=utf-8')
        if parsed.path in ('/api/observer-frame', '/api/observer-video') and folder:
            view = parse_qs(parsed.query).get('view', ['follow'])[0]
            if view not in {'follow', 'overview'}:
                return self.reply(400, {'error': 'Unknown observer camera'})
            if parsed.path == '/api/observer-frame':
                return self.file(folder / f'observer-{view}.jpg', 'image/jpeg')
            return self.file(folder / f'observer-{view}.mp4', 'video/mp4', ranged=True)
        if parsed.path in ('/api/report', '/api/video', '/api/stale-vision'):
            task = parse_qs(parsed.query).get('task', [''])[0]
            if task not in TASKS:
                return self.reply(400, {'error': 'Unknown task'})
            if not folder:
                return self.reply(404, {'error': 'No mission run yet'})
            if parsed.path == '/api/stale-vision':
                if task != 'hall_avoidance':
                    return self.reply(400, {'error': 'Control report only available for hall_avoidance'})
                return self.file(folder / 'hall_avoidance-stale-vision.json', 'application/json; charset=utf-8')
            if parsed.path == '/api/report':
                return self.file(folder / f'{task}-audit.json', 'application/json; charset=utf-8')
            return self.file(folder / f'{task}-evaluation.mp4', 'video/mp4', ranged=True)
        return self.reply(404, {'error': 'Not found'})

    def do_POST(self):
        if urlsplit(self.path).path != '/api/control':
            return self.reply(404, {'error': 'Not found'})
        allowed = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
        if self.request_headers.get('Origin') not in allowed or self.request_headers.get('Sec-Fetch-Site') == 'cross-site':
            return self.reply(403, {'error': 'Only same-origin dashboard controls are allowed'})
        if self.request_headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
            return self.reply(415, {'error': 'Expected application/json'})
        try:
            length = int(self.request_headers.get('Content-Length', '0'))
            if not 0 < length <= 4096:
                raise ValueError()
            self.connection.settimeout(5)
            data = json.loads(self.rfile.read(length))
        except (ValueError, OSError):
            return self.reply(400, {'error': 'Invalid control request'})
        command = data.get('command', data.get('action')) if isinstance(data, dict) else None
        if command not in {'pause', 'resume', 'stop'}:
            return self.reply(400, {'error': 'Unknown action'})
        ident, folder = current_run()
        if not folder or data.get('run_id') != ident:
            return self.reply(409, {'error': 'Run changed or is unavailable; refresh first'})
        state = read_json(folder / 'status.json', {}).get('state')
        if state in {'completed', 'complete', 'stopped', 'error', 'failed'}:
            return self.reply(409, {'error': 'This run has already ended'})
        control = {'run_id': ident, 'command': command, 'timestamp': time.time()}
        try:
            with tempfile.NamedTemporaryFile('w', dir=folder, prefix='.control-', suffix='.json', delete=False) as handle:
                json.dump(control, handle)
                temporary = Path(handle.name)
            temporary.replace(folder / 'control.json')
        except OSError:
            return self.reply(500, {'error': 'Unable to write control request'})
        self.reply(202, {'accepted': True, **control})

    @property
    def request_headers(self):
        return self.headers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8769)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Mission dashboard: http://127.0.0.1:{args.port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
