#!/usr/bin/env python3
"""Open the local live dashboard, starting its server if necessary."""
from pathlib import Path
import subprocess
import sys
import time
import webbrowser
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'http://127.0.0.1:8765'

def ready():
    try:
        with urllib.request.urlopen(URL + '/api/status', timeout=1) as reply:
            return reply.status == 200
    except OSError:
        return False

if not ready():
    logdir = ROOT / 'outputs/training'
    logdir.mkdir(parents=True, exist_ok=True)
    with (logdir / 'dashboard-server.log').open('a') as log:
        subprocess.Popen([sys.executable, str(ROOT/'scripts/serve_training.py')], cwd=ROOT,
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(30):
        if ready():
            break
        time.sleep(.1)
    else:
        raise SystemExit('Dashboard did not start; see outputs/training/dashboard-server.log')
webbrowser.open(URL)
print(URL)
