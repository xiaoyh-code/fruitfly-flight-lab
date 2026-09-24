#!/usr/bin/env python3
"""User launcher for the local captured 3DGS viewer (no upload)."""
from pathlib import Path
import subprocess
import sys
import time
import webbrowser
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
URL="http://127.0.0.1:8766/"

def ready():
    try:
        with urlopen(URL,timeout=1) as reply:return reply.status==200
    except OSError:return False

if not ready():
    out=ROOT/"outputs/realistic";out.mkdir(parents=True,exist_ok=True)
    with (out/"viewer-server.log").open("a") as log:
        subprocess.Popen([sys.executable,str(ROOT/"scripts/serve_scene.py")],cwd=ROOT,
                         stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(30):
        if ready():break
        time.sleep(.1)
    else:raise SystemExit("Scene viewer did not start; see outputs/realistic/viewer-server.log")
webbrowser.open(URL)
print(URL)
