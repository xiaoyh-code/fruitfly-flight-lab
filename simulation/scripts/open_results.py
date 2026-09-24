#!/usr/bin/env python3
"""Start local viewing services and open the saved scene/results explorer."""
from pathlib import Path
import argparse
import subprocess
import sys
import time
import webbrowser
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8768/explore"


def ready(url):
    try:
        with urlopen(url, timeout=1) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def ensure_server(script, endpoint):
    if ready(endpoint):
        return
    logs = ROOT / "outputs/realistic"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / (Path(script).stem + ".log")).open("a") as log:
        process = subprocess.Popen([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(40):
        if ready(endpoint):
            return
        if process.poll() is not None:
            break
        time.sleep(.1)
    raise SystemExit(f"無法啟動 {script}，請查看 {logs / (Path(script).stem + '.log')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true", help="Start viewing services without opening a browser")
    args = parser.parse_args()
    for script, endpoint in (
        ("serve_scene.py", "http://127.0.0.1:8766/"),
        ("serve_render_bridge.py", "http://127.0.0.1:8768/health"),
        ("serve_training.py", "http://127.0.0.1:8765/api/status"),
    ):
        ensure_server(script, endpoint)
    if not ready(URL):
        raise SystemExit("現有 8768 服務未載入新總覽頁；請重新啟動 scripts/serve_render_bridge.py。")
    if not args.no_browser:
        webbrowser.open(URL)
    print(f"場景與結果：{URL}\n只開啟瀏覽服務；不會重新訓練或下載場景。")


if __name__ == "__main__":
    main()
