#!/usr/bin/env python3
"""Open live/saved mission progress without starting or restarting training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import webbrowser
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
URL = 'http://127.0.0.1:8769/'


def ready():
    try:
        with urlopen(URL + 'api/health', timeout=1) as response:
            return json.load(response).get('service') == 'fruitfly-missions'
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not ready():
        log_dir = ROOT / 'outputs/missions'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / 'server.log'
        with log_path.open('a') as log:
            process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/serve_missions.py')], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(50):
            if ready():
                break
            if process.poll() is not None:
                break
            time.sleep(.1)
        if not ready():
            raise SystemExit(f'監察服務未能啟動；請查看 {log_path}')
    if not args.no_browser:
        webbrowser.open(URL)
    print(f'任務訓練畫面：{URL}\n只開啟監察服務；唔會開始或重跑訓練。')


if __name__ == '__main__':
    main()
