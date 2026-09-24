#!/usr/bin/env python3
"""Create a local Python 3.12 environment, then download verified research assets."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('core', 'vision', 'hall'), default='core')
    parser.add_argument('--skip-downloads', action='store_true', help='Install packages only; no asset/model downloads')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error('Use Python 3.12 (recorded version: 3.12.10). Windows: py -3.12 scripts/setup_student.py')
    if sys.maxsize <= 2**32:
        parser.error('64-bit Python is required. Install the x86-64 installer on Intel/AMD Windows.')
    if sys.platform == 'win32' and platform.machine().lower() not in ('amd64', 'x86_64'):
        parser.error('The student Windows profile targets x64. Native Windows ARM64 is not validated.')
    target = ROOT / '.venv'
    python = target / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if target.exists() and not python.is_file():
        parser.error('Existing .venv is incomplete or belongs to another OS. Rename it, then rerun setup.')
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(target)
    environment = dict(os.environ, PYTHONUTF8='1', PIP_DISABLE_PIP_VERSION_CHECK='1')

    def run(*arguments):
        subprocess.run([str(python), '-X', 'utf8', *arguments], cwd=ROOT, env=environment, check=True)

    run('-c', 'import sys; assert sys.version_info[:2] == (3,12), "Recreate .venv using Python 3.12"')
    run('-m', 'pip', 'install', '--upgrade', 'pip')
    requirements = 'requirements-core.txt' if args.profile == 'core' else 'requirements-vision.txt'
    run('-m', 'pip', 'install', '-r', requirements)
    run('-m', 'pip', 'check')
    if not args.skip_downloads:
        run('scripts/download_drone_model.py')
        if args.profile in ('vision', 'hall'):
            run('scripts/download_flyvis_models.py')
        if args.profile == 'hall':
            run('scripts/download_hall_scene.py')
            run('scripts/restore_public_evidence.py')
    print(f'Profile {args.profile} installed. Interpreter: {python}')
    print('Next: run scripts/doctor.py; add --render and --flyvis as separate checks when ready.')
    print('Setup does not start training. Keep camera-bridge and training terminals separate.')


if __name__ == '__main__':
    main()
