#!/usr/bin/env python3
"""Diagnose an installation, with optional real rendering and Flyvis probes.

This entry point uses only Python's standard library so it can explain missing
dependencies. Run with the SAME interpreter used for training. Exit status:
0 = requested checks pass; 1 = a check failed; 2 = invalid command-line usage.
By default only the core simulation profile is required; --flyvis also requires
the vision/training profile. Optional assets not requested are warnings. Passing on one machine does not
establish Windows support for every GPU/driver combination.
"""
from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
CORE_PACKAGES = (
    ("numpy", "numpy"), ("gymnasium", "gymnasium"), ("mujoco", "mujoco"),
    ("opencv-python", "cv2"), ("pillow", "PIL"), ("imageio", "imageio"),
    ("imageio-ffmpeg", "imageio_ffmpeg"), ("pytest", "pytest"),
)
VISION_PACKAGES = (
    ("torch", "torch"), ("torchvision", "torchvision"), ("flyvis", "flyvis"),
    ("stable-baselines3", "stable_baselines3"), ("pandas", "pandas"),
)


def probe(label: str, command: list[str], *, timeout: int = 120) -> bool:
    print(f"\n[{label}]", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("FLYVIS_ROOT_DIR", str(ROOT / "data/flyvis"))
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    env.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache/numba"))
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                encoding="utf-8", errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: {label}: {error}")
        return False
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    print(f"{'PASS' if result.returncode == 0 else 'FAIL'}: {label} (exit {result.returncode})")
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Also render one real MuJoCo RGB frame (needs working OpenGL drivers)")
    parser.add_argument("--flyvis", action="store_true", help="Also require the vision/training packages, load official weights, and process two RGB images on CPU")
    args = parser.parse_args()
    print("FruitFly installation doctor")
    print(f"Interpreter: {sys.executable}\nPython: {platform.python_version()}\n"
          f"System: {platform.platform()} ({platform.machine()})\nProject: {ROOT}")
    failures = 0
    if sys.version_info[:2] != (3, 12):
        print("WARN: this project was tested with Python 3.12; use a Python 3.12 virtual environment.")
    if sys.prefix == sys.base_prefix:
        print("WARN: no virtual environment detected; dependencies may belong to another interpreter.")

    packages = CORE_PACKAGES + (VISION_PACKAGES if args.flyvis else ())
    print(f"\n[Required package versions: {'core + vision' if args.flyvis else 'core'}]")
    missing = []
    for distribution, _ in packages:
        try:
            print(f"  {distribution}: {version(distribution)}")
        except PackageNotFoundError:
            missing.append(distribution)
            print(f"  MISSING: {distribution}")
    if missing:
        failures += 1
        print("FAIL: install the project's requirements with this interpreter, then rerun doctor.")
    if not args.flyvis:
        print("\n[Optional vision/training packages: not imported without --flyvis]")
        for distribution, _ in VISION_PACKAGES:
            try:
                print(f"  {distribution}: {version(distribution)}")
            except PackageNotFoundError:
                print(f"  NOT INSTALLED (optional): {distribution}")

    import_code = "import importlib\nfailed = []\n"
    import_code += f"for name in {tuple(module for _, module in packages)!r}:\n"
    import_code += ("    try:\n        importlib.import_module(name)\n        print('OK import:', name)\n"
                    "    except Exception as error:\n        failed.append(name)\n"
                    "        print('FAIL import:', name, type(error).__name__, str(error))\n"
                    "raise SystemExit(bool(failed))\n")
    if not probe("Dependency imports", [sys.executable, "-c", import_code]):
        failures += 1
        print("Hint: import errors can indicate incompatible NumPy/Pandas versions, a Torch/Torchvision "
              "mismatch, or missing Windows DLLs. Follow the README installation steps in a fresh environment.")

    physics_code = textwrap.dedent("""\
        import numpy as np
        from fruitfly_sim.env import FruitFlyDroneEnv
        env = FruitFlyDroneEnv()
        try:
            observation, info = env.reset(seed=0)
            for _ in range(5):
                observation, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            assert np.isfinite(observation['state']).all(), 'Non-finite physics state'
            print('Crazyflie XML and meshes loaded; five physics steps; state is finite.')
        finally:
            env.close()
    """)
    if not probe("MuJoCo physics and model assets", [sys.executable, "-c", physics_code]):
        failures += 1
        print("Hint: check models/bitcraze_crazyflie_2/cf2.xml and its assets directory are present.")

    weights = ROOT / "data/flyvis/results/flow/0000/000"
    print("\n[Optional assets]")
    print(f"{'FOUND' if weights.is_dir() else 'WARN missing'}: official Flyvis model {weights}")
    if not weights.is_dir():
        print("  Download separately: python scripts/download_flyvis_models.py")
    hall = ROOT / "models/gaussian/robot_hall.ply"
    print(f"{'FOUND' if hall.is_file() else 'WARN missing'}: Hall scene {hall}")
    print("  Hall verification/download: python scripts/download_hall_scene.py [--check-only]")

    if args.render:
        render_code = textwrap.dedent("""\
            import numpy as np
            from fruitfly_sim.env import FruitFlyDroneEnv
            env = FruitFlyDroneEnv(width=160, height=120, render_mode='rgb_array')
            try:
                env.reset(seed=0)
                rgb = env.render()
                assert rgb.shape == (120, 160, 3) and rgb.dtype == np.uint8, (rgb.shape, rgb.dtype)
                assert int(rgb.max()) > int(rgb.min()), 'Rendered frame is uniform'
                print('Real MuJoCo camera frame:', rgb.shape, rgb.dtype, 'non-uniform=True')
            finally:
                env.close()
        """)
        if not probe("MuJoCo OpenGL render", [sys.executable, "-c", render_code]):
            failures += 1
            print("Hint: use a local desktop session and current GPU drivers. A remote/headless session may "
                  "lack OpenGL. Do not copy macOS-only MUJOCO_GL settings to Windows.")
    if args.flyvis:
        if not probe("Flyvis RGB example", [sys.executable, str(ROOT / "scripts/flyvis_example.py")], timeout=180):
            failures += 1

    print(f"\n{'PASS' if failures == 0 else 'FAIL'}: {failures} failed stage(s). "
          "Optional unrequested assets do not fail this check.")
    print("This checks only this computer; the WebGL Hall camera bridge is a separate browser step.")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
