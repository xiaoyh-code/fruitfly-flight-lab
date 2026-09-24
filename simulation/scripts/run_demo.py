#!/usr/bin/env python3
"""Run a ground-truth navigation baseline and save actual simulated FPV frames.

On macOS use .venv/bin/mjpython scripts/run_demo.py --viewer for a live window.
Physics-only is the default. --video and --png render offscreen without a viewer.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fruitfly_sim import FruitFlyDroneEnv, waypoint_velocity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--viewer", action="store_true", help="Open MuJoCo's interactive window (mjpython on Mac)")
    mode.add_argument("--headless", action="store_true", help="Run without the interactive window (default)")
    parser.add_argument("--hold-after-route", action="store_true",
                        help="With --viewer, hover after returning until the window is closed")
    parser.add_argument("--camera", choices=["fpv", "overview", "track"], default="fpv", help="Initial live viewer camera")
    parser.add_argument("--seconds", type=float, default=32.0, help="Maximum simulated time")
    parser.add_argument("--fps", type=int, default=20, help="Video frames per simulated second")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", type=Path, help="MP4 or GIF output, FPV left and overview right")
    parser.add_argument("--png", type=Path, help="Save a representative FPV + overview image")
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs" / "demo_summary.json")
    args = parser.parse_args()
    if not (0 < args.fps <= 50) or args.seconds <= 0:
        parser.error("fps must be in [1,50] and seconds must be positive")
    if args.hold_after_route and not args.viewer:
        parser.error("--hold-after-route requires --viewer")
    env = FruitFlyDroneEnv(width=args.width, height=args.height, episode_seconds=args.seconds)
    writer = None
    viewer_context = nullcontext(None)
    if args.viewer:
        import mujoco.viewer
        viewer_context = mujoco.viewer.launch_passive(env.model, env.data)
    if args.video or args.png:
        import imageio.v2 as imageio
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        if args.video.suffix.lower() == ".gif":
            writer = imageio.get_writer(args.video, mode="I", duration=1000 / args.fps, loop=0)
        else:
            writer = imageio.get_writer(args.video, fps=args.fps, codec="libx264", quality=7, macro_block_size=1)
    route = [np.array(point, float) for point in [(0, 0, 1.2), (3, 0, 1.2), (3, 3, 1.2), (0, 3, 1.2), (0, 0, 1.2)]]
    waypoint_index = 0
    reached_count = 0
    collision = False
    completed = False
    route_finished_time = None
    last_frame = None
    next_frame_time = 0.0
    path = []
    start = time.monotonic()
    try:
        obs, info = env.reset(seed=args.seed)
        env.set_goal(route[0])
        with viewer_context as viewer:
            if viewer:
                import mujoco
                with viewer.lock():
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = env.model.camera(args.camera).id
            while env.data.time < args.seconds or (viewer and args.hold_after_route and completed):
                if viewer and not viewer.is_running():
                    break
                tick_start = time.monotonic()
                command = waypoint_velocity(obs["state"], route[waypoint_index])
                obs, reward, terminated, truncated, info = env.step(command)
                if not completed:
                    path.append([info["time"], *map(float, obs["state"][:3])])
                distance = np.linalg.norm(obs["state"][:3] - route[waypoint_index])
                if not completed and distance < 0.13 and np.linalg.norm(obs["state"][7:10]) < 0.25:
                    reached_count += 1
                    print(f"Waypoint {reached_count}/{len(route)} reached at t={info['time']:.1f}s", flush=True)
                    if waypoint_index == len(route) - 1:
                        completed = True
                        route_finished_time = info["time"]
                        if args.hold_after_route:
                            env.max_steps = 2 ** 60
                            print("Returned home. Hovering until you close the viewer window.", flush=True)
                    else:
                        waypoint_index += 1
                        env.set_goal(route[waypoint_index])
                recording_route = route_finished_time is None or info["time"] <= route_finished_time
                if recording_route and (writer or args.png) and info["time"] >= next_frame_time:
                    last_frame = np.concatenate((env.render("fpv"), env.render("overview")), axis=1)
                    if writer:
                        writer.append_data(last_frame)
                    # Mid-route image is more informative than an end-of-flight image.
                    if args.png and (not args.png.exists() or 5.0 <= info["time"] < 5.0 + env.dt):
                        args.png.parent.mkdir(parents=True, exist_ok=True)
                        imageio.imwrite(args.png, last_frame)
                    next_frame_time += 1.0 / args.fps
                if viewer:
                    viewer.sync()
                    time.sleep(max(0.0, env.dt - (time.monotonic() - tick_start)))
                if terminated or truncated or (completed and not args.hold_after_route):
                    collision = bool(info.get("collision", False))
                    break
        summary = {
            "controller": "ground-truth waypoint baseline with geometric PD stabilization",
            "model": "official Menagerie Crazyflie 2 with in-memory SI-moment actuator adaptation",
            "uses_flyvis": False, "uses_gaussian_splatting": False,
            "route_completed": completed, "waypoints_reached": reached_count,
            "waypoints_total": len(route), "collision": collision,
            "simulated_seconds": float(env.data.time), "wall_seconds": time.monotonic() - start,
            "route_completed_at_seconds": route_finished_time,
            "final_position_m": env.data.qpos[:3].tolist(),
            "return_error_m": float(np.linalg.norm(env.data.qpos[:3] - route[-1])),
            "action_units": ["world vx m/s", "world vy m/s", "world vz m/s", "yaw rate rad/s"],
            "trajectory_time_xyz": path,
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding='utf-8')
        print(json.dumps({k: v for k, v in summary.items() if k != "trajectory_time_xyz"}, indent=2))
        return 0 if completed and not collision else 1
    finally:
        if writer:
            writer.close()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
