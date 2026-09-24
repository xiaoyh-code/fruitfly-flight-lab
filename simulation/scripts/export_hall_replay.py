#!/usr/bin/env python3
"""Export measured Hall audit trajectories for a local 3DGS replay.

This does not rerun the policy or certify the captured hall's surfaces. The new
clearance calculation covers linear interpolation between recorded positions,
against the same single approximate cylinder used by SplatNavigationEnv.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile


ROOT = Path(__file__).resolve().parents[1]
# SplatNavigationEnv constants, in metres; Z is up in both coordinate systems.
SCENE_OFFSET = (15.15, -3.15, 0.0)
PROXY_CENTER = (0.0, 0.2)
PROXY_RADIUS = 0.45
PROXY_HEIGHT = 1.55
VEHICLE_RADIUS = 0.09
NOMINAL_DT = 0.1  # 5 navigation substeps × 10 physics ticks × 0.002 s.
PHYSICS_DT = 0.002
ALTITUDE = 1.2
TOLERANCE = 1e-7


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def boolean(value, label):
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def vector(value, size, label):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must contain {size} coordinates")
    return [number(item, label) for item in value]


def near(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=TOLERANCE):
        raise ValueError(f"{label} disagrees with the recorded trajectory ({actual} vs {expected})")


def translated(point):
    return [value + shift for value, shift in zip(point, SCENE_OFFSET)]


def clearance_metrics(points):
    """Exact XY circle clearance of a piecewise-linear position replay.

    The .09 m disc is a conservative horizontal vehicle envelope. No Z clipping
    is applied, matching the source's sampled horizontal metric; this is not a
    signed 3D distance to the finite-height cylinder or a physics contact test.
    """
    combined_radius = PROXY_RADIUS + VEHICLE_RADIUS
    sampled = min(math.hypot(p[0] - PROXY_CENTER[0], p[1] - PROXY_CENTER[1])
                  - combined_radius for p in points)
    closest = (math.inf, 0, 0.0)
    for index, (a, b) in enumerate(zip(points, points[1:])):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length_squared = dx * dx + dy * dy
        fraction = 0.0 if length_squared == 0 else max(0.0, min(1.0,
            ((PROXY_CENTER[0] - a[0]) * dx + (PROXY_CENTER[1] - a[1]) * dy) / length_squared))
        distance = math.hypot(a[0] + fraction * dx - PROXY_CENTER[0],
                              a[1] + fraction * dy - PROXY_CENTER[1]) - combined_radius
        if distance < closest[0]:
            closest = (distance, index, fraction)
    return dict(recomputed_sampled_clearance_m=sampled,
                swept_proxy_clearance_m=closest[0],
                closest_segment_index=closest[1], closest_segment_fraction=closest[2])


def episode_times(row, steps, seconds):
    # All completed nonterminal steps are .1 s. Collision can stop on any .002 s
    # physics tick inside a .02 s control substep; do not stretch earlier steps.
    last_dt = seconds - (steps - 1) * NOMINAL_DT
    if not PHYSICS_DT - TOLERANCE <= last_dt <= NOMINAL_DT + TOLERANCE:
        raise ValueError("episode seconds are inconsistent with the 0.1 s navigation timestep")
    near(last_dt / PHYSICS_DT, round(last_dt / PHYSICS_DT), "terminal physics timestep")
    inferred = [index * NOMINAL_DT for index in range(steps)] + [seconds]
    if "trajectory_times_s" not in row:
        return inferred, "0.1 s control cadence; recorded terminal time"
    times = vector(row["trajectory_times_s"], steps + 1, "trajectory_times_s")
    for actual, expected in zip(times, inferred):
        near(actual, expected, "recorded timestamp")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("trajectory timestamps must increase strictly")
    return times, "recorded MuJoCo timestamps"


def export_episode(row):
    if not isinstance(row, dict):
        raise ValueError("episode must be an object")
    seed = integer(row.get("seed"), "seed")
    steps = integer(row.get("steps"), "steps", 1)
    seconds = number(row.get("seconds"), "seconds")
    points = row.get("trajectory_xyz")
    if not isinstance(points, list) or len(points) != steps + 1:
        raise ValueError("trajectory_xyz must include the reset pose and every step")
    points = [vector(point, 3, "trajectory_xyz") for point in points]
    goal = vector(row.get("goal"), 2, "goal")
    success, collision = boolean(row.get("success"), "success"), boolean(row.get("collision"), "collision")
    if success and collision:
        raise ValueError("an episode cannot report both success and collision")
    reason = row.get("failure_reason")
    if not isinstance(reason, str) or (success and reason) or (not success and not reason):
        raise ValueError("failure_reason must agree with the success flag")
    metrics = clearance_metrics(points)
    metrics["source_sampled_clearance_m"] = number(row.get("min_clearance"), "min_clearance")
    near(metrics["source_sampled_clearance_m"], metrics["recomputed_sampled_clearance_m"], "sampled clearance")
    if "min_polyline_clearance" in row:
        metrics["source_polyline_clearance_m"] = number(row["min_polyline_clearance"], "min_polyline_clearance")
        near(metrics["source_polyline_clearance_m"], metrics["swept_proxy_clearance_m"], "polyline clearance")
    final_distance = math.hypot(points[-1][0] - goal[0], points[-1][1] - goal[1])
    near(number(row.get("final_distance"), "final_distance"), final_distance, "final goal distance")
    path_length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    near(number(row.get("path_length_m"), "path_length_m"), path_length, "path length")
    if success and final_distance >= .25 + TOLERANCE:
        raise ValueError("success is inconsistent with the 0.25 m goal tolerance")
    times, timing_source = episode_times(row, steps, seconds)
    result = dict(seed=seed, success=success, collision=collision, steps=steps,
        seconds=seconds, failure_reason=reason, start_xyz=translated(points[0]),
        goal_xyz=translated([*goal, ALTITUDE]), trajectory_xyz=[translated(p) for p in points],
        times_s=times, timing_source=timing_source, orientation_recorded=False,
        metrics={**metrics, "final_distance_m": final_distance, "path_length_m": path_length},
        proxy_clear=metrics["swept_proxy_clearance_m"] > TOLERANCE)
    if "trajectory_quat_wxyz" in row:
        quaternions = row["trajectory_quat_wxyz"]
        if not isinstance(quaternions, list) or len(quaternions) != len(points):
            raise ValueError("trajectory_quat_wxyz must match trajectory_xyz length")
        quaternions = [vector(q, 4, "trajectory_quat_wxyz") for q in quaternions]
        for quaternion in quaternions:
            near(math.sqrt(sum(component * component for component in quaternion)), 1.0, "quaternion norm")
        result["trajectory_quat_xyzw"] = [[x, y, z, w] for w, x, y, z in quaternions]
        result["orientation_recorded"] = True
    return result


def build_replay(report, audit_sha256):
    if not isinstance(report, dict) or report.get("task") != "hall_avoidance" or report.get("kind") != "audit":
        raise ValueError("expected a completed hall_avoidance audit report")
    run_id = report.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("invalid audit run_id")
    for flag, expected in (("stale_vision", False), ("teacher_used_in_evaluation", False),
                           ("gaussian_scene_used", True), ("obstacle_geometry_in_policy", False)):
        if report.get(flag) is not expected:
            raise ValueError(f"audit {flag} must be {expected}")
    rows = report.get("episodes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("audit must contain episodes")
    episodes = [export_episode(row) for row in rows]
    if len({row["seed"] for row in episodes}) != len(episodes):
        raise ValueError("audit seeds must be unique")
    count = len(episodes)
    successes = sum(row["success"] for row in episodes)
    collisions = sum(row["collision"] for row in episodes)
    near(number(report.get("success_rate"), "success_rate"), successes / count, "success_rate")
    near(number(report.get("collision_rate"), "collision_rate"), collisions / count, "collision_rate")
    gate = report.get("pass_gate")
    if not isinstance(gate, dict):
        raise ValueError("audit pass_gate is missing")
    minimum = integer(gate.get("minimum_episodes"), "minimum_episodes", 1)
    minimum_success = number(gate.get("success_rate_at_least"), "success_rate_at_least")
    maximum_collision = number(gate.get("collision_rate_at_most"), "collision_rate_at_most")
    if not 0 <= minimum_success <= 1 or not 0 <= maximum_collision <= 1:
        raise ValueError("pass_gate rates must be in [0, 1]")
    source_passed = boolean(report.get("passed"), "passed")
    if source_passed != (count >= minimum and successes / count >= minimum_success and collisions / count <= maximum_collision):
        raise ValueError("source passed flag disagrees with episode outcomes")
    intersections = sum(not row["proxy_clear"] for row in episodes)
    all_attitudes = all(row["orientation_recorded"] for row in episodes)
    return dict(version=1, scene_id="hall", run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        source=dict(audit_file="hall_avoidance-audit.json", audit_sha256=audit_sha256, kind="audit",
                    policy=report.get("policy", "")),
        transform=dict(translation_xyz=list(SCENE_OFFSET), units="m", up=[0, 0, 1]),
        proxy=dict(center_xyz=translated([*PROXY_CENTER, PROXY_HEIGHT / 2]),
                   radius_m=PROXY_RADIUS, height_m=PROXY_HEIGHT, approximate=True),
        vehicle=dict(envelope_radius_m=VEHICLE_RADIUS,
                     orientation="recorded quaternion" if all_attitudes else "not recorded; fixed display pose"),
        summary=dict(episodes=count, successes=successes, recorded_collisions=collisions,
                     swept_proxy_intersections=intersections,
                     min_swept_proxy_clearance_m=min(row["metrics"]["swept_proxy_clearance_m"] for row in episodes),
                     source_passed=source_passed, all_proxy_clear=intersections == 0,
                     all_successful=successes == count,
                     local_acceptance_passed=count >= 20 and successes == count and collisions == 0 and intersections == 0),
        clearance_definition="Horizontal 0.09 m vehicle-envelope clearance to one approximate 0.45 m cylinder; exact along linear replay segments, not between all physics ticks.",
        limitations=[
            "Only the single manually fitted cylinder is checked; other captured hall surfaces have no validated collision geometry.",
            "A positive swept value certifies the interpolated replay against this horizontal proxy only, not real-world or full-hall collision avoidance.",
            "Source MuJoCo contact flags and recomputed geometric envelope intersections are separate outcomes.",
            "Recorded quaternion is used where present; otherwise attitude is a fixed display pose, not a recovered physical orientation.",
        ], episodes=episodes)


def load_replay(run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("run-id must contain only letters, digits, underscores or hyphens")
    path = ROOT / "outputs/missions" / run_id / "hall_avoidance-audit.json"
    raw = path.read_bytes()
    report = json.loads(raw)
    if report.get("run_id") != run_id:
        raise ValueError("audit run_id does not match its requested run directory")
    return build_replay(report, hashlib.sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/scene-replay/hall-replay.json")
    parser.add_argument("--check", action="store_true", help="validate and print metrics without writing a replay")
    parser.add_argument("--require-clear", action="store_true", help="write only with >=20 successful trials and no recorded or swept proxy collisions")
    args = parser.parse_args()
    try:
        replay = load_replay(args.run_id)
        if args.require_clear and not replay["summary"]["local_acceptance_passed"]:
            raise ValueError("local acceptance failed; no replay was replaced")
        if not args.check:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                             prefix=args.output.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    json.dump(replay, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                    stream.write("\n")
                    stream.flush()
                    temporary.replace(args.output)
                finally:
                    temporary.unlink(missing_ok=True)
        print(json.dumps(dict(run_id=args.run_id, **replay["summary"]), ensure_ascii=False, indent=2))
    except (ValueError, OSError, TypeError) as error:
        parser.exit(2, f"Cannot export Hall replay: {error}\n")


if __name__ == "__main__":
    main()
