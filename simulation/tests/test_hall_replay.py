"""Replay conversion must not turn sparse, successful-looking paths into safety claims."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("export_hall_replay", Path(__file__).resolve().parents[1] / "scripts/export_hall_replay.py")
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def episode(points=None, *, seed=100, collision=False, success=True):
    points = points or [[-1.6, -1., 1.2], [-.5, -1., 1.2], [.5, -1., 1.2], [1.6, -1., 1.2]]
    goal = points[-1][:2] if success else [3., 0.]
    return dict(seed=seed, steps=len(points)-1, seconds=(len(points)-1)*.1,
                trajectory_xyz=points, goal=goal, success=success, collision=collision,
                failure_reason="" if success else "proxy_collision" if collision else "timeout",
                min_clearance=REPLAY.clearance_metrics(points)["recomputed_sampled_clearance_m"],
                final_distance=math.dist(points[-1][:2], goal),
                path_length_m=sum(math.dist(a, b) for a, b in zip(points, points[1:])))


def report(rows=None):
    rows = rows or [episode(seed=seed) for seed in range(20)]
    success = sum(row["success"] for row in rows)/len(rows)
    collision = sum(row["collision"] for row in rows)/len(rows)
    return dict(task="hall_avoidance", kind="audit", run_id="hall-test", stale_vision=False,
                teacher_used_in_evaluation=False, gaussian_scene_used=True, obstacle_geometry_in_policy=False,
                episodes=rows, success_rate=success, collision_rate=collision,
                pass_gate=dict(minimum_episodes=20, success_rate_at_least=.8, collision_rate_at_most=.1),
                passed=len(rows)>=20 and success>=.8 and collision<=.1)


def test_swept_check_catches_obstacle_crossing_between_clear_samples():
    row = episode([[-1., .2, 1.2], [1., .2, 1.2]])
    result = REPLAY.export_episode(row)
    assert result["metrics"]["recomputed_sampled_clearance_m"] == pytest.approx(.46)
    assert result["metrics"]["swept_proxy_clearance_m"] == pytest.approx(-.54)
    assert result["metrics"]["closest_segment_fraction"] == pytest.approx(.5)
    assert result["collision"] is False  # MuJoCo's recorded flag is retained, not rewritten.
    assert result["proxy_clear"] is False


def test_stationary_and_tangent_segments_do_not_become_infinite_or_clear():
    stationary = REPLAY.clearance_metrics([[0, .2, 1.2], [0, .2, 1.2]])
    assert stationary["swept_proxy_clearance_m"] == pytest.approx(-.54)
    tangent = REPLAY.export_episode(episode([[-1., .74, 1.2], [1., .74, 1.2]]))
    assert tangent["metrics"]["swept_proxy_clearance_m"] == pytest.approx(0.)
    assert tangent["proxy_clear"] is False


def test_translates_reset_goal_path_and_preserves_recorded_attitude():
    row = episode()
    row["trajectory_times_s"] = [0., .1, .2, .3]
    q = [math.sqrt(.5), 0., 0., math.sqrt(.5)]
    row["trajectory_quat_wxyz"] = [q] * 4
    result = REPLAY.export_episode(row)
    assert result["start_xyz"] == pytest.approx([13.55, -4.15, 1.2])
    assert result["goal_xyz"] == pytest.approx([16.75, -4.15, 1.2])
    assert result["trajectory_xyz"][-1] == result["goal_xyz"]
    assert result["trajectory_quat_xyzw"][0] == pytest.approx([0, 0, math.sqrt(.5), math.sqrt(.5)])
    assert result["times_s"] == row["trajectory_times_s"]
    assert result["orientation_recorded"] is True


def test_partial_terminal_timestep_does_not_rescale_earlier_samples():
    row = episode(success=False, collision=True)
    row["seconds"] = .234
    result = REPLAY.export_episode(row)
    assert result["times_s"] == pytest.approx([0., .1, .2, .234])
    assert result["orientation_recorded"] is False
    assert "trajectory_quat_xyzw" not in result
    row["trajectory_times_s"] = [0., .078, .156, .234]
    with pytest.raises(ValueError, match="timestamp"):
        REPLAY.export_episode(row)


@pytest.mark.parametrize("change,match", [
    (lambda row: row["trajectory_xyz"][0].__setitem__(0, float("nan")), "finite"),
    (lambda row: row.__setitem__("steps", 1), "reset pose"),
    (lambda row: row.__setitem__("seconds", .235), "terminal physics timestep"),
    (lambda row: row.__setitem__("seconds", .4), "navigation timestep"),
    (lambda row: row.__setitem__("min_clearance", 4.), "sampled clearance"),
    (lambda row: row.__setitem__("min_polyline_clearance", 4.), "polyline clearance"),
    (lambda row: row.__setitem__("final_distance", 8.), "goal distance"),
    (lambda row: row.__setitem__("path_length_m", 8.), "path length"),
    (lambda row: row.__setitem__("collision", "false"), "boolean"),
    (lambda row: row.__setitem__("trajectory_quat_wxyz", [[2., 0, 0, 0]] * 4), "quaternion norm"),
])
def test_rejects_malformed_or_inconsistent_source_data(change, match):
    row = episode()
    change(row)
    with pytest.raises(ValueError, match=match):
        REPLAY.export_episode(row)


def test_keeps_failed_episodes_and_separates_recorded_contacts_from_envelope():
    rows = [episode(seed=seed) for seed in range(20)]
    rows[-1] = episode(seed=19, success=False, collision=True)
    result = REPLAY.build_replay(report(rows), "test-hash")
    assert len(result["episodes"]) == 20
    assert result["summary"]["successes"] == 19
    assert result["summary"]["recorded_collisions"] == 1
    assert result["summary"]["swept_proxy_intersections"] == 0
    assert result["summary"]["source_passed"] is True
    assert result["summary"]["local_acceptance_passed"] is False


def test_rejects_duplicate_trials_and_fabricated_source_summary():
    original = report()
    duplicate = deepcopy(original)
    duplicate["episodes"][1]["seed"] = duplicate["episodes"][0]["seed"]
    with pytest.raises(ValueError, match="seeds must be unique"):
        REPLAY.build_replay(duplicate, "test-hash")
    fabricated = deepcopy(original)
    fabricated["success_rate"] = .9
    with pytest.raises(ValueError, match="success_rate"):
        REPLAY.build_replay(fabricated, "test-hash")
    fabricated = deepcopy(original)
    fabricated["passed"] = False
    with pytest.raises(ValueError, match="passed flag"):
        REPLAY.build_replay(fabricated, "test-hash")


def test_failed_acceptance_leaves_existing_replay_untouched(tmp_path, monkeypatch):
    audit_dir = tmp_path / "outputs/missions/hall-test"
    audit_dir.mkdir(parents=True)
    (audit_dir / "hall_avoidance-audit.json").write_text(json.dumps(report([episode()])))
    existing = tmp_path / "replay.json"
    existing.write_text("previous replay")
    monkeypatch.setattr(REPLAY, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["export_hall_replay.py", "--run-id", "hall-test", "--output", str(existing), "--require-clear"])
    with pytest.raises(SystemExit) as stopped:
        REPLAY.main()
    assert stopped.value.code == 2
    assert existing.read_text() == "previous replay"


def test_existing_twenty_seed_audit_has_exact_sampled_clearance_and_safe_replay():
    source = REPLAY.ROOT / "outputs/missions/hall-20260921-fast/hall_avoidance-audit.json"
    if not source.exists():
        pytest.skip("saved integration audit is unavailable")
    result = REPLAY.load_replay("hall-20260921-fast")
    assert result["summary"]["episodes"] == 20
    assert result["summary"]["local_acceptance_passed"] is True
    assert result["summary"]["min_swept_proxy_clearance_m"] > .3
    for row in result["episodes"]:
        assert row["metrics"]["swept_proxy_clearance_m"] <= row["metrics"]["source_sampled_clearance_m"] + 1e-7
        assert len(row["trajectory_xyz"]) == len(row["times_s"]) == row["steps"] + 1
