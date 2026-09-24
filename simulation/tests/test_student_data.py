"""Teaching exports must retain unsuccessful trials and exact recorded values."""
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("export_student_data", ROOT / "scripts/export_student_data.py")
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)
SOURCE = ROOT / "outputs/missions" / EXPORT.DEFAULT_RUN


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    if not (SOURCE / "hall-demonstrations.npz").is_file():
        pytest.skip("Local recorded training evidence is not installed")
    destination = tmp_path_factory.mktemp("student-export") / "data"
    summary = EXPORT.export(output=destination)
    return destination, summary


def test_entire_dataset_roundtrips_exactly_and_selected_membership_is_truthful(package):
    destination, summary = package
    exported = rows(destination / "training_samples.csv")
    with np.load(SOURCE / "hall-demonstrations.npz", allow_pickle=False) as original:
        matrix = np.array([[float(row[col]) for col in EXPORT.OBS_COLUMNS + ["action_x", "action_y"]]
                           for row in exported], dtype=np.float32)
        np.testing.assert_array_equal(matrix[:, :78], original["observations"])
        np.testing.assert_array_equal(matrix[:, 78:], original["actions"])
    assert len(exported) == summary["counts"]["training_samples"] == 2400
    assert [int(row["sample_index"]) for row in exported] == list(range(2400))
    assert [row["collection_round"] for row in exported] == ["0"] * 1600 + ["1"] * 800
    assert [row["used_in_selected_model"] for row in exported] == ["true"] * 1600 + ["false"] * 800
    assert "seed" not in exported[0] and "timestamp" not in exported[0]
    assert EXPORT.sha256(destination / "hall-demonstrations.npz") == EXPORT.sha256(SOURCE / "hall-demonstrations.npz")


def test_failed_candidate_and_all_control_failures_are_retained(package):
    destination, summary = package
    episodes = rows(destination / "episodes.csv")
    assert len(episodes) == 64
    expected_counts = dict(zip(EXPORT.PHASES, [8, 8, 8, 20, 20]))
    for phase, count in expected_counts.items():
        selected = [row for row in episodes if row["phase"] == phase]
        assert len(selected) == count
        source_rows = EXPORT.read_json(SOURCE / f"hall_avoidance-{phase}.json")["episodes"]
        assert [int(row["seed"]) for row in selected] == [row["seed"] for row in source_rows]
        assert [row["failure_reason"] for row in selected] == [row["failure_reason"] for row in source_rows]
    candidate = next(p for p in summary["phases"] if p["phase"] == "validation-1")
    assert (candidate["successes"], candidate["collisions"]) == (4, 1)
    assert candidate["failure_reasons"] == {"proxy_collision": 1, "timeout": 3}
    control = next(p for p in summary["phases"] if p["phase"] == "stale-vision")
    assert control["failure_reasons"] == {"timeout": 20}
    assert summary["audit"]["min_polyline_clearance_m"] == pytest.approx(0.28695449046133226)
    assert summary["audit"]["mean_seconds"] == pytest.approx(5.195)


def test_all_recorded_poses_and_partial_collision_time_are_preserved(package):
    destination, summary = package
    exported = rows(destination / "trajectories.csv")
    assert len(exported) == summary["counts"]["trajectory_rows"] == 4983
    offset = 0
    for phase in EXPORT.PHASES:
        report = EXPORT.read_json(SOURCE / f"hall_avoidance-{phase}.json")
        for episode_index, episode in enumerate(report["episodes"]):
            n = len(episode["trajectory_xyz"])
            actual = exported[offset:offset + n]
            offset += n
            assert [int(row["frame_index"]) for row in actual] == list(range(n))
            assert all(row["phase"] == phase and int(row["episode_index"]) == episode_index
                       and int(row["seed"]) == episode["seed"] for row in actual)
            np.testing.assert_array_equal([[float(row[c]) for c in ("x_m", "y_m", "z_m")]
                                           for row in actual], episode["trajectory_xyz"])
            np.testing.assert_array_equal([[float(row[c]) for c in ("quat_w", "quat_x", "quat_y", "quat_z")]
                                           for row in actual], episode["trajectory_quat_wxyz"])
            np.testing.assert_array_equal([float(row["time_s"]) for row in actual], episode["trajectory_times_s"])
    partial = next(row for row in reversed(exported) if row["phase"] == "validation-1" and row["seed"] == "2005200")
    assert float(partial["time_s"]) == pytest.approx(8.046)
    assert float(partial["time_s"]) != pytest.approx(int(partial["frame_index"]) * .1)


def test_pairing_uses_same_actual_seeds_and_history_has_only_recorded_endpoints(package):
    destination, _ = package
    paired = rows(destination / "audit_paired_control.csv")
    assert len(paired) == 20
    assert [int(row["seed"]) for row in paired] == list(range(2006208, 2006228))
    assert all(row["live_success"] == "true" and row["stale_success"] == "false" for row in paired)
    history = rows(destination / "training_history.csv")
    assert [row["samples"] for row in history] == ["1600", "2400"]
    assert [row["selected"] for row in history] == ["true", "false"]
    original = [r for r in EXPORT.read_json(SOURCE / "history.json") if r["phase"] == "fit"]
    assert [float(r["final_epoch_mse"]) for r in history] == [r["loss"] for r in original]
    assert "epoch" not in history[0]


def test_manifest_verifies_every_file_and_original_inputs_remain_unchanged(package):
    destination, _ = package
    manifest = EXPORT.read_json(destination / "provenance.json")
    for row in manifest["originals"]:
        original = ROOT / row["path"]
        assert original.stat().st_size == row["bytes"]
        assert EXPORT.sha256(original) == row["sha256"]
    for row in manifest["generated_files"]:
        generated = destination / row["path"]
        assert generated.stat().st_size == row["bytes"]
        assert EXPORT.sha256(generated) == row["sha256"]
    for document in destination.rglob("*.json"):
        assert "/Users/" not in document.read_text(encoding="utf-8")
    config = EXPORT.read_json(destination / "raw/run-config.json")
    assert config["arguments"]["source_checkpoint"].startswith("checkpoints/")


def test_dictionary_covers_every_csv_column_without_inventing_depth_or_sample_ids(package):
    destination, _ = package
    dictionary = EXPORT.read_json(destination / "feature_dictionary.json")
    descriptions = dictionary["training_samples_columns"]
    assert [row["name"] for row in descriptions] == EXPORT.TRAIN_COLUMNS
    features = descriptions[3:75]
    assert features[0]["cell_type"] == "T4a"
    assert features[-1]["cell_type"] == "T5d"
    assert (features[-1]["spatial_row"], features[-1]["spatial_column"]) == (2, 2)
    assert all(row["unit"] == "dimensionless" for row in features)


def test_validation_rejects_dropped_failed_rows_or_invented_trajectory(package):
    reports = {phase: EXPORT.read_json(SOURCE / f"hall_avoidance-{phase}.json") for phase in EXPORT.PHASES}
    seeds = EXPORT.read_json(SOURCE / "seed-manifest.json")
    tampered = deepcopy(reports)
    tampered["validation-1"]["episodes"] = [row for row in tampered["validation-1"]["episodes"] if row["success"]]
    with pytest.raises(ValueError, match="Seed mismatch"):
        EXPORT.validate_reports(tampered, seeds)
    tampered = deepcopy(reports)
    del tampered["audit"]["episodes"][0]["trajectory_quat_wxyz"][-1]
    with pytest.raises(ValueError, match="shape mismatch"):
        EXPORT.validate_reports(tampered, seeds)


def test_refuses_source_or_unowned_output_and_sanitizes_external_paths(tmp_path):
    for target in (ROOT, SOURCE, SOURCE / "raw"):
        with pytest.raises(ValueError, match="overwrite"):
            EXPORT.export(output=target)
    unowned = tmp_path / "user-data"
    unowned.mkdir()
    (unowned / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="not an exporter-owned"):
        EXPORT.export(output=unowned)
    assert (unowned / "keep.txt").read_text() == "keep"
    assert EXPORT.sanitize({"path": "/Users/another/private/model.pt"}) == {"path": "<external-path>/model.pt"}
    assert EXPORT.sanitize(r'C:\Users\student\private\model.pt') == '<external-path>/model.pt'
    assert EXPORT.sanitize(r'\\school-server\private\model.pt') == '<external-path>/model.pt'
    assert EXPORT.sanitize(r'C:\Lab\simulation\checkpoints\head.pt', root=r'C:\Lab\simulation') == 'checkpoints/head.pt'
