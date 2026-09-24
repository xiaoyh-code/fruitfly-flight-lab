#!/usr/bin/env python3
"""Export recorded Flight Hall evidence for teaching; no simulation or publication.

The CSV files are lossless, machine-readable exports of saved arrays/reports.
This script never reconstructs missing RGB frames, training curves or sample seeds.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = "hall-20260922-replay1"
PHASES = ("validation-source", "validation-0", "validation-1", "audit", "stale-vision")
LABELS = {
    "validation-source": "來源模型：驗證組",
    "validation-0": "第 0 輪：驗證組（入選）",
    "validation-1": "第 1 輪：驗證組（未入選）",
    "audit": "入選模型：獨立驗收",
    "stale-vision": "入選模型：凍結首幀視覺對照",
}
OBS_COLUMNS = [f"feature_{i:02d}" for i in range(72)] + [
    "goal_x", "goal_y", "velocity_x", "velocity_y", "previous_action_x", "previous_action_y"]
TRAIN_COLUMNS = ["sample_index", "collection_round", "used_in_selected_model", *OBS_COLUMNS,
                 "action_x", "action_y"]
EPISODE_COLUMNS = ["phase", "episode_index", "seed", "success", "collision", "steps", "seconds",
    "failure_reason", "final_distance_m", "min_sampled_clearance_m", "min_polyline_clearance_m",
    "path_length_m", "reward", "goal_x_m", "goal_y_m"]
TRAJECTORY_COLUMNS = ["phase", "episode_index", "seed", "frame_index", "time_s", "x_m", "y_m", "z_m",
                      "quat_w", "quat_x", "quat_y", "quat_z"]
MISSING = [
    "完整訓練 RGB 影像序列未有保存；NPZ 只保存 78 維 observation 與 2 維教師 action。",
    "每筆訓練樣本所屬 episode、seed、時間及原始圖像沒有逐筆保存，不可從匯出 CSV 推算。",
    "只保存每輪最後一個 epoch 的平均 batch MSE；沒有完整 60-epoch loss 曲線。",
    "驗證／驗收／對照保存了位置、姿態、時間與結果，沒有保存逐步 policy action／observation。",
    "沒有全場景碰撞網格、實測深度、校園測試或實機飛行結果。",
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sanitize(value, root=ROOT):
    """Retain project-relative provenance while removing personal machine paths."""
    if isinstance(value, dict):
        return {sanitize(k, root): sanitize(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, root) for v in value]
    if isinstance(value, str):
        from pathlib import PurePosixPath, PureWindowsPath
        normalized = value.replace("\\", "/")
        prefix = str(root).replace("\\", "/").rstrip("/")
        if normalized == prefix:
            return "."
        if normalized.startswith(prefix + "/"):
            return normalized[len(prefix) + 1:]
        if normalized.startswith("/") or PureWindowsPath(value).is_absolute():
            return "<external-path>/" + PurePosixPath(normalized).name
    return value


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: str(v).lower() if isinstance(v, bool) else v for k, v in row.items()})


def feature_dictionary():
    columns = [
        dict(name="sample_index", unit="index", description="NPZ 的零起始列序號；非時間或 episode ID。"),
        dict(name="collection_round", unit="index", description="按 history／seed-manifest 的收集先後及樣本數標示；0 是示範，1 是 DAgger。"),
        dict(name="used_in_selected_model", unit="boolean", description="此新收集樣本是否屬入選輪次累積 fitting dataset；不包括來源 checkpoint 之前的訓練資料。"),
    ]
    for index in range(72):
        cell = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")[index // 9]
        row, col = divmod(index % 9, 3)
        columns.append(dict(name=f"feature_{index:02d}", observation_index=index, cell_type=cell,
            spatial_row=row, spatial_column=col, unit="dimensionless", range=[-1, 1],
            description="tanh（此 cell type 空間格的平均神經活動 − 本 episode 灰畫面 warmup 基準）；不是距離。"))
    for index, name in enumerate(OBS_COLUMNS[72:], start=72):
        if name.startswith("goal_"):
            description = "clip((目標世界座標 − 無人機世界座標) / 4 m, -1, 1)；理想模擬器目標資訊，非視覺推算。"
        elif name.startswith("velocity_"):
            description = "clip(世界座標水平速度 / 0.8 m/s, -1, 1)；理想模擬器速度資訊。"
        else:
            description = "上一步實際送入導航環境的正規化水平速度命令；每個 episode 首步為 0。"
        columns.append(dict(name=name, observation_index=index, unit="dimensionless", range=[-1, 1],
                            description=description))
    for name in ("action_x", "action_y"):
        columns.append(dict(name=name, unit="dimensionless", range=[-1, 1],
            description="privileged waypoint teacher 的水平速度標籤；乘 0.8 得期望 m/s。這是教師標籤，非必然實際 rollout action；第 0 輪用教師加噪音，第 1 輪用 learner。"))
    return dict(schema_version=1, training_samples_columns=columns,
        preprocessing="320×240 RGB → 中央方形裁切 → 灰階及縮放 → BoxEye 721 retinal hexals → 凍結 Flyvis → T4/T5 的 8×3×3 空間平均。",
        temporal_state="Flyvis 神經狀態跨相機幀保留；每個 episode 重設，gray warmup 0.1 s；camera_dt 0.1 s，neural_dt 0.02 s。",
        csv_format="UTF-8 with BOM; comma-separated; true/false Boolean text; sample and frame indices start at 0.",
        trajectory_coordinates=dict(frame="MuJoCo local world", units="metres", axes="Z up",
            to_3dgs_translation=[15.15, -3.15, 0.0], quaternion_order="w,x,y,z",
            timestamps="Recorded simulation time; terminal collision may stop before a full 0.1 s step."),
        metrics={
            "seconds": "模擬秒數；不是電腦運算 wall time。",
            "collision": "原始 MuJoCo contact 標記，保留 false/true；不是對所有 3DGS 表面碰撞的判定。",
            "min_sampled_clearance_m": "10 Hz 保存位置到圓柱中心的水平距離 − 0.45 m 代理半徑 − 0.09 m 機身保守半徑。",
            "min_polyline_clearance_m": "保存相鄰 XY 點之間線段至圓柱的最小水平淨距；線性插值檢查，並非連續完整物理／全場景安全證明。",
            "final_distance_m": "終點 XY 與目標 XY 的水平距離。成功門檻 < 0.25 m。",
            "path_length_m": "相鄰保存 XYZ 位置的歐氏距離總和。",
            "final_epoch_mse": "該輪最後 epoch 各 batch 的訓練 MSE 算術平均；fitting 時有 previous-action dropout，並非固定模型在完整 dataset 的再評分。",
            "reward": "導航環境記錄的 reward 總和；這輪是監督式 imitation/DAgger，沒有用 reward 做 RL 更新。"},
        source_code=["src/fruitfly_sim/flyvis_features.py", "src/fruitfly_sim/visual_env.py",
                     "src/fruitfly_sim/learning_env.py", "src/fruitfly_sim/splat_env.py", "scripts/train_hall.py"])


def validate_reports(reports, seeds):
    for phase, report in reports.items():
        eps = report["episodes"]
        if not eps or report["kind"] != phase:
            raise ValueError(f"Empty/mismatched report: {phase}")
        expected = seeds["audit"] if phase in ("audit", "stale-vision") else seeds["validation"]
        if [e["seed"] for e in eps] != expected:
            raise ValueError(f"Seed mismatch: {phase}")
        for metric, flag in (("success_rate", "success"), ("collision_rate", "collision")):
            if not np.isclose(report[metric], sum(e[flag] for e in eps) / len(eps)):
                raise ValueError(f"Summary disagrees with episode rows: {phase} {metric}")
        for e in eps:
            xyz = np.asarray(e["trajectory_xyz"], dtype=float)
            quat = np.asarray(e["trajectory_quat_wxyz"], dtype=float)
            time = np.asarray(e["trajectory_times_s"], dtype=float)
            if xyz.shape != (e["steps"] + 1, 3) or quat.shape != (len(xyz), 4) or time.shape != (len(xyz),):
                raise ValueError(f"Missing recorded trajectory or shape mismatch: {phase} seed {e['seed']}")
            if not all(np.isfinite(a).all() for a in (xyz, quat, time)) or np.any(np.diff(time) <= 0):
                raise ValueError(f"Invalid recorded trajectory: {phase} seed {e['seed']}")
            if not np.isclose(time[-1], e["seconds"]):
                raise ValueError("Final recorded timestamp differs from episode duration")


def phase_summary(phase, report):
    eps = report["episodes"]
    return dict(phase=phase, label=LABELS[phase], episodes=len(eps),
        successes=sum(e["success"] for e in eps), collisions=sum(e["collision"] for e in eps),
        success_rate=report["success_rate"], collision_rate=report["collision_rate"],
        mean_seconds=float(np.mean([e["seconds"] for e in eps])),
        min_polyline_clearance_m=min(e["min_polyline_clearance"] for e in eps),
        min_sampled_clearance_m=min(e["min_clearance"] for e in eps),
        mean_final_distance_m=float(np.mean([e["final_distance"] for e in eps])),
        mean_path_length_m=float(np.mean([e["path_length_m"] for e in eps])),
        failure_reasons=dict(Counter(e["failure_reason"] for e in eps if not e["success"])))


def export(run_id=DEFAULT_RUN, output=None, *, root=ROOT):
    root = Path(root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("Invalid run id")
    source = root / "outputs/missions" / run_id
    target = Path(output) if output else root / "outputs/teaching" / run_id / "data"
    if target.is_symlink():
        raise ValueError("Refusing output symlink")
    target = target.resolve()
    if target == root or target in root.parents or target == source or source in target.parents:
        raise ValueError("Output must not overwrite project or source data")
    if target.exists() and (not (target / "provenance.json").is_file() or (target / ".git").exists()):
        raise ValueError("Existing output is not an exporter-owned data directory")
    names = ["run-config.json", "hall-training.json", "history.json", "seed-manifest.json", "renderer-performance.json"]
    names += [f"hall_avoidance-{phase}.json" for phase in PHASES]
    documents = {name: read_json(source / name) for name in names}
    config, training, history, seeds = (documents[name] for name in names[:4])
    if training["run_id"] != run_id or config["arguments"]["run_id"] != run_id:
        raise ValueError("Run id mismatch")
    reports = {phase: documents[f"hall_avoidance-{phase}.json"] for phase in PHASES}
    validate_reports(reports, seeds)
    collections = [row for row in history if row["phase"] == "collection"]
    fits = [row for row in history if row["phase"] == "fit"]
    if collections != [dict(task="hall_avoidance", phase="collection", **row) for row in seeds["training_rollouts"]]:
        raise ValueError("Collection history and seed manifest disagree")
    selected = training["selected_iteration"]
    fit_by_round = {row["iteration"]: row for row in fits}
    selected_samples = fit_by_round[selected]["samples"]
    npz_path = source / "hall-demonstrations.npz"
    with np.load(npz_path, allow_pickle=False) as arrays:
        obs, actions = arrays["observations"], arrays["actions"]
        if obs.shape != (training["training_samples"], 78) or actions.shape != (len(obs), 2):
            raise ValueError("NPZ dimensions disagree with training summary")
        if not np.isfinite(obs).all() or not np.isfinite(actions).all():
            raise ValueError("Non-finite dataset values")
    bounds = np.cumsum([row["samples"] for row in collections])
    if bounds[-1] != len(obs) or [row["samples"] for row in fits] != bounds.tolist():
        raise ValueError("Collected and cumulative fitting counts disagree")
    source_checkpoint = Path(config["arguments"]["source_checkpoint"])
    if not source_checkpoint.is_absolute():
        source_checkpoint = root / source_checkpoint
    checkpoint_dir = root / "checkpoints/missions" / run_id
    checkpoint_paths = [source_checkpoint, checkpoint_dir / "hall-visual-head.pt"]
    checkpoint_paths += [checkpoint_dir / f"hall-head-round-{r['iteration']}.pt" for r in fits]
    all_inputs = [source / name for name in names] + [npz_path] + checkpoint_paths
    originals = {p: sha256(p) for p in all_inputs}
    if originals[source_checkpoint] != config["source_checkpoint_sha256"]:
        raise ValueError("Source checkpoint hash changed since training")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".student-data-", dir=target.parent))
    try:
        (staging / "raw").mkdir()
        for name, document in documents.items():
            write_json(staging / "raw" / name, sanitize(document, root))
        shutil.copyfile(npz_path, staging / npz_path.name)
        write_json(staging / "seed_manifest.json", sanitize(seeds, root))
        write_json(staging / "feature_dictionary.json", feature_dictionary())
        def training_rows():
            for i, (observation, action) in enumerate(zip(obs, actions)):
                index = int(np.searchsorted(bounds, i, side="right"))
                row = dict(sample_index=i, collection_round=collections[index]["iteration"],
                           used_in_selected_model=i < selected_samples)
                row.update(zip(OBS_COLUMNS + ["action_x", "action_y"],
                               map(float, np.concatenate((observation, action)))))
                yield row
        write_csv(staging / "training_samples.csv", TRAIN_COLUMNS, training_rows())
        episode_rows, trajectory_rows = [], []
        for phase, report in reports.items():
            for index, e in enumerate(report["episodes"]):
                row = dict(phase=phase, episode_index=index, **{key: e[key] for key in (
                    "seed", "success", "collision", "steps", "seconds", "failure_reason", "path_length_m", "reward")},
                    final_distance_m=e["final_distance"], min_sampled_clearance_m=e["min_clearance"],
                    min_polyline_clearance_m=e["min_polyline_clearance"], goal_x_m=e["goal"][0], goal_y_m=e["goal"][1])
                episode_rows.append(row)
                for frame, (xyz, quat, timestamp) in enumerate(zip(e["trajectory_xyz"], e["trajectory_quat_wxyz"], e["trajectory_times_s"])):
                    trajectory_rows.append(dict(phase=phase, episode_index=index, seed=e["seed"], frame_index=frame,
                        time_s=timestamp, **dict(zip(["x_m", "y_m", "z_m"], xyz)),
                        **dict(zip(["quat_w", "quat_x", "quat_y", "quat_z"], quat))))
        write_csv(staging / "episodes.csv", EPISODE_COLUMNS, episode_rows)
        write_csv(staging / "trajectories.csv", TRAJECTORY_COLUMNS, trajectory_rows)
        pair_metrics = ["success", "collision", "seconds", "failure_reason", "final_distance", "min_polyline_clearance", "path_length_m"]
        paired = []
        for live, stale in zip(reports["audit"]["episodes"], reports["stale-vision"]["episodes"]):
            row = dict(seed=live["seed"])
            for prefix, episode in (("live", live), ("stale", stale)):
                row.update({f"{prefix}_{metric}": episode[metric] for metric in pair_metrics})
            paired.append(row)
        write_csv(staging / "audit_paired_control.csv", ["seed"] + [f"{p}_{m}" for p in ("live", "stale") for m in pair_metrics], paired)
        rounds = [dict(iteration=row["iteration"], samples=row["samples"],
                       final_epoch_mse=row["loss"], selected=row["iteration"] == selected) for row in fits]
        write_csv(staging / "training_history.csv", ["iteration", "samples", "final_epoch_mse", "selected"], rounds)
        collection_fields = ["iteration", "samples", "completed_episodes", "rollout_successes", "rollout_collisions", "teacher_mix", "action_noise_std"]
        write_csv(staging / "collection_history.csv", collection_fields,
                  [{key: row[key] for key in collection_fields} for row in collections])
        phases = [phase_summary(phase, reports[phase]) for phase in PHASES]
        checkpoint_records = [dict(path=sanitize(str(p), root), sha256=originals[p], bytes=p.stat().st_size,
                                  role="source" if p == source_checkpoint else "selected" if p.name == "hall-visual-head.pt" else "candidate")
                              for p in checkpoint_paths]
        write_json(staging / "checkpoint_hashes.json", checkpoint_records)
        summary = dict(schema_version=1, run_id=run_id, scene="TUM Flight Hall",
            counts=dict(training_samples=len(obs), selected_training_samples=selected_samples,
                        observation_dimensions=78, action_dimensions=2, evaluation_episodes=len(episode_rows),
                        trajectory_rows=len(trajectory_rows), training_fit_endpoints=len(rounds)),
            selection=dict(selected_iteration=selected, criterion=training["selected_by"], rounds=rounds,
                           audit_used_in_selection=False, source_baseline_candidate=False),
            phases=phases, audit=next(p for p in phases if p["phase"] == "audit"),
            paired_control=dict(episodes=len(paired), same_seeds=True,
                                live_successes=sum(e["live_success"] for e in paired),
                                stale_successes=sum(e["stale_success"] for e in paired),
                                interpretation="冻结首帧视觉特征的时序对照；不能证明模型获得真实深度。"),
            collection=seeds["training_rollouts"],
            training=dict(algorithm=config["algorithm"], epochs_per_round=config["arguments"]["epochs"],
                          previous_action_dropout=config["arguments"]["previous_action_dropout"],
                          torch_seed=config["arguments"]["torch_seed"], seed_base=config["arguments"]["seed_base"],
                          recorded_wall_time_s=training["completed_at"] - config["started_at"]),
            proxy_geometry=config["proxy_geometry"], renderer_performance=documents["renderer-performance.json"],
            limitations=training["limitations"], missing_data=MISSING,
            scope="完整收錄此次 run 的記錄；phase 間可能使用相同驗證 seeds，不可把 64 次混合測試當作獨立總成功率。")
        write_json(staging / "summary.json", summary)
        provenance = dict(schema_version=1, exporter="scripts/export_student_data.py", run_id=run_id,
            transformations=["NPZ byte-for-byte copy", "float32 values expanded to lossless CSV decimal strings",
                             "JSON personal absolute paths replaced by project-relative paths",
                             "All five evaluation phases retained, including rejected round and control failures"],
            originals=[dict(path=sanitize(str(p), root), sha256=value, bytes=p.stat().st_size) for p, value in originals.items()],
            source_hashes_recorded_at_training=config["source_files"],
            generated_files=[dict(path=str(p.relative_to(staging)), bytes=p.stat().st_size, sha256=sha256(p))
                             for p in sorted(staging.rglob("*")) if p.is_file()])
        write_json(staging / "provenance.json", provenance)
        for p, digest in originals.items():
            if sha256(p) != digest:
                raise RuntimeError(f"Source changed during export: {p.name}")
        for p in staging.rglob("*.json"):
            if "/Users/" in p.read_text(encoding="utf-8"):
                raise ValueError(f"Personal path escaped sanitization: {p.name}")
        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = export(args.run_id, args.output)
    print(json.dumps(dict(run_id=summary["run_id"], **summary["counts"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
