#!/usr/bin/env python3
"""Bounded fine-tuning using ONLY captured TUM Flight Hall RGB.

Frozen Flyvis vision + goal/velocity/previous-command assistance feeds the existing
78-input M4 head. Privileged geometry labels demonstrations only; neither geometry
nor the diagnostic distance readout enters the policy. No synthetic RGB fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from torch import nn
from fruitfly_sim.flyvis_features import FlyvisFeatureExtractor
from fruitfly_sim.splat_env import SplatNavigationEnv, WebSplatRenderer
from fruitfly_sim.observer_view import DroneObserver
from train_visual import VisualHead
from train_missions import Live, atomic_json

torch.set_num_threads(2)
TASK = "hall_avoidance"
LABEL = "原有 Flight Hall · 真實視覺避障及到達"
SOURCE = ROOT / "checkpoints/training/M4-visual-head.pt"
LIMITATIONS = [
    "Only the original static TUM Flight Hall 3DGS, not 4DGS or a new scene.",
    "One manually fitted enlarged cylinder; no full-hall collision mesh or measured surface depth.",
    "Goal/velocity-assisted planar navigation, fixed altitude and yaw; not outdoor/long-distance navigation.",
    "Frozen Flyvis visual circuitry plus a separately trained action head, not whole-brain training.",
    "Frozen-feature control is a temporal-vision diagnostic, not proof of causal depth use.",
]


class HallLive(Live):
    def __init__(self, run_id):
        self.observer = None
        super().__init__(run_id, [TASK], labels={TASK: LABEL})
        self.update(scene="TUM Flight Hall", mode="hall_visual", task=TASK,
                    task_label=LABEL, algorithm="原 M4 微調：示範學習 + 純學習策略 DAgger；Flyvis 凍結",
                    observation_mode="72 frozen Flyvis T4/T5 + goal xy + velocity xy + previous action xy",
                    preview_label="等待真實 3DGS 相機", renderer="WebGL 3DGS + MuJoCo")

    def update(self, **fields):
        super().update(**fields)
        # Only the bridge's progress file changes. Original pilot results stay intact.
        atomic_json(ROOT / "outputs/realistic/status.json", dict(
            phase=self.status.get("state", "initializing"),
            message=self.status.get("message", "原有 Flight Hall 視覺微調"),
            completed=self.status.get("progress", 0), total=self.status.get("total", 0),
            run_id=self.status["run_id"], updated=time.time(),
            renderer="WebGL 3DGS + MuJoCo", scene="TUM Flight Hall",
            unity_status="not used for this training run"))

    def frame(self, env, label, force=False):
        if env.latest_rgb is None or (not force and time.monotonic() - self.last_frame < 2):
            return
        rgb = env.latest_rgb.copy()
        frame = np.concatenate((rgb, env.render("overview")), axis=1)
        for name, pixels in (("live.jpg", frame), ("latest-rgb.png", rgb)):
            extension = ".png" if name.endswith(".png") else ".jpg"
            ok, encoded = cv2.imencode(extension, cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError("Could not encode actual hall observation")
            target = self.out / name
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(encoded.tobytes())
            temporary.replace(target)
        if self.observer is None:
            self.observer = DroneObserver(env, width=960, height=540)
        for view in ("follow", "overview"):
            pixels = self.observer.render(view)
            ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR),
                                      [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError('Could not encode MuJoCo observer view')
            target = self.out / f'observer-{view}.jpg'
            temporary = target.with_suffix('.tmp')
            temporary.write_bytes(encoded.tobytes())
            temporary.replace(target)
        atomic_json(self.out / 'observer-preview.json', dict(
            kind='live', run_id=self.status['run_id'], width=960, height=540,
            updated_at=time.time(), display_only=True, policy_rgb_size=[320, 240]))
        self.last_frame = time.monotonic()
        self.update(frame_timestamp=time.time(), preview_label=label)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def polyline_clearance(trace, center, obstacle_radius, vehicle_radius):
    """Exact clearance of the saved XY polyline to an inflated cylinder.

    Linear interpolation between 10 Hz observations is a replay check, not a
    replacement for MuJoCo contact checks at each physics substep.
    """
    points = np.asarray(trace, dtype=float)[:, :2]
    starts = points[:-1]
    segments = points[1:] - starts
    lengths_squared = np.sum(segments * segments, axis=1)
    fractions = np.divide(np.sum((np.asarray(center) - starts) * segments, axis=1),
                          lengths_squared, out=np.zeros(len(segments)), where=lengths_squared > 0)
    nearest = starts + np.clip(fractions, 0, 1)[:, None] * segments
    distances = np.linalg.norm(nearest - np.asarray(center), axis=1)
    return float(np.min(distances) - obstacle_radius - vehicle_radius)


def collect(env, model, live, xs, ys, frames, seed_start, iteration):
    """DAgger queries the teacher at states visited by the UNMIXED learner."""
    initial = iteration == 0
    rng = np.random.default_rng(seed_start)
    seed = seed_start
    used_seeds = [seed]
    obs, _ = env.reset(seed=seed)
    completed = successes = collisions = 0
    live.update(state="training", phase="demonstrations" if initial else "dagger",
                progress=0, total=frames)
    for index in range(frames):
        if index % 20 == 0:
            live.control()
            live.update(progress=index, steps=len(xs),
                        message=f"原有 Flight Hall：{'示範' if initial else '純學習策略 DAgger'} {index}/{frames} 幀")
            live.frame(env, "真實 3DGS RGB / MuJoCo 碰撞近似；訓練資料收集")
        teacher = env.expert_action()
        xs.append(obs.copy())
        ys.append(teacher.copy())
        action = np.clip(teacher + rng.normal(0, .035, 2), -1, 1) if initial else model.predict(obs)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            completed += 1
            successes += int(info["is_success"])
            collisions += int(info["collision"])
            if index + 1 < frames:
                seed += 1
                used_seeds.append(seed)
                obs, _ = env.reset(seed=seed)
    live.frame(env, "真實 3DGS RGB / MuJoCo 碰撞近似；資料收集完成", force=True)
    live.update(progress=frames, steps=len(xs))
    result = dict(iteration=iteration, seeds=used_seeds, samples=frames,
                  completed_episodes=completed, rollout_successes=successes,
                  rollout_collisions=collisions, teacher_mix=1.0 if initial else 0.0,
                  action_noise_std=.035 if initial else 0.0,
                  labels="Privileged waypoint teacher; absent from policy observations")
    live.record(dict(task=TASK, phase="collection", **result))
    return result


def fit(model, xs, ys, live, epochs, iteration, previous_action_dropout):
    x = torch.as_tensor(np.asarray(xs), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(ys), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    model.train()
    live.update(state="training", phase="fit", progress=0, total=epochs,
                message=f"原有 Flight Hall：訓練決策 head；{len(xs)} 筆真實視覺資料")
    loss_value = None
    for epoch in range(epochs):
        live.control()
        losses = []
        for indices in torch.randperm(len(x)).split(256):
            batch = x[indices].clone()
            # Reduce the temporally correlated previous-command shortcut without
            # changing the pretrained model's observation or action dimensions.
            if previous_action_dropout:
                keep = (torch.rand(len(indices), 1) >= previous_action_dropout)
                batch[:, 76:78] *= keep
            loss = nn.functional.mse_loss(model(batch), y[indices])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            losses.append(float(loss.detach()))
        loss_value = float(np.mean(losses))
        live.update(loss=loss_value, progress=epoch + 1)
    model.eval()
    live.record(dict(task=TASK, phase="fit", iteration=iteration, samples=len(xs), loss=loss_value))
    return loss_value


def evaluate(env, model, seeds, live, kind, *, video=False, stale_vision=False):
    seeds = list(seeds)
    live.update(state="evaluating", phase=kind, progress=0, total=len(seeds),
                message=f"原有 Flight Hall：{kind}；純學習策略，無示範控制")
    episodes = []
    writer = imageio.get_writer(live.out / f"{TASK}-evaluation.mp4", fps=10,
                               codec="libx264", macro_block_size=1) if video else None
    try:
        for index, seed in enumerate(seeds):
            live.control()
            obs, info = env.reset(seed=int(seed))
            frozen = obs[:72].copy()
            trace = [np.asarray(info["position"]).tolist()]
            orientations = [env.data.qpos[3:7].copy().tolist()]
            timestamps = [float(info["time"])]
            goal = np.asarray(info["goal"]).tolist()
            reward_sum = path_length = 0.
            steps = 0
            while True:
                policy_obs = obs.copy()
                if stale_vision:
                    policy_obs[:72] = frozen
                obs, reward, terminated, truncated, info = env.step(model.predict(policy_obs))
                position = np.asarray(info["position"]).tolist()
                path_length += float(np.linalg.norm(np.asarray(position) - trace[-1]))
                trace.append(position)
                orientations.append(env.data.qpos[3:7].copy().tolist())
                timestamps.append(float(info["time"]))
                steps += 1
                reward_sum += reward
                if steps % 20 == 0:
                    live.control()
                if index == 0:
                    live.frame(env, f"{kind} · 固定首個 seed {seed}；左真實 RGB・右碰撞近似")
                    if writer:
                        # Reuse the exact most recent observation, never issue a
                        # second camera request or advance Flyvis for the video.
                        frame = np.concatenate((env.latest_rgb.copy(), env.render("overview")), axis=1)
                        cv2.putText(frame, f"{kind} | seed {seed} | actual RGB / proxy physics",
                                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, .36, (255, 240, 150), 1)
                        writer.append_data(frame)
                if terminated or truncated:
                    break
            success = bool(info["is_success"])
            collision = bool(info["collision"])
            reason = "" if success else ("proxy_collision" if collision else
                     "out_of_bounds" if info.get("out_of_bounds") else "timeout" if truncated else "physics_termination")
            episodes.append(dict(seed=int(seed), success=success, collision=collision,
                steps=steps, seconds=float(info["time"]), failure_reason=reason,
                final_distance=float(info["distance_to_goal"]),
                min_clearance=float(env.minimum_clearance),
                clearance_definition="10Hz sampled horizontal vehicle-envelope clearance to approximate cylinder",
                min_polyline_clearance=polyline_clearance(trace, env.obstacle_center,
                                                         env.OBSTACLE_RADIUS, env.VEHICLE_RADIUS),
                polyline_clearance_definition="Minimum horizontal inflated-cylinder clearance along linear segments between 10Hz saved poses; not a full-scene collision test",
                trajectory_xyz=trace, trajectory_quat_wxyz=orientations,
                trajectory_times_s=timestamps, goal=goal, reward=float(reward_sum), path_length_m=path_length))
            live.update(progress=index + 1)
    finally:
        if writer:
            writer.close()
    success = float(np.mean([row["success"] for row in episodes]))
    collision = float(np.mean([row["collision"] for row in episodes]))
    report = dict(task=TASK, label=LABEL, kind=kind, run_id=live.status["run_id"],
        scene="TUM Flight Hall", backend="WebSplatRenderer", episodes=episodes,
        success_rate=success, collision_rate=collision,
        mean_final_distance=float(np.mean([row["final_distance"] for row in episodes])),
        policy="fine-tuned M4 head; frozen Flyvis + ideal relative goal/velocity/previous action",
        teacher_used_in_evaluation=False, flyvis_used=True, flyvis_weights_trained=False,
        gaussian_scene_used=True, obstacle_geometry_in_policy=False,
        stale_vision=stale_vision, temporal_control="hold first 72 features" if stale_vision else "live neural features",
        actual_camera_rendered_every_step=True, trial_horizon_seconds=12, goal_tolerance_m=.25,
        video_seed=seeds[0] if video else None,
        pass_gate=dict(minimum_episodes=20, success_rate_at_least=.8, collision_rate_at_most=.1),
        passed=len(episodes) >= 20 and success >= .8 and collision <= .1,
        limitations=LIMITATIONS)
    atomic_json(live.out / f"{TASK}-{kind}.json", report)
    live.record(dict(task=TASK, phase=kind, samples=live.status["steps"],
                     success_rate=success, collision_rate=collision, episodes=len(episodes)))
    live.log(f"{kind}：到達 {sum(row['success'] for row in episodes)}/{len(episodes)}；"
             f"簡化物件碰撞 {sum(row['collision'] for row in episodes)}/{len(episodes)}")
    return report


def save_checkpoint(path, model, metadata):
    torch.save(dict(state_dict=model.state_dict(), input_dimension=78, architecture=[78, 128, 128, 2],
                    flyvis_frozen=True, scene="TUM Flight Hall", **metadata), path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=time.strftime("hall-%Y%m%d-%H%M%S"))
    parser.add_argument("--frames", type=int, default=1600)
    parser.add_argument("--dagger-frames", type=int, default=800)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--audit-episodes", type=int, default=20)
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--previous-action-dropout", type=float, default=.5)
    parser.add_argument("--source-checkpoint", default=str(SOURCE),
                        help="Initialize this new run from a compatible frozen-Flyvis action-head checkpoint")
    parser.add_argument("--seed-base", type=int, default=1_000_000,
                        help="Start of this run's disjoint training/validation/audit seed blocks")
    parser.add_argument("--torch-seed", type=int, default=921)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.run_id):
        parser.error("run-id must be 1–80 letters, digits, underscores or hyphens")
    if min(args.frames, args.dagger_frames, args.epochs, args.validation_episodes, args.audit_episodes) < 1 or args.rounds < 0:
        parser.error("frame/epoch/episode counts must be positive; rounds may be zero")
    if not 0 <= args.previous_action_dropout < 1:
        parser.error("previous-action-dropout must be in [0,1)")
    if args.seed_base < 0 or not 0 <= args.torch_seed < 2**32:
        parser.error("seed-base must be nonnegative and torch-seed must be in [0, 2**32)")
    source = Path(args.source_checkpoint).expanduser().resolve()
    if not source.is_file():
        parser.error(f"Source checkpoint is missing: {source}")
    args.source_checkpoint = str(source)
    torch.manual_seed(args.torch_seed)
    np.random.seed(args.torch_seed)
    source_hash = digest(source)
    # Reserve disjoint blocks large enough for reset-after-every-frame worst case.
    block = max(args.frames, args.dagger_frames) + 1000
    training_bases = [args.seed_base + i * block for i in range(args.rounds + 1)]
    validation_start = args.seed_base + (args.rounds + 1) * block
    validation_seeds = list(range(validation_start, validation_start + args.validation_episodes))
    audit_start = validation_start + args.validation_episodes + 1000
    audit_seeds = list(range(audit_start, audit_start + args.audit_episodes))
    seed_manifest = dict(training_seed_blocks=[dict(iteration=i, first=base, last_reserved=base + block - 1)
                         for i, base in enumerate(training_bases)], validation=validation_seeds,
                         audit=audit_seeds, stale_vision_control=audit_seeds, training_rollouts=[])
    live = HallLive(args.run_id)
    live.update(algorithm="由來源 checkpoint 繼續微調：示範學習 + 純學習策略 DAgger；Flyvis 凍結")
    model = VisualHead()
    env = renderer = None
    metadata = dict(source_checkpoint=str(source), source_checkpoint_sha256=source_hash,
                    initialization="Source checkpoint weights; new optimizer and new rollout data",
                    run_id=args.run_id, previous_action_dropout=args.previous_action_dropout,
                    torch_seed=args.torch_seed, seed_base=args.seed_base)
    config = dict(arguments=vars(args), started_at=time.time(), task=TASK,
                  scene="TUM Flight Hall", source_checkpoint_sha256=source_hash,
                  source_scene_sha256=digest(ROOT / "models/gaussian/robot_hall.ply"),
                  observations="72 frozen streaming Flyvis + goal xy + velocity xy + previous action xy",
                  algorithm="supervised head fine-tuning + pure-learner DAgger; not reinforcement learning",
                  training_regularizer="randomly zero previous-action pair in fitting batches only",
                  teacher_only_supplies_training_labels=True, limitations=LIMITATIONS,
                  source_files={str(p.relative_to(ROOT)): digest(p) for p in
                      [Path(__file__), ROOT / "src/fruitfly_sim/splat_env.py", ROOT / "src/fruitfly_sim/flyvis_features.py"]})
    atomic_json(live.out / "run-config.json", config)
    atomic_json(live.out / "seed-manifest.json", seed_manifest)
    xs, ys = [], []
    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        live.task(TASK, state="training")
        live.log("只用原有 Flight Hall；真實 WebGL 3DGS RGB，Flyvis 權重保持凍結")
        live.log(f"由 {source.parent.name}/{source.name} 繼續微調；本輪 seed base {args.seed_base}")
        renderer = WebSplatRenderer(port=args.port)
        extractor = FlyvisFeatureExtractor(cpu_threads=2)
        env = SplatNavigationEnv(extractor, renderer)
        geom = env._obstacle_geom
        previous_aabb = env.model.geom_aabb[geom].copy()
        # Keep broad-phase extents coherent with the new pilot's existing
        # radius=.45 and half-height=.775 cylinder. Do not alter old reports.
        env.model.geom_aabb[geom, 3:] = [.45, .45, .775]
        config["proxy_geometry"] = dict(
            scene_center=[15.15, -2.95], radius_m=.45, bottom_m=0., top_m=1.55,
            vehicle_envelope_radius_m=.09,
            aabb_before=previous_aabb.tolist(), aabb_after=env.model.geom_aabb[geom].tolist(),
            rbound=float(env.model.geom_rbound[geom]),
            correction="New run only: update compiled broad-phase half-extents to actual geom_size")
        atomic_json(live.out / "run-config.json", config)
        baseline = evaluate(env, model, validation_seeds, live, "validation-source")
        baseline_score = (baseline["success_rate"], -baseline["collision_rate"],
                          -baseline["mean_final_distance"])
        best_score = None
        best_weights = None
        best_iteration = None
        for iteration in range(args.rounds + 1):
            frames = args.frames if iteration == 0 else args.dagger_frames
            collection = collect(env, model, live, xs, ys, frames, training_bases[iteration], iteration)
            seed_manifest["training_rollouts"].append(collection)
            atomic_json(live.out / "seed-manifest.json", seed_manifest)
            np.savez_compressed(live.out / "hall-demonstrations.npz",
                                observations=np.asarray(xs), actions=np.asarray(ys))
            fit(model, xs, ys, live, args.epochs, iteration, args.previous_action_dropout)
            report = evaluate(env, model, validation_seeds, live, f"validation-{iteration}")
            score = (report["success_rate"], -report["collision_rate"], -report["mean_final_distance"])
            save_checkpoint(live.ckpt / f"hall-head-round-{iteration}.pt", model,
                            dict(metadata, training_samples=len(xs), iteration=iteration, validation=score))
            if best_score is None or score > best_score:
                best_score, best_iteration = score, iteration
                best_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
                save_checkpoint(live.ckpt / "hall-visual-head.pt", model,
                                dict(metadata, training_samples=len(xs), iteration=iteration, validation=score))
            # Subsequent pure-learner DAgger gathers corrections around the best
            # validation candidate; final audit never selects weights or a round.
            model.load_state_dict(best_weights)
        model.load_state_dict(best_weights)
        audit = evaluate(env, model, audit_seeds, live, "audit", video=True)
        live.frame(env, "最终验收：固定首個 seed 錄影；最新畫面為最後測試觀測", force=True)
        audit_preview = {name: (live.out / name).read_bytes() for name in (
            "live.jpg", "latest-rgb.png", "observer-follow.jpg", "observer-overview.jpg", "observer-preview.json")}
        audit_frame_timestamp = live.status["frame_timestamp"]
        stale = evaluate(env, model, audit_seeds, live, "stale-vision", stale_vision=True)
        for name, pixels in audit_preview.items():
            (live.out / name).write_bytes(pixels)
        if digest(source) != source_hash:
            raise RuntimeError("Source checkpoint changed during the run")
        summary = dict(task=TASK, run_id=args.run_id, completed_at=time.time(),
                       training_samples=len(xs), selected_iteration=best_iteration,
                       selected_by="validation success, collision, then final distance",
                       audit_seeds_used_in_training_or_selection=False,
                       source_checkpoint_unchanged=True, source_checkpoint_sha256=source_hash,
                       source_checkpoint=str(source),
                       source_validation_score=list(baseline_score),
                       selected_validation_score=list(best_score),
                       selected_validation_regressed=best_score < baseline_score,
                       comparison_scope="Same validation seeds only; source baseline never selected as a trained candidate",
                       checkpoint=str(live.ckpt / "hall-visual-head.pt"),
                       success_rate=audit["success_rate"], collision_rate=audit["collision_rate"],
                       stale_vision_success_rate=stale["success_rate"],
                       stale_vision_collision_rate=stale["collision_rate"],
                       passed=audit["passed"], limitations=LIMITATIONS)
        atomic_json(live.out / "hall-training.json", summary)
        atomic_json(live.out / "renderer-performance.json", dict(backend="WebSplatRenderer",
                    frames=len(renderer.latencies), median_ms=float(np.median(renderer.latencies) * 1000),
                    p95_ms=float(np.percentile(renderer.latencies, 95) * 1000), scope="camera bridge only"))
        live.task(TASK, state="passed" if audit["passed"] else "needs_work",
                  success_rate=audit["success_rate"], collision_rate=audit["collision_rate"],
                  episodes=len(audit["episodes"]),
                  result="通過本輪新種子驗收" if audit["passed"] else "本輪完成，未達到達／碰撞門檻")
        live.update(state="complete", phase="complete", progress=len(audit_seeds), total=len(audit_seeds),
                    frame_timestamp=audit_frame_timestamp,
                    preview_label="已完成：真實 3DGS 最後驗收畫面（非持續訓練）",
                    message=f"Flight Hall 本輪完成：到達 {audit['success_rate']:.0%}，簡化物件碰撞 {audit['collision_rate']:.0%}；{'達到本輪門檻' if audit['passed'] else '未達本輪門檻'}")
    except (InterruptedError, KeyboardInterrupt) as error:
        save_checkpoint(live.ckpt / "hall-interrupted.pt", model, dict(metadata, training_samples=len(xs)))
        live.task(TASK, state="stopped")
        live.update(state="stopped", phase="stopped", message=str(error) or "本輪已停止")
    except Exception as error:
        save_checkpoint(live.ckpt / "hall-error.pt", model, dict(metadata, training_samples=len(xs)))
        (live.out / "error.txt").write_text(traceback.format_exc())
        live.task(TASK, state="error")
        live.update(state="error", phase="error", message=f"{type(error).__name__}: {error}；未使用替代相機")
        raise
    finally:
        if live.observer is not None:
            live.observer.close()
        if env is not None:
            env.close()
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    main()
