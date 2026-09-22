#!/usr/bin/env python3
"""Bounded supervised/DAgger training for separate MuJoCo mission baselines.

The policy receives simulator position, velocity and obstacle geometry. It is
NOT the existing Flyvis visual policy. Expert actions only label training data;
validation and final audits execute the neural policy without teacher mixing.
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
import numpy as np
import torch
from torch import nn

torch.set_num_threads(2)
from fruitfly_sim.mission_env import MissionEnv

TASKS = {
    "landing": ("M5 固定平台精準降落", "landing", 0),
    "obstacles3": ("M6a 三個障礙", "obstacles", 3),
    "obstacles5": ("M6b 五個障礙", "obstacles", 5),
    "obstacles8": ("M6c 八個障礙", "obstacles", 8),
    "obstacle_landing": ("M7 避障後降落", "obstacle_landing", 5),
}


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


class MissionPolicy(nn.Module):
    def __init__(self, observation_size=52):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(observation_size, 128), nn.Tanh(),
                                 nn.Linear(128, 128), nn.Tanh(),
                                 nn.Linear(128, 128), nn.Tanh(),
                                 nn.Linear(128, 3), nn.Tanh())

    def forward(self, value):
        return self.net(value)

    def predict(self, observation):
        with torch.no_grad():
            return self(torch.as_tensor(observation, dtype=torch.float32)).numpy()


class Live:
    def __init__(self, run_id, tasks, labels=None):
        self.out = ROOT / "outputs/missions" / run_id
        self.ckpt = ROOT / "checkpoints/missions" / run_id
        self.out.mkdir(parents=True, exist_ok=False)
        self.ckpt.mkdir(parents=True, exist_ok=True)
        self.history = []
        self.last_control = time.time()
        self.last_frame = 0.0
        self.started = time.monotonic()
        self.status = dict(run_id=run_id, state="initializing", task="", task_label="",
                           phase="", algorithm="示範學習 + DAgger（位置／幾何基準）",
                           steps=0, progress=0, total=0, loss=None,
                           preview_label="等待第一張 MuJoCo 影像", frame_timestamp=None,
                           tasks=[dict(id=key, label=(labels or {}).get(key, TASKS.get(key, (key,))[0]), state="pending") for key in tasks],
                           log=[])
        self.update(message="建立獨立任務訓練；保留原有 M0–M4")
        atomic_json(self.out / "history.json", self.history)
        atomic_json(self.out.parent / "latest.json", {"run_id": run_id})

    def update(self, **fields):
        self.status.update(fields)
        self.status.update(updated_at=time.time(), elapsed_seconds=round(time.monotonic() - self.started, 2))
        atomic_json(self.out / "status.json", self.status)

    def log(self, message):
        print(message, flush=True)
        self.status["log"] = (self.status["log"] + [time.strftime("%H:%M:%S ") + message])[-40:]
        self.update(message=message)

    def task(self, key, **fields):
        for item in self.status["tasks"]:
            if item["id"] == key:
                item.update(fields)
        self.update()

    def record(self, row):
        self.history.append(row)
        atomic_json(self.out / "history.json", self.history)

    def control(self):
        def read():
            try:
                return json.loads((self.out / "control.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                return {}
        value = read()
        if value.get("timestamp", 0) <= self.last_control:
            return
        self.last_control = value["timestamp"]
        if value.get("command") == "stop":
            raise InterruptedError("使用者停止本輪；已保存嘅模型及結果保留")
        if value.get("command") == "pause":
            previous = self.status["state"]
            self.update(state="paused", message="已暫停；可繼續或停止")
            while True:
                time.sleep(.2)
                value = read()
                if value.get("timestamp", 0) <= self.last_control:
                    continue
                self.last_control = value["timestamp"]
                if value.get("command") == "stop":
                    raise InterruptedError("使用者停止本輪")
                if value.get("command") == "resume":
                    self.update(state=previous, message="繼續本輪")
                    return

    def frame(self, env, label, force=False):
        if not force and time.monotonic() - self.last_frame < 2:
            return
        import cv2
        frame = np.concatenate((env.render("fpv"), env.render("down"), env.render("overview")), axis=1)
        ok, data = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            temp = self.out / "live.tmp"
            temp.write_bytes(data.tobytes())
            temp.replace(self.out / "live.jpg")
            self.last_frame = time.monotonic()
            self.update(frame_timestamp=time.time(), preview_label=label)


def make_env(key):
    _, task, count = TASKS[key]
    return MissionEnv(task=task, obstacle_count=count or 3)


def collect(env, model, key, live, xs, ys, frames, seed_start, teacher_mix, iteration):
    rng = np.random.default_rng(seed_start)
    obs, _ = env.reset(seed=seed_start)
    episode, successes, collisions = 0, 0, 0
    live.update(state="training", phase="demonstrations" if iteration == 0 else "dagger",
                progress=0, total=frames)
    for index in range(frames):
        if index % 40 == 0:
            live.control()
            live.update(progress=index, steps=len(xs),
                        message=f"{TASKS[key][0]}：收集訓練標籤 {index}/{frames}，示範混合率 {teacher_mix:.0%}")
            live.frame(env, "資料收集：示範／學習動作混合；左前視・中向下・右總覽")
        teacher = env.expert_action()
        xs.append(obs.copy())
        ys.append(teacher.copy())
        action = teacher if rng.random() < teacher_mix else model.predict(obs)
        # Small velocity exploration; physical safety remains evaluated by the env.
        action = np.clip(action + rng.normal(0, .025, size=3), -1, 1)
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            successes += int(info.get("is_success", False))
            collisions += int(info.get("collision", False))
            episode += 1
            obs, _ = env.reset(seed=seed_start + episode)
    live.update(progress=frames, steps=len(xs))
    return {"first_seed": seed_start, "last_seed": seed_start + episode,
            "samples": frames, "completed_episodes": episode,
            "mixed_rollout_successes": successes, "mixed_rollout_collisions": collisions,
            "teacher_mix": teacher_mix, "action_noise_std": .025}


def fit(model, xs, ys, key, live, epochs, iteration):
    x = torch.as_tensor(np.asarray(xs), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(ys), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=7e-4)
    model.train()
    live.update(state="training", phase="fit", progress=0, total=epochs,
                message=f"{TASKS[key][0]}：訓練神經網絡，共 {len(xs)} 筆標籤")
    for epoch in range(epochs):
        live.control()
        losses = []
        for indices in torch.randperm(len(x)).split(256):
            loss = nn.functional.mse_loss(model(x[indices]), y[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        mean = float(np.mean(losses))
        live.update(loss=mean, progress=epoch + 1)
    model.eval()
    live.record(dict(task=key, phase="fit", iteration=iteration, samples=len(xs), loss=mean))
    return mean


def safe_number(value):
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def evaluate(env, model, key, seeds, live, kind, *, video=False):
    import imageio.v2 as imageio
    live.update(state="evaluating", phase=kind, progress=0, total=len(seeds),
                message=f"{TASKS[key][0]}：{kind}；只由神經網絡控制，無示範混合")
    episodes = []
    writer = None
    if video:
        writer = imageio.get_writer(live.out / f"{key}-evaluation.mp4", fps=10,
                                   codec="libx264", macro_block_size=1)
    try:
        for index, seed in enumerate(seeds):
            live.control()
            obs, info = env.reset(seed=int(seed))
            trace = [np.asarray(info["position"]).tolist()]
            goal = np.asarray(info["goal"]).tolist()
            steps, reward_sum, path_length = 0, 0., 0.
            while True:
                obs, reward, terminated, truncated, info = env.step(model.predict(obs))
                position = np.asarray(info["position"]).tolist()
                path_length += float(np.linalg.norm(np.asarray(position) - trace[-1]))
                trace.append(position)
                steps += 1
                reward_sum += reward
                if steps % 30 == 0:
                    live.control()
                if index == 0:
                    live.frame(env, f"{TASKS[key][0]}：{kind} 學習策略 • seed {seed} • 前視／向下／總覽")
                    if writer:
                        writer.append_data(np.concatenate((env.render("fpv"), env.render("down"),
                                                           env.render("overview")), axis=1))
                if terminated or truncated:
                    break
            episodes.append(dict(seed=int(seed), success=bool(info.get("is_success", False)),
                                  collision=bool(info.get("collision", False)),
                                  failure_reason=info.get("failure_reason") or ("timeout" if truncated else ""),
                                  steps=steps, seconds=safe_number(info.get("time")),
                                  reward=float(reward_sum), final_distance=safe_number(info.get("distance_to_goal")),
                                  min_clearance=None if key == "landing" else safe_number(info.get("min_clearance")),
                                  touchdown_speed=safe_number(info.get("touchdown_speed")),
                                  touchdown_horizontal_speed=safe_number(info.get("touchdown_horizontal_speed")),
                                  touchdown_tilt_deg=safe_number(info.get("touchdown_tilt_deg")),
                                  stable_contact_seconds=safe_number(info.get("stable_contact_seconds")),
                                  phase=info.get("phase"), goal=goal, path_length_m=path_length,
                                  trajectory_xyz=trace))
            live.update(progress=index + 1)
    finally:
        if writer:
            writer.close()
    success = float(np.mean([row["success"] for row in episodes]))
    collision = float(np.mean([row["collision"] for row in episodes]))
    report = dict(task=key, label=TASKS[key][0], kind=kind, episodes=episodes,
                  success_rate=success, collision_rate=collision,
                  policy="trained neural MLP; privileged state + obstacle geometry; fixed attitude controller",
                  teacher_used_in_evaluation=False, flyvis_used=False, gaussian_scene_used=False,
                  video_seed=int(seeds[0]) if video else None,
                  pass_gate={"minimum_episodes": 50, "success_rate_at_least": .8, "collision_rate_at_most": .1},
                  passed=len(episodes) >= 50 and success >= .8 and collision <= .1)
    atomic_json(live.out / f"{key}-{kind}.json", report)
    live.record(dict(task=key, phase=kind, samples=live.status["steps"],
                     success_rate=success, collision_rate=collision, episodes=len(episodes)))
    live.log(f"{TASKS[key][0]} {kind}：成功 {sum(e['success'] for e in episodes)}/{len(episodes)}，"
             f"碰撞 {sum(e['collision'] for e in episodes)}/{len(episodes)}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--dagger-samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--validation-episodes", type=int, default=12)
    parser.add_argument("--audit-episodes", type=int, default=50)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.run_id):
        parser.error("run-id must be 1–80 letters, digits, underscores or hyphens")
    if min(args.samples, args.dagger_samples, args.epochs, args.validation_episodes, args.audit_episodes) < 1 or args.rounds < 0:
        parser.error("counts must be positive; rounds may be zero")
    torch.manual_seed(617)
    np.random.seed(617)
    live = Live(args.run_id, args.tasks)
    manifest = dict(arguments=vars(args), started_at=time.time(), observations="52 privileged simulator features",
                    algorithm="supervised imitation + DAgger; no reinforcement learning in this run",
                    pretrained_flyvis_used=False, files={})
    for file in [Path(__file__), ROOT / "src/fruitfly_sim/mission_env.py"]:
        manifest["files"][str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
    atomic_json(live.out / "run-config.json", manifest)
    previous_obstacle = None
    try:
        for key in args.tasks:
            index = list(TASKS).index(key)
            live.update(task=key, task_label=TASKS[key][0], loss=None, steps=0)
            live.task(key, state="running")
            live.log(f"開始 {TASKS[key][0]}；模擬場景及位置／幾何基準")
            env = make_env(key)
            model = MissionPolicy(env.observation_space.shape[0])
            if key.startswith("obstacles") and previous_obstacle is not None:
                model.load_state_dict(previous_obstacle)
            xs, ys, collections = [], [], []
            try:
                for iteration in range(args.rounds + 1):
                    frames = args.samples if iteration == 0 else args.dagger_samples
                    collections.append(collect(env, model, key, live, xs, ys, frames,
                                               1000 + index * 10000 + iteration * 1000,
                                               1.0 if iteration == 0 else .25, iteration))
                    loss = fit(model, xs, ys, key, live, args.epochs, iteration)
                    torch.save({"state_dict": model.state_dict(), "observation_size": env.observation_space.shape[0],
                                "action_size": 3, "task": key, "samples": len(xs), "iteration": iteration,
                                "flyvis_used": False}, live.ckpt / f"{key}.pt")
                    report = evaluate(env, model, key, range(600000 + index * 1000,
                                      600000 + index * 1000 + args.validation_episodes), live, "validation")
                    atomic_json(live.out / f"{key}-training.json", dict(collections=collections, samples=len(xs),
                                iteration=iteration, loss=loss, training_seed_ranges=[[c["first_seed"], c["last_seed"]] for c in collections]))
                    if report["success_rate"] >= .9 and report["collision_rate"] <= .05:
                        break
                report = evaluate(env, model, key, range(800000 + index * 1000,
                                  800000 + index * 1000 + args.audit_episodes), live, "audit", video=True)
                live.task(key, state="passed" if report["passed"] else "needs_work",
                          success_rate=report["success_rate"], collision_rate=report["collision_rate"],
                          episodes=len(report["episodes"]),
                          result="通過 50 個新種子驗收" if report["passed"] else "已完成本輪，未達驗收門檻")
                if key.startswith("obstacles"):
                    previous_obstacle = {k: v.detach().clone() for k, v in model.state_dict().items()}
            finally:
                env.close()
        live.update(state="completed", phase="completed", message="本輪訓練及獨立驗收已完成；各任務是否過關見結果")
    except InterruptedError as exc:
        live.log(str(exc))
        live.update(state="stopped", message=str(exc))
    except Exception as exc:
        traceback.print_exc()
        live.update(state="error", message=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
