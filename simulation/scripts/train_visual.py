#!/usr/bin/env python3
"""Bounded M4 imitation learning: frozen Flyvis vision → learned action head.

The head additionally receives relative goal (GPS), velocity (IMU/estimator),
and its previous action. It receives no ranges or obstacle coordinates. The
teacher is used for training labels only, never validation/final policy actions.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import numpy as np
import torch
from torch import nn
torch.set_num_threads(2)
from fruitfly_sim.flyvis_features import FlyvisFeatureExtractor
from fruitfly_sim.visual_env import VisualNavigationEnv
from train_curriculum import Live, OUT, CKPT, atomic_json


class VisualHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(78, 128), nn.Tanh(), nn.Linear(128, 128),
                                 nn.Tanh(), nn.Linear(128, 2), nn.Tanh())

    def forward(self, x):
        return self.net(x)

    def predict(self, obs):
        with torch.no_grad():
            return self(torch.as_tensor(obs, dtype=torch.float32)).numpy()


def collect(env, model, frames, seed_start, live, xs, ys, *, teacher_mix, label):
    rng = np.random.default_rng(seed_start)
    episode = 0
    obs, _ = env.reset(seed=seed_start)
    started = time.monotonic()
    for index in range(frames):
        if index % 50 == 0 and not live.control():
            raise InterruptedError("Stopped during visual demonstrations")
        teacher = env.expert_action()
        xs.append(obs.copy()); ys.append(teacher.copy())
        policy = model.predict(obs)
        action = teacher if rng.random() < teacher_mix else policy
        action = np.clip(action + rng.normal(0, 0.07, 2), -1, 1)
        obs, _, terminated, truncated, _ = env.step(action)
        if index % 75 == 0:
            live.update(state="training", timesteps=len(xs), fps=round((index + 1) / max(time.monotonic() - started, 1e-3), 1),
                        message=f"M4 {label}：{index + 1}/{frames} 幀；累計 {len(xs)}")
            live.frame(env, f"M4 {label}（示範／DAgger 資料收集，非最終測試）")
        if terminated or truncated:
            episode += 1
            obs, _ = env.reset(seed=seed_start + episode)
    return episode + 1


def fit(model, xs, ys, live, epochs=70):
    x = torch.as_tensor(np.asarray(xs), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(ys), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)
    model.train()
    losses = []
    for epoch in range(epochs):
        if not live.control():
            raise InterruptedError("Stopped during visual head fitting")
        losses = []
        for indices in torch.randperm(len(x)).split(256):
            loss = nn.functional.mse_loss(model(x[indices]), y[indices])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % 10 == 0:
            live.log(f"M4 訓練決策網絡 {epoch + 1}/{epochs} epochs；action MSE {np.mean(losses):.5f}；Flyvis 權重凍結")
    model.eval()
    return float(np.mean(losses))


def evaluate(env, model, seeds, live, kind, *, ablate=False, video=False):
    import imageio.v2 as imageio
    live.update(state="evaluating", message=f"M4 {kind}：只用學習策略控制")
    episodes = []
    writer = imageio.get_writer(OUT / "M4-evaluation.mp4", fps=10, codec="libx264", macro_block_size=1) if video else None
    try:
        for episode_index, seed in enumerate(seeds):
            if not live.control():
                raise InterruptedError("Stopped during visual evaluation")
            obs, _ = env.reset(seed=seed)
            total = 0.0
            while True:
                if ablate:
                    obs = obs.copy(); obs[:72] = 0
                action = model.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                if episode_index == 0 and info["episode_steps"] % 4 == 0:
                    live.frame(env, f"M4 {kind}｜學習策略，無示範控制器｜seed {seed}")
                if writer and episode_index == 0:
                    writer.append_data(np.concatenate((env.render("fpv"), env.render("overview")), axis=1))
                if info["episode_steps"] % 50 == 0 and not live.control():
                    raise InterruptedError("Stopped during visual evaluation")
                if terminated or truncated:
                    break
            episodes.append({"seed": int(seed), "success": bool(info["is_success"]),
                             "collision": bool(info["collision"]), "reward": total,
                             "steps": info["episode_steps"], "distance_to_goal": info["distance_to_goal"]})
        report = {"stage": "M4", "task": "forward visual obstacle navigation with GPS/velocity assistance",
                  "kind": kind, "timesteps": live.status["timesteps"], "episodes": episodes,
                  "success_rate": float(np.mean([e["success"] for e in episodes])),
                  "collision_rate": float(np.mean([e["collision"] for e in episodes])),
                  "mean_reward": float(np.mean([e["reward"] for e in episodes])),
                  "observations": "72 frozen streaming Flyvis T4/T5 features + relative GPS goal xy + estimated velocity xy + previous command xy",
                  "obstacle_geometry_in_policy_observation": False, "teacher_used_in_evaluation": False,
                  "flyvis_trainable": False, "learned_parameters": "78→128→128→2 action head",
                  "visual_features_ablated": ablate,
                  "scope": "Small synthetic forward-facing task, one randomized cylinder; not long-distance flight, vision-only navigation or 4DGS"}
        atomic_json(OUT / f"M4-{kind}.json", report)
        live.record(report)
        live.log(f"M4 {kind}：成功 {report['success_rate']:.0%}，碰撞 {report['collision_rate']:.0%}")
        return report
    finally:
        if writer:
            writer.close()


def qualifies(report):
    return report["success_rate"] >= 0.8 and report["collision_rate"] <= 0.1


def wait_for_baselines():
    print("Waiting for M1–M3 complete before taking dashboard ownership", flush=True)
    while True:
        try:
            status = json.loads((OUT / "status.json").read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(2)
            continue
        passed = any(item["id"] == "M3" and item["status"] == "passed" for item in status.get("milestones", []))
        if passed and status.get("state") == "complete":
            return
        if status.get("state") in {"stopped", "error", "blocked", "failed"}:
            raise SystemExit("Baseline run did not complete; visual training was not started")
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-baselines", action="store_true")
    parser.add_argument("--frames", type=int, default=4000)
    parser.add_argument("--dagger-frames", type=int, default=1500)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if args.wait_for_baselines:
        wait_for_baselines()
    torch.manual_seed(81); np.random.seed(81)
    live = Live()
    live.start -= float(live.status.get("elapsed_seconds", 0))
    live.milestone("M4", "running")
    live.update(stage="M4", stage_title="果蠅視覺 → 學習決策", state="training", timesteps=0,
                stage_budget=args.frames + args.rounds * args.dagger_frames,
                algorithm="模仿學習（需要時加 DAgger）；Flyvis 視覺凍結",
                observation_mode="72 Flyvis 相機特徵 + GPS 相對目標 + 速度 + 上次動作；無障礙座標／射線",
                success_rate=None, collision_rate=None,
                message="M4 初始化果蠅視覺；相機 + GPS/速度，不提供障礙幾何")
    env = None
    xs, ys = [], []
    model = VisualHead()
    report = None
    try:
        extractor = FlyvisFeatureExtractor(cpu_threads=2)
        env = VisualNavigationEnv(extractor)
        live.log("M4 開始：只訓練決策網絡；Flyvis 保持時序神經狀態；policy 不讀取距離射線／障礙座標")
        collect(env, model, args.frames, 50000, live, xs, ys, teacher_mix=1.0, label="示範預訓練")
        mse = fit(model, xs, ys, live)
        best_score = (-1, -1)
        best_weights = None
        dagger_rounds_run = 0
        for round_index in range(args.rounds + 1):
            live.update(timesteps=len(xs))
            report = evaluate(env, model, range(60000, 60010), live, f"validation-{round_index}")
            score = (report["success_rate"], -report["collision_rate"])
            if score > best_score:
                best_score = score
                best_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
            if qualifies(report) or round_index == args.rounds:
                break
            dagger_rounds_run += 1
            collect(env, model, args.dagger_frames, 51000 + round_index * 1000, live, xs, ys,
                    teacher_mix=0.45 if round_index == 0 else 0.2, label=f"DAgger 第 {round_index + 1} 輪")
            mse = fit(model, xs, ys, live, epochs=55)
        model.load_state_dict(best_weights)
        torch.save({"state_dict": model.state_dict(), "input_dimension": 78, "training_samples": len(xs),
                    "architecture": [78, 128, 128, 2], "flyvis_model": "flow/0000/000",
                    "flyvis_frozen": True, "observations": "72 Flyvis + 2 goal + 2 velocity + 2 previous command",
                    "normalization": "Flyvis fixed gray-centered tanh; relative goal /4; velocity /0.8",
                    "training": "imitation + bounded DAgger; teacher absent in evaluation"}, CKPT / "M4-visual-head.pt")
        np.savez_compressed(OUT / "M4-demonstrations.npz", observations=np.asarray(xs), actions=np.asarray(ys))
        audit = evaluate(env, model, range(70000, 70020), live, "final-audit", video=True)
        audit_preview = (OUT / "live.jpg").read_bytes()
        ablation = evaluate(env, model, range(70000, 70020), live, "camera-ablation", ablate=True)
        preview_tmp = OUT / "live.tmp"
        preview_tmp.write_bytes(audit_preview)
        preview_tmp.replace(OUT / "live.jpg")
        live.update(preview_label="M4 已完成學習策略評估｜左 FPV・右總覽", frame_timestamp=time.time())
        atomic_json(OUT / "M4-training.json", {"samples": len(xs), "final_training_action_mse": mse,
                    "algorithm": "imitation" if dagger_rounds_run == 0 else "imitation + DAgger",
                    "dagger_rounds_run": dagger_rounds_run, "maximum_dagger_rounds": args.rounds,
                    "frozen_flyvis": True, "audit_seeds_used_for_training": False,
                    "passed": qualifies(audit), "camera_ablation_success_drop": audit["success_rate"] - ablation["success_rate"],
                    "checkpoint": str(CKPT / "M4-visual-head.pt")})
        if qualifies(audit):
            live.milestone("M4", "passed", f"20 個未見 seed：成功 {audit['success_rate']:.0%}，碰撞 {audit['collision_rate']:.0%}")
            live.update(state="complete", success_rate=audit["success_rate"], collision_rate=audit["collision_rate"],
                        mean_reward=audit["mean_reward"], message="M1–M4 完成；M4 係 Flyvis 視覺 + GPS/速度輔助，決策網絡已學習")
        else:
            live.milestone("M4", "blocked", f"未達標：成功 {audit['success_rate']:.0%}，碰撞 {audit['collision_rate']:.0%}")
            live.update(state="blocked", success_rate=audit["success_rate"], collision_rate=audit["collision_rate"],
                        message="M4 已完成有上限訓練，但未達 80% 成功／≤10% 碰撞；結果及模型已保存")
    except InterruptedError as error:
        torch.save(model.state_dict(), CKPT / "M4-interrupted.pt")
        live.update(state="stopped", message=str(error))
    except Exception as error:
        live.update(state="error", message=f"M4 {type(error).__name__}: {error}")
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
