#!/usr/bin/env python3
"""Train and audit small navigation policies; publish honest local live metrics.

M1-M3 are privileged state/synthetic-range baselines, not visual navigation.
The MuJoCo attitude controller remains fixed. Policy actions are planar velocity.
"""
from __future__ import annotations
import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import numpy as np
import torch
torch.set_num_threads(2)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from fruitfly_sim.learning_env import NavigationLearningEnv

OUT = ROOT / 'outputs/training'
CKPT = ROOT / 'checkpoints/training'
STAGES = {'reach': ('M1', '學識去目標'), 'avoid': ('M2', '學識避障'), 'return': ('M3', '出發再返航')}

def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)

class Live:
    def __init__(self):
        OUT.mkdir(parents=True, exist_ok=True)
        CKPT.mkdir(parents=True, exist_ok=True)
        status_path, history_path = OUT / 'status.json', OUT / 'history.json'
        if status_path.is_file():
            self.status = json.loads(status_path.read_text(encoding='utf-8'))
        else:
            titles = ['Environment checks', 'Reach a goal', 'Avoid an obstacle', 'Return home', 'Flyvis visual navigation']
            self.status = dict(state='idle', stage='', log=[], elapsed_seconds=0,
                               message='New workspace; no training results yet',
                               milestones=[dict(id=f'M{i}', title=title, status='pending', result='')
                                           for i, title in enumerate(titles)])
            atomic_json(status_path, self.status)
        self.history = json.loads(history_path.read_text(encoding='utf-8')) if history_path.is_file() else []
        if not history_path.is_file():
            atomic_json(history_path, self.history)
        self.start = time.monotonic()
        self.last_control = time.time()
        self.stopped = False
        self.stage_start_steps = 0
        self.recent = deque(maxlen=12)
    def update(self, **fields):
        self.status.update(fields)
        self.status['elapsed_seconds'] = round(time.monotonic() - self.start, 1)
        self.status['updated_at'] = time.time()
        atomic_json(OUT / 'status.json', self.status)
    def log(self, message):
        print(message, flush=True)
        self.status['log'] = (self.status.get('log', []) + [time.strftime('%H:%M:%S ') + message])[-30:]
        self.update(message=message)
    def milestone(self, mid, status, result=''):
        for item in self.status['milestones']:
            if item['id'] == mid:
                item.update(status=status, result=result)
        self.update()
    def control(self):
        path = OUT / 'control.json'
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            return not self.stopped
        stamp = value.get('timestamp', 0)
        if stamp <= self.last_control:
            return not self.stopped
        self.last_control = stamp
        command = value.get('command')
        if command == 'stop':
            self.stopped = True
            self.update(state='stopped', message='已停止；最近模型已保存')
            return False
        if command == 'pause':
            previous = self.status['state']
            self.update(state='paused', message='已暫停，按繼續可恢復')
            while True:
                time.sleep(0.2)
                try:
                    value = json.loads(path.read_text(encoding='utf-8'))
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
                if value.get('timestamp', 0) > self.last_control:
                    self.last_control = value['timestamp']
                    if value.get('command') == 'stop':
                        self.stopped = True
                        self.update(state='stopped', message='已停止')
                        return False
                    if value.get('command') == 'resume':
                        self.update(state=previous, message='繼續訓練')
                        break
        return True
    def frame(self, env, label):
        import cv2
        frame = np.concatenate((env.render('fpv'), env.render('overview')), axis=1)
        ok, data = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            tmp = OUT / 'live.tmp'
            tmp.write_bytes(data.tobytes())
            tmp.replace(OUT / 'live.jpg')
            self.update(frame_timestamp=time.time(), preview_label=label)
    def record(self, report):
        self.history.append({k: report[k] for k in ['stage', 'timesteps', 'success_rate', 'collision_rate', 'mean_reward']} | {'elapsed_seconds':round(time.monotonic()-self.start,1), 'kind': report.get('kind','validation')})
        atomic_json(OUT / 'history.json', self.history)
        self.update(success_rate=report['success_rate'], collision_rate=report['collision_rate'], mean_reward=report['mean_reward'])

def evaluate(model, stage, seeds, live, *, kind='validation', video=False):
    import imageio.v2 as imageio
    live.update(state='evaluating', message=f'{STAGES[stage][0]}：{kind} 獨立任務評估')
    env = NavigationLearningEnv(stage=stage)
    episodes = []
    writer = None
    if video:
        writer = imageio.get_writer(OUT / f'{STAGES[stage][0]}-evaluation.mp4', fps=10, codec='libx264', macro_block_size=1)
    try:
        for index, seed in enumerate(seeds):
            if not live.control():
                break
            obs, info = env.reset(seed=int(seed))
            total, length, done = 0., 0, False
            while not done:
                action = env.action_space.sample() if model is None else model.predict(obs, deterministic=True)[0]
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                length += 1
                done = terminated or truncated
                if index == 0 and length % 3 == 0:
                    live.frame(env, f'{STAGES[stage][0]} 最新策略測試｜左 FPV・右總覽｜seed {seed}')
                if writer and index == 0:
                    writer.append_data(np.concatenate((env.render('fpv'), env.render('overview')), axis=1))
                if length % 25 == 0 and not live.control():
                    done = True
            episodes.append({'seed':int(seed), 'success':bool(info.get('is_success', False)), 'collision':bool(info.get('collision',False)), 'reward':float(total), 'steps':length, 'distance_to_goal':float(info.get('distance_to_goal',np.nan))})
    finally:
        if writer:
            writer.close()
        env.close()
    if not episodes:
        raise InterruptedError('Stopped before evaluation')
    report = {'stage':STAGES[stage][0], 'task':stage, 'kind':kind, 'timesteps':live.status['timesteps'], 'episodes':episodes,
              'success_rate':float(np.mean([e['success'] for e in episodes])), 'collision_rate':float(np.mean([e['collision'] for e in episodes])),
              'mean_reward':float(np.mean([e['reward'] for e in episodes])), 'observations':'privileged simulator position + synthetic range',
              'policy':'random' if model is None else 'learned neural policy; fixed attitude stabilizer'}
    atomic_json(OUT / f'{STAGES[stage][0]}-{kind}.json', report)
    live.record(report)
    return report

def qualifies(report):
    return len(report['episodes']) >= 20 and report['success_rate'] >= 0.8 and report['collision_rate'] <= 0.1

def warm_start(model, stage, live, samples=4000):
    """Explicitly labelled imitation pretraining; no teacher is used at evaluation."""
    live.update(state='training', algorithm='示範預訓練 → PPO', message=f'{STAGES[stage][0]} 收集示範，初始化策略')
    env = NavigationLearningEnv(stage=stage)
    rng = np.random.default_rng(912)
    xs, ys = [], []
    obs, _ = env.reset(seed=300)
    episode = 0
    try:
        for i in range(samples):
            if i % 100 == 0 and not live.control():
                raise InterruptedError('Stopped during demonstrations')
            target = env.expert_action()
            xs.append(obs.copy()); ys.append(target.copy())
            action = np.clip(target + rng.normal(0, .15, size=2), -1, 1)
            obs, _, term, trunc, _ = env.step(action)
            if i % 200 == 0:
                live.update(message=f'示範預訓練：收集 {i}/{samples} 筆；測試時移除示範控制器')
                live.frame(env, '示範資料收集（尚未係學習策略）')
            if term or trunc:
                episode += 1
                obs, _ = env.reset(seed=300+episode)
    finally:
        env.close()
    x, y = torch.as_tensor(np.asarray(xs)), torch.as_tensor(np.asarray(ys))
    opt = torch.optim.Adam(model.policy.parameters(), lr=8e-4)
    model.policy.set_training_mode(True)
    for epoch in range(50):
        if not live.control():
            raise InterruptedError('Stopped during imitation')
        order = torch.randperm(len(x))
        losses=[]
        for idx in order.split(256):
            dist = model.policy.get_distribution(x[idx])
            loss = torch.nn.functional.mse_loss(dist.distribution.mean, y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        if epoch % 10 == 0:
            live.log(f'示範預訓練 {epoch+1}/50，action MSE {np.mean(losses):.4f}')
    with torch.no_grad():
        model.policy.log_std.fill_(-1.5)
    model.policy.set_training_mode(False)
    model.save(CKPT / f'{STAGES[stage][0]}-imitation')
    atomic_json(OUT / f'{STAGES[stage][0]}-imitation.json', {'samples':samples,'epochs':50,'final_action_mse':float(np.mean(losses)), 'teacher_used_in_evaluation':False})
    live.update(algorithm='PPO（示範初始化）')

class Progress(BaseCallback):
    def __init__(self, live, model_base_steps):
        super().__init__()
        self.live=live; self.last=0.; self.last_frame=0.; self.started=time.monotonic(); self.base_steps=model_base_steps
    def _on_step(self):
        now=time.monotonic()
        if now-self.last > 1:
            self.last=now
            self.live.update(state='training', timesteps=self.num_timesteps-self.base_steps, fps=round((self.num_timesteps-self.base_steps)/max(.1,now-self.started),1))
            if not self.live.control():
                return False
        if now-self.last_frame > 2.5:
            self.last_frame=now
            self.live.frame(self.training_env.envs[0], '即時訓練中：探索動作｜左 FPV・右總覽')
        return True

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages', nargs='+', choices=list(STAGES), default=list(STAGES))
    parser.add_argument('--budget', type=int, default=65536, help='PPO interaction cap per milestone')
    parser.add_argument('--chunk', type=int, default=8192)
    parser.add_argument('--warm-start', action='store_true', help='Use demonstrations before the first selected stage too')
    parser.add_argument('--load', type=Path, help='Continue a saved policy')
    args=parser.parse_args()
    live=Live()
    model=None
    last_path=args.load
    try:
        for stage in args.stages:
            mid,title=STAGES[stage]
            live.milestone(mid,'running')
            live.update(stage=mid,stage_title=title,timesteps=0,stage_budget=args.budget,state='training',algorithm='PPO / CPU',success_rate=None,collision_rate=None,mean_reward=None)
            live.log(f'{mid} 開始：{title}；validation seeds 10000–10019，final audit seeds 20000–20019')
            vec=DummyVecEnv([lambda s=stage:NavigationLearningEnv(stage=s) for _ in range(4)])
            vec.seed(42)
            if last_path:
                model=PPO.load(last_path,env=vec,device='cpu')
                model.learning_rate=2e-4
                model.lr_schedule=lambda remaining: 2e-4
            else:
                model=PPO('MlpPolicy',vec,learning_rate=3e-4,n_steps=256,batch_size=256,n_epochs=8,gamma=.98,gae_lambda=.95,ent_coef=.004,policy_kwargs={'net_arch':dict(pi=[64,64],vf=[64,64])},device='cpu',seed=42,verbose=0)
            initial=evaluate(model,stage,range(10000,10020),live,kind='initial')
            if args.warm_start or stage!='reach':
                warm_start(model,stage,live)
                initial=evaluate(model,stage,range(10000,10020),live,kind='after-imitation')
            best_score=(initial['success_rate'],-initial['collision_rate'],initial['mean_reward'])
            model.save(CKPT/f'{mid}-best')
            base_steps=model.num_timesteps
            progress=Progress(live,base_steps)
            passed=False
            for target in range(args.chunk,args.budget+args.chunk,args.chunk):
                if not live.control():
                    raise InterruptedError('Stopped')
                live.update(state='training',message=f'{mid} PPO 自主探索與參數更新')
                model.learn(total_timesteps=min(args.chunk,args.budget-(target-args.chunk)),reset_num_timesteps=False,callback=progress)
                model.save(CKPT/f'{mid}-latest')
                if live.stopped:
                    raise InterruptedError('Stopped')
                live.update(timesteps=model.num_timesteps-base_steps)
                report=evaluate(model,stage,range(10000,10020),live)
                score=(report['success_rate'],-report['collision_rate'],report['mean_reward'])
                if score>best_score:
                    best_score=score; model.save(CKPT/f'{mid}-best')
                live.log(f'{mid} {live.status["timesteps"]} steps：成功 {report["success_rate"]:.0%}，碰撞 {report["collision_rate"]:.0%}')
                if qualifies(report):
                    audit=evaluate(model,stage,range(20000,20020),live,kind='audit',video=True)
                    if qualifies(audit):
                        model.save(CKPT/f'{mid}-passed')
                        last_path=CKPT/f'{mid}-passed.zip'
                        live.milestone(mid,'passed',f'20個新任務：成功 {audit["success_rate"]:.0%}／碰撞 {audit["collision_rate"]:.0%}')
                        live.log(f'{mid} 獨立驗收通過。')
                        passed=True; break
            vec.close()
            if not passed:
                # Evaluate the best validation checkpoint if PPO degraded after imitation.
                best=PPO.load(CKPT/f'{mid}-best.zip',device='cpu')
                audit=evaluate(best,stage,range(20000,20020),live,kind='audit',video=True)
                if qualifies(audit):
                    best.save(CKPT/f'{mid}-passed'); last_path=CKPT/f'{mid}-passed.zip'; passed=True
                    live.milestone(mid,'passed',f'最佳保存策略：成功 {audit["success_rate"]:.0%}／碰撞 {audit["collision_rate"]:.0%}')
                else:
                    live.milestone(mid,'blocked',f'本輪上限後成功 {audit["success_rate"]:.0%}，需繼續訓練／調整')
                    live.update(state='failed',message=f'{mid} 未達驗收門檻；後續關卡未啟動')
                    return 2
        live.update(state='complete',message='本輪 M1–M3 已完成；M4 果蠅視覺由獨立階段接續',preview_label='最後一次獨立驗收畫面')
        return 0
    except InterruptedError:
        if model is not None:
            model.save(CKPT/'interrupted')
        live.update(state='stopped',message='已停止並保存模型')
        return 130
    except Exception as exc:
        live.update(state='failed',message=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        return 1

if __name__=='__main__':
    raise SystemExit(main())
