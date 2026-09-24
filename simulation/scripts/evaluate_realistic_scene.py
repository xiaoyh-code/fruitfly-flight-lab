#!/usr/bin/env python3
"""Captured 3DGS pilot: unchanged M4 transfer and independent distance probe data.

Only RGB feeds Flyvis. Policy retains its original goal/velocity/previous-action
assistance. Approximate obstacle geometry is used for labels/evaluation only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from fruitfly_sim.flyvis_features import FlyvisFeatureExtractor
from fruitfly_sim.splat_env import SplatNavigationEnv,WebSplatRenderer
from fruitfly_sim.unity_renderer import UnitySplatRenderer,look_at
from train_visual import VisualHead

OUT=ROOT/"outputs/realistic"


def atomic_json(path,value):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def status(phase,message,completed=0,total=0,**extra):
    print(f"{phase}: {message} ({completed}/{total})",flush=True)
    atomic_json(OUT/"status.json",dict(phase=phase,message=message,completed=completed,total=total,updated=time.time(),output_dir=str(OUT),**extra))


def live(rgb):
    cv2.imwrite(str(OUT/"live.tmp.jpg"),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    (OUT/"live.tmp.jpg").replace(OUT/"live.jpg")


def inspect(renderer):
    # View both sides and front before authorizing the approximate collision proxy.
    positions=[[13.55,-3.15,1.2],[15.15,-4.45,1.2],[16.75,-3.15,1.2],[15.15,-1.45,1.2]]
    tiles=[]
    for index,pos in enumerate(positions):
        target=[15.15,-2.95,1.0]
        frame=renderer.render(*look_at(pos,target))
        cv2.imwrite(str(OUT/f"obstacle-view-{index}.png"),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        tile=cv2.resize(frame,(640,480))
        cv2.putText(tile,f"camera {pos}",(12,28),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,230,150),2)
        tiles.append(tile)
    image=np.concatenate([np.concatenate(tiles[:2],axis=1),np.concatenate(tiles[2:],axis=1)],axis=0)
    cv2.imwrite(str(OUT/"obstacle-inspection.png"),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    status("inspection","已輸出四個方向嘅實際 3DGS 相機畫面；等候核對簡化碰撞範圍",4,4)


def audit(renderer,extractor,episodes):
    env=SplatNavigationEnv(extractor,renderer)
    model=VisualHead()
    path=ROOT/"checkpoints/training/M4-visual-head.pt"
    checkpoint=torch.load(path,map_location="cpu",weights_only=False)
    model.load_state_dict(checkpoint["state_dict"]);model.eval()
    modes=["m4","frozen_vision","direct_to_goal","geometry_teacher"]
    reports={}
    video=imageio.get_writer(OUT/"avoidance-pilot.mp4",fps=10,codec="libx264",macro_block_size=1)
    try:
        for mode in modes:
            results=[]
            for episode in range(episodes):
                seed=87000+episode
                obs,_=env.reset(seed=seed)
                frozen=obs[:72].copy()
                path_length=0.;last=env.data.qpos[:2].copy();positions=[]
                status("avoidance",f"{mode}：原模型／對照測試，未作微調",episode,episodes,mode=mode)
                for step in range(env.max_episode_steps):
                    if mode in {"m4","frozen_vision"}:
                        policy_obs=obs.copy()
                        if mode=="frozen_vision":policy_obs[:72]=frozen
                        action=model.predict(policy_obs)
                    elif mode=="direct_to_goal":
                        action=(env.goal-env.data.qpos[:2])*1.6/env.SPEED
                    else:action=env.expert_action()
                    obs,_,terminated,truncated,info=env.step(action)
                    position=env.data.qpos[:2].copy();path_length+=float(np.linalg.norm(position-last));last=position
                    positions.append(position.tolist())
                    if step%5==0:live(env.latest_rgb)
                    if episode==0:
                        frame=np.concatenate([env.latest_rgb,env.render("overview")],axis=1)
                        # Labels stay on display/video only, never camera frames passed to policy.
                        cv2.putText(frame,f"{mode} | captured RGB / approximate physics proxy",(7,16),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,240,140),1)
                        video.append_data(frame)
                    if terminated or truncated:break
                results.append(dict(seed=seed,success=bool(info["is_success"]),collision=bool(info["collision"]),
                    timeout=bool(truncated),steps=step+1,path_length_m=path_length,
                    crossed_object_plane_without_collision=bool(not info["collision"] and max(p[0] for p in positions)>env.PROXY_CENTER[0]+env.OBSTACLE_RADIUS+env.VEHICLE_RADIUS),
                    out_of_bounds=bool(info.get("out_of_bounds",False)),
                    approximate_sampled_minimum_envelope_clearance_m=env.minimum_clearance,
                    end_distance_to_goal=float(info["distance_to_goal"]),trajectory_local_xy=positions))
            reports[mode]=dict(episodes=results,success_count=sum(x["success"] for x in results),
                collision_count=sum(x["collision"] for x in results),timeout_count=sum(x["timeout"] for x in results),
                crossed_object_plane_without_collision_count=sum(x["crossed_object_plane_without_collision"] for x in results),total=episodes)
            atomic_json(OUT/"avoidance-audit.json",dict(reports=reports,
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checkpoint_unchanged=True,
                output_dir=str(OUT),
                backend=type(renderer).__name__,camera="320x240 RGB, 85 degree vertical FOV, sim dt .1s",
                trial_horizon_seconds=12,clearance_sample_rate_hz=10,
                secondary_trajectory_diagnostic="Crossing local x=.54 (proxy far side plus vehicle envelope) with no proxy collision; does not replace goal-reaching success. Added after inspecting pilot trajectories.",
                geometry_in_policy=False,teacher_only_in_explicit_baseline=True,
                limitations=["One captured static scene, one manually fitted collision cylinder, no measured hall mesh.",
                             "Proxy collision and clearance, not validated physical safety.",
                             "Frozen-vision control freezes neural features at first frame (not a frozen camera video).",
                             "Original policy retains ideal relative goal, velocity and previous-action inputs."]))
    finally:video.close();env.close()
    return reports


def distance_sequences(renderer,extractor):
    rng=np.random.default_rng(19260921)
    fields={name:[] for name in ("features","velocity","distance","trajectory","split")}
    center=np.array([15.15,-2.95])
    forward=np.array([np.cos(np.deg2rad(8)),0,-np.sin(np.deg2rad(8))])
    up=np.array([np.sin(np.deg2rad(8)),0,np.cos(np.deg2rad(8))])
    plans=["train"]*24+["val"]*8+["test"]*12
    records=[]
    for trajectory,split in enumerate(plans):
        speed=float(rng.uniform(.25,.75))
        lateral=float(rng.uniform(-.48,.48))
        initial_x=float(rng.uniform(1.75,2.9))
        final_x=.68
        steps=int(np.ceil((initial_x-final_x)/(speed*.1)))
        extractor.reset()
        for i in range(steps):
            x=initial_x-speed*.1*i
            pos=np.array([center[0]-x,center[1]+lateral,1.225])
            frame=renderer.render(pos,forward,up)
            features=extractor.transform(frame)
            # Discard onset transient. No frame, position, time or label enters a readout input.
            if i>=5:
                fields["features"].append(features.copy())
                fields["velocity"].append([speed,0.])
                fields["distance"].append(float(np.hypot(x,lateral)-.45))
                fields["trajectory"].append(trajectory)
                fields["split"].append(split)
            if i%10==0:live(frame)
        records.append(dict(id=trajectory,split=split,speed=speed,lateral=lateral,initial_x=initial_x,frames=steps))
        status("distance_data",f"獨立遠近測試：{split} 軌跡；Flyvis 權重凍結",trajectory+1,len(plans))
    np.savez_compressed(OUT/"distance-sequences.npz",**{k:np.asarray(v) for k,v in fields.items()})
    atomic_json(OUT/"distance-data-provenance.json",dict(trajectories=records,
        source="robot_hall.ply real captured 3DGS",backend=type(renderer).__name__,output_dir=str(OUT),
        labels="Approximate horizontal camera-to-cylinder surface distance in scene coordinate metres; manual proxy radius .45, center [15.15,-2.95].",
        motion="Controlled kinematic camera approaches at .1s simulated time; independently varied speed, offset, start.",
        policy_training=False,flyvis_trainable=False,labels_in_features=False,
        split="Whole trajectories fixed before fitting. Single scene and same object; no cross-scene generalization claim."))


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend",choices=["webgl","unity"],default="webgl")
    parser.add_argument("--phase",choices=["inspect","avoidance","distance","all"],default="inspect")
    parser.add_argument("--episodes",type=int,default=20)
    parser.add_argument("--output-dir",type=Path,metavar="PATH",
        help="Directory for every experiment output (default: outputs/realistic for webgl; outputs/realistic/unity for unity)")
    args=parser.parse_args()
    default_output=ROOT/"outputs/realistic"
    if args.backend=="unity":default_output=default_output/"unity"
    OUT=(args.output_dir if args.output_dir is not None else default_output).expanduser().resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    print(f"Renderer backend: {args.backend}; all experiment outputs: {OUT}",flush=True)
    renderer=WebSplatRenderer() if args.backend=="webgl" else UnitySplatRenderer()
    try:
        if args.phase=="inspect":inspect(renderer);return
        extractor=FlyvisFeatureExtractor(cpu_threads=2)
        if args.phase in {"avoidance","all"}:audit(renderer,extractor,args.episodes)
        if args.phase in {"distance","all"}:distance_sequences(renderer,extractor)
        atomic_json(OUT/"renderer-performance.json",dict(backend=type(renderer).__name__,frames=len(renderer.latencies),
            output_dir=str(OUT),
            median_ms=float(np.median(renderer.latencies)*1000),p95_ms=float(np.percentile(renderer.latencies,95)*1000)))
        status("complete","本輪測試已完成；數據已儲存",1,1)
    except Exception as exc:
        status("error",str(exc));raise
    finally:renderer.close()


if __name__=="__main__":main()
