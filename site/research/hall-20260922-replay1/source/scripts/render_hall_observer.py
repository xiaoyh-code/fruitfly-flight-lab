#!/usr/bin/env python3
"""Replay one saved hall policy trial through MuJoCo with larger observer cameras.

This does not train or rewrite the original audit. The policy still consumes the
original 320x240 WebGL camera. Only independent display cameras render at 960x540.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from fruitfly_sim.flyvis_features import FlyvisFeatureExtractor
from fruitfly_sim.observer_view import DroneObserver
from fruitfly_sim.splat_env import SplatNavigationEnv, WebSplatRenderer
from train_visual import VisualHead
from train_missions import atomic_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id')
    parser.add_argument('--port', type=int, default=8770)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    run_id = args.run_id or json.loads((ROOT / 'outputs/missions/latest.json').read_text())['run_id']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        parser.error('Invalid run id')
    out = ROOT / 'outputs/missions' / run_id
    report_path = out / 'hall_avoidance-audit.json'
    audit = json.loads(report_path.read_text())
    state = json.loads((out / 'status.json').read_text())
    if state.get('state') not in {'complete', 'completed'}:
        parser.error('Wait until the hall run completes before replaying its audit')
    seed = args.seed if args.seed is not None else audit['episodes'][0]['seed']
    reference = next((e for e in audit['episodes'] if e['seed'] == seed), None)
    if reference is None:
        parser.error('Choose a seed from the saved audit')
    checkpoint_path = ROOT / 'checkpoints/missions' / run_id / 'hall-visual-head.pt'
    original_hashes = {str(path): sha(path) for path in [report_path, checkpoint_path]}
    model = VisualHead()
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=False)['state_dict'])
    model.eval()
    torch.set_num_threads(2)
    renderer = WebSplatRenderer(port=args.port)
    env = SplatNavigationEnv(FlyvisFeatureExtractor(cpu_threads=2), renderer)
    # Match the corrected cylinder broad-phase bounds used by the saved run.
    gid = env._obstacle_geom
    env.model.geom_aabb[gid, 3:] = env.model.geom_size[gid, [0, 0, 1]]
    observer = DroneObserver(env, width=960, height=540)
    writers, temporary, snapshots = {}, {}, {}
    trace = []
    steps = 0
    print(f'Replaying seed {seed}: display cameras only; no training', flush=True)
    try:
        for view in ['follow', 'overview']:
            temporary[view] = out / f'.observer-{view}-recording.mp4'
            writers[view] = imageio.get_writer(temporary[view], fps=10, codec='libx264',
                                               macro_block_size=1, quality=8)
        obs, info = env.reset(seed=seed)
        trace.append(info['position'].tolist())
        while True:
            obs, _, terminated, truncated, info = env.step(model.predict(obs))
            steps += 1
            trace.append(info['position'].tolist())
            for view in writers:
                pixels = observer.render(view)
                cv2.putText(pixels, f'MuJoCo {view} | saved policy replay | seed {seed}', (16, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 230), 1, cv2.LINE_AA)
                writers[view].append_data(pixels)
                if steps == max(1, reference['steps'] // 2):
                    snapshots[view] = pixels.copy()
            if terminated or truncated:
                break
    finally:
        for writer in writers.values():
            writer.close()
        observer.close()
        env.close()
        renderer.close()
    for view, file in temporary.items():
        file.replace(out / f'observer-{view}.mp4')
        if view not in snapshots:
            raise RuntimeError('Replay finished before the expected snapshot')
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(snapshots[view], cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            raise RuntimeError('Could not encode observer snapshot')
        (out / f'observer-{view}.jpg').write_bytes(encoded.tobytes())
    original = np.asarray(reference['trajectory_xyz'])
    replay = np.asarray(trace)
    maximum_error = float(np.max(np.linalg.norm(original - replay, axis=1))) if original.shape == replay.shape else None
    if any(sha(Path(path)) != value for path, value in original_hashes.items()):
        raise RuntimeError('Original audit or checkpoint changed')
    metadata = dict(kind='evaluation_replay', run_id=run_id, seed=seed, frames=steps, fps=10,
                    width=960, height=540, updated_at=time.time(),
                    success=bool(info['is_success']), collision=bool(info['collision']),
                    final_distance=float(info['distance_to_goal']),
                    source_trajectory_max_error_m=maximum_error,
                    checkpoint_sha256=original_hashes[str(checkpoint_path)],
                    source_audit_sha256=original_hashes[str(report_path)],
                    physics_and_policy_unchanged=True, policy_rgb_size=[320, 240],
                    camera_distance_m=.8, display_only=True,
                    note='Separate observer cameras; same saved policy/seed rerun without training.')
    atomic_json(out / 'observer-preview.json', metadata)
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
