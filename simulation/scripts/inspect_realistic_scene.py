#!/usr/bin/env python3
"""Capture actual Unity 3DGS views before selecting a collision proxy."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cv2
import numpy as np
from fruitfly_sim.unity_renderer import UnitySplatRenderer, look_at


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    out = ROOT / "outputs/realistic"
    out.mkdir(parents=True, exist_ok=True)
    views = []
    with UnitySplatRenderer(port=args.port) as renderer:
        for i, (pos, yaw) in enumerate([([0,0,1.2], 0), ([0,0,1.2], 90),
                                        ([0,0,1.2], 180), ([0,0,1.2], 270),
                                        ([7,0,1.2], 0), ([-7,0,1.2], 180),
                                        ([0,2,1.2], 90), ([0,-2,1.2], 270)]):
            angle = np.deg2rad(yaw)
            target = np.asarray(pos) + [np.cos(angle), np.sin(angle), -0.14]
            frame = renderer.render(*look_at(pos, target))
            cv2.imwrite(str(out / f"hall-view-{i}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            tile = cv2.resize(frame, (480,360))
            cv2.putText(tile, f"{i}: {pos} yaw {yaw}", (12,24), cv2.FONT_HERSHEY_SIMPLEX, .52, (255,180,20), 2)
            views.append(tile)
        montage = np.concatenate([np.concatenate(views[:4],axis=1),np.concatenate(views[4:],axis=1)],axis=0)
        cv2.imwrite(str(out / "hall-inspection.png"), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
        (out / "renderer-check.json").write_text(json.dumps(dict(frames=len(views),
            camera="320x240 RGB, vertical FOV85", scene="robot_hall.ply", renderer="Unity Metal 3DGS",
            mean_latency_ms=float(np.mean(renderer.latencies)*1000)),indent=2), encoding='utf-8')
    print(out / "hall-inspection.png")


if __name__ == "__main__":
    main()
