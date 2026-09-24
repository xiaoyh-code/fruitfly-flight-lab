#!/usr/bin/env python3
"""Small runnable example: two RGB images -> two frozen Flyvis feature vectors.

Run from any directory with the project's Python environment. This needs the
official weights from download_flyvis_models.py, but no renderer or Hall scene.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")

    model_dir = ROOT / "data/flyvis/results/flow/0000/000"
    if not model_dir.is_dir():
        print("Missing official pretrained Flyvis weights. Run:\n"
              "  python scripts/download_flyvis_models.py", file=sys.stderr)
        return 1

    import numpy as np
    from fruitfly_sim.flyvis_features import FlyvisFeatureExtractor

    extractor = FlyvisFeatureExtractor(cpu_threads=args.cpu_threads)
    # RGB, uint8, H x W x 3. These two frames show a bright bar moving right.
    frames = []
    for left in (110, 130):
        rgb = np.full((240, 320, 3), 25, dtype=np.uint8)
        rgb[:, left:left + 24, :] = 230
        frames.append(rgb)

    extractor.reset()  # Once at the START of an episode, not between frames.
    features = []
    for index, rgb in enumerate(frames, start=1):
        feature = extractor.transform(rgb)
        if feature.shape != (72,) or not np.isfinite(feature).all():
            raise RuntimeError("Expected 72 finite Flyvis features")
        features.append(feature)
        print(f"Frame {index}: RGB {rgb.shape} {rgb.dtype} -> "
              f"features {feature.shape} {feature.dtype}; "
              f"finite=True; range=[{feature.min():.6f}, {feature.max():.6f}]")

    # Reset must restore a new episode rather than carry the old neural state.
    extractor.reset()
    first_again = extractor.transform(frames[0])
    if not np.allclose(features[0], first_again, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Episode reset did not reproduce the first frame")
    print("PASS: two frames produced 72 finite features each; episode reset reproduced frame 1.")
    print("These are frozen visual-circuit features, not distance estimates or flight commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
