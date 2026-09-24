#!/usr/bin/env python3
"""Benchmark persistent-state Flyvis camera features on the local CPU."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fruitfly_sim.flyvis_features import FlyvisFeatureExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "outputs/flight-fpv.mp4")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/training/flyvis-benchmark.json")
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise SystemExit("Source FPS unavailable")
    source_index = 0
    frames = []
    # The existing demo is side-by-side, with the 320-pixel FPV image on the left.
    for frame_index in range(args.frames):
        requested = round(frame_index * fps / 10)
        while source_index <= requested:
            ok, frame = capture.read()
            if not ok:
                raise SystemExit("Video is shorter than the requested benchmark")
            source_index += 1
        frames.append(cv2.cvtColor(frame[:, :320], cv2.COLOR_BGR2RGB))
    capture.release()
    extractor = FlyvisFeatureExtractor(cpu_threads=args.threads)
    durations = []
    features = []
    for frame in frames:
        started = time.perf_counter()
        features.append(extractor.transform(frame))
        durations.append(time.perf_counter() - started)
    features = np.stack(features)
    final_state = extractor.state.nodes.activity.detach().clone()
    # Replay the first frame after reset: equality validates the episode reset,
    # while its difference from the next image catches accidental frame resets.
    extractor.reset()
    replay = extractor.transform(frames[0])
    reset_error = float(np.max(np.abs(replay - features[0])))
    # A static-input control separates the shared startup transient from motion.
    static_features = [replay]
    for _ in frames[1:]:
        static_features.append(extractor.transform(frames[0]))
    static_features = np.stack(static_features)
    motion_difference = float(np.max(np.abs(features - static_features)))
    # Independently compare the final streaming state to one uninterrupted
    # sequence. This would fail if transform() silently reset between frames.
    extractor.reset()
    with extractor.torch.no_grad():
        movie = extractor.torch.cat([
            extractor.retinal_input(frame).expand(1, extractor.substeps, 1, 721)
            for frame in frames
        ], dim=1)
        full_states = extractor.network.simulate(movie, extractor.neural_dt,
                                                initial_state=extractor.state, as_states=True)
        stream_equivalence_error = float((full_states[-1].nodes.activity - final_state).abs().max())
    duration = sum(durations)
    report = {
        "passed": bool(np.isfinite(features).all() and np.ptp(features, axis=0).max() > 1e-7
                       and reset_error < 1e-6 and motion_difference > 1e-7
                       and stream_equivalence_error < 1e-6),
        "scope": "Frozen pretrained Flyvis streaming T4/T5 features; no navigation training or claim of whole-brain learning",
        "model": "flow/0000/000", "device": "cpu", "cpu_threads": args.threads,
        "source": str(args.video.resolve()), "source_fps": fps, "source_crop": "left 320 pixels (FPV)",
        "camera_hz": 10, "neural_dt_seconds": extractor.neural_dt,
        "neural_substeps_per_camera_frame": extractor.substeps,
        "neural_state_persisted_between_frames": True,
        "retinal_elements": 721, "feature_shape": list(features.shape),
        "pooling": "8 T4/T5 types, each 3x3 spatial bins",
        "normalization": "tanh(pooled activity - fixed 0.1-second gray-warmup activity)",
        "initialization_seconds": round(extractor.init_seconds, 3),
        "stream_seconds": round(duration, 3), "frames_per_second": round(len(frames) / duration, 3),
        "mean_frame_seconds": round(float(np.mean(durations)), 4),
        "p95_frame_seconds": round(float(np.percentile(durations, 95)), 4),
        "realtime_factor_at_10hz": round((len(frames) / 10) / duration, 3),
        "estimated_extraction_seconds_1000_frames": round(duration / len(frames) * 1000, 1),
        "estimated_extraction_seconds_3000_frames": round(duration / len(frames) * 3000, 1),
        "estimates_exclude": "camera rendering, environment steps, dataset IO, reset overhead and head training",
        "feature_min": float(features.min()), "feature_max": float(features.max()),
        "max_temporal_feature_range": float(np.ptp(features, axis=0).max()),
        "max_motion_difference_from_static_control": motion_difference,
        "reset_replay_max_error": reset_error,
        "stream_vs_uninterrupted_sequence_max_error": stream_equivalence_error,
        "final_neural_activity_finite": bool(extractor.torch.isfinite(final_state).all()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Streaming feature benchmark failed")


if __name__ == "__main__":
    main()
