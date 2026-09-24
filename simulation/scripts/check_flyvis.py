#!/usr/bin/env python3
"""Run one official pretrained Flyvis visual circuit on a short CPU stimulus.

This verifies stimulus-dependent T4/T5 neural activity, not optical-flow accuracy
or a drone flight controller. No model or dataset is downloaded by this script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="Read a short clip instead of a synthetic moving bar")
    parser.add_argument(
        "--video-layout", choices=("single", "fpv-left"), default="single",
        help="Use the entire video frame, or only its left FPV half before retinal preprocessing",
    )
    parser.add_argument("--frames", type=int, default=20, help="Input frames, from 2 to 60 (default: 20)")
    parser.add_argument("--output-prefix", help="Output filename prefix, without extension")
    args = parser.parse_args()
    if not 2 <= args.frames <= 60:
        parser.error("--frames must be between 2 and 60")

    os.environ["FLYVIS_ROOT_DIR"] = str(ROOT / "data" / "flyvis")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    from fruitfly_sim.flyvis_compat import configure_flyvis_cache
    configure_flyvis_cache()
    import flyvis
    from flyvis.network import initialization
    from flyvis.datasets.rendering import BoxEye
    from flyvis.utils.activity_utils import LayerActivity

    # Flyvis chooses CPU automatically on this Mac. Keep this probe reproducible
    # on other machines too; MPS is not an upstream-validated Flyvis backend.
    flyvis.device = torch.device("cpu")
    initialization.device = flyvis.device
    torch.set_default_device("cpu")
    torch.manual_seed(0)
    started = time.perf_counter()
    model_dir = flyvis.results_dir / "flow" / "0000" / "000"
    if not model_dir.is_dir():
        raise SystemExit(
            "Pretrained model missing. Download the checksum-verified official models first:\n"
            "python scripts/download_flyvis_models.py"
        )

    eye = BoxEye(extent=15, kernel_size=13)
    centers = eye.receptor_centers.cpu().numpy()
    dt = 0.01
    source = "synthetic horizontal moving bright bar"
    input_info: dict = {"source_frames": args.frames, "source_fps": 100.0}
    if args.video:
        import cv2

        video = args.video.expanduser().resolve()
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise SystemExit(f"Cannot open video: {video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            capture.release()
            raise SystemExit("Video has no usable frame rate; encode it with an explicit FPS first")
        side = int(eye.min_frame_size.max())
        frames = []
        try:
            for _ in range(args.frames):
                ok, frame = capture.read()
                if not ok:
                    break
                if args.video_layout == "fpv-left":
                    frame = frame[:, :frame.shape[1] // 2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                height, width = gray.shape
                crop = min(height, width)
                y, x = (height - crop) // 2, (width - crop) // 2
                frames.append(cv2.resize(gray[y:y + crop, x:x + crop], (side, side)))
        finally:
            capture.release()
        if len(frames) < 2:
            raise SystemExit("Video must contain at least two readable frames")
        # Integrate the neural dynamics at 100 Hz, preserving video timing by
        # holding frames between camera updates. Cap work to two simulated seconds.
        duration = min(len(frames) / fps, 2.0)
        indices = np.minimum((np.arange(0, duration, dt) * fps).astype(int), len(frames) - 1)
        cartesian = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
        with torch.no_grad():
            movie = eye(cartesian[indices][None])
        source = str(video)
        input_info = {
            "source_frames": len(frames), "source_fps": fps,
            "video_layout": args.video_layout,
            "source_crop": "left half only (FPV)" if args.video_layout == "fpv-left" else "full frame",
            "adapter": "grayscale, center square crop, resize, official BoxEye mean filter",
            "temporal_resampling": "hold source frames at 100 Hz; maximum duration 2 seconds",
        }
    else:
        xs = centers[:, 1]
        bar_locations = np.linspace(xs.min() * 0.65, xs.max() * 0.65, args.frames)
        brightness = np.where(np.abs(xs[None] - bar_locations[:, None]) < 26, 0.9, 0.1)
        movie = torch.from_numpy(brightness.astype(np.float32))[None, :, None, :]

    assert movie.shape[-1] == 721
    if float(movie[:, 1:].sub(movie[:, :-1]).abs().max()) == 0:
        raise SystemExit("Stimulus is static; choose a clip with movement for this response check")

    view = flyvis.NetworkView(model_dir)
    network = view.init_network()
    network.eval()
    network.requires_grad_(False)
    # The second sequence holds the first image fixed. Differences from it help
    # distinguish motion-driven responses from the shared startup transient.
    stationary_movie = movie[:, :1].expand_as(movie)
    paired_movie = torch.cat([movie, stationary_movie], dim=0)
    with torch.no_grad():
        state = network.steady_state(0.1, dt, batch_size=2, value=0.5)
        responses = network.simulate(paired_movie, dt, initial_state=state).cpu()

    if not torch.isfinite(responses).all():
        raise RuntimeError("Pretrained network returned non-finite activity")
    activity = LayerActivity(responses, network.connectome, keepref=True)
    cell_types = [f"T{layer}{direction}" for layer in (4, 5) for direction in "abcd"]
    metrics = {}
    traces = {}
    for cell_type in cell_types:
        values = activity[cell_type].detach().cpu().numpy()
        moving, held = values[0], values[1]
        temporal_span = float(np.max(np.ptp(moving, axis=0)))
        stimulus_difference = float(np.max(np.abs(moving - held)))
        metrics[cell_type] = {
            "finite": bool(np.isfinite(values).all()),
            "max_temporal_activity_range": temporal_span,
            "max_difference_from_static_control": stimulus_difference,
            "passed": temporal_span > 1e-7 and stimulus_difference > 1e-7,
        }
        # Plot the neuron most affected by motion, with its matching static trace.
        neuron = int(np.argmax(np.max(np.abs(moving - held), axis=0)))
        traces[cell_type] = (moving[:, neuron], held[:, neuron])

    passed = all(item["passed"] for item in metrics.values())
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    prefix = args.output_prefix or ("flyvis-video-check" if args.video else "flyvis-check")
    if Path(prefix).name != prefix:
        parser.error("--output-prefix must be a filename, not a path")
    png_path = output_dir / f"{prefix}.png"
    json_path = output_dir / f"{prefix}.json"
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, index, title in zip(axes[0], (0, movie.shape[1] - 1), ("First retinal frame", "Last retinal frame")):
        ax.scatter(centers[:, 1], -centers[:, 0], c=movie[0, index, 0].numpy(), cmap="gray", vmin=0, vmax=1, s=12)
        ax.set(title=title, aspect="equal")
        ax.axis("off")
    times = np.arange(responses.shape[1]) * dt
    for ax, layer in zip(axes[1], (4, 5)):
        for direction in "abcd":
            name = f"T{layer}{direction}"
            moving, held = traces[name]
            ax.plot(times, moving - held, label=name)
        ax.set(title=f"T{layer}: activity difference from static control", xlabel="Time (s)", ylabel="Model activity (a.u.)")
        ax.legend(ncol=4, fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Pretrained Flyvis visual-circuit check | CPU | one model", fontsize=13)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    result = {
        "passed": passed,
        "scope": "Stimulus-dependent pretrained T4/T5 neural activity; not optical-flow accuracy or autonomous navigation",
        "model": "flow/0000/000", "model_directory": str(model_dir),
        "checkpoint": str(view.get_checkpoint("best")),
        "device": "cpu", "cpu_threads": torch.get_num_threads(),
        "flyvis_version": flyvis.__version__, "torch_version": torch.__version__,
        "source": source, "input": input_info,
        "input_shape": list(movie.shape), "activity_shape_with_static_control": list(responses.shape),
        "dt_seconds": dt, "warmup_seconds": 0.1,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "cells": metrics, "plot": str(png_path),
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(f"Report: {json_path}")
    if not passed:
        raise SystemExit("One or more T4/T5 checks failed; inspect the report and stimulus")


if __name__ == "__main__":
    main()
