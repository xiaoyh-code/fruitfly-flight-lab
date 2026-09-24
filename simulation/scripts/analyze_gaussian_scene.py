#!/usr/bin/env python3
"""Inspect a 3DGS PLY without treating Gaussian centres as collision surfaces.

Produces plots and descriptive statistics only. Alpha is sigmoid(opacity), and
the standard 3DGS linear axis scales are exp(scale_0..2). Coordinates are used
as stored in the file; this script does not establish metric scale or geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_float_ply(path: Path):
    properties, vertex_count = [], None
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("Not a PLY file")
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("Incomplete PLY header")
            line = raw.decode("ascii").strip()
            if line.startswith("format ") and line != "format binary_little_endian 1.0":
                raise ValueError("Only binary little-endian PLY is supported")
            if line.startswith("element "):
                if not line.startswith("element vertex "):
                    raise ValueError("Only a vertex-only PLY is supported")
                vertex_count = int(line.split()[-1])
            if line.startswith("property "):
                kind, name = line.split()[1:]
                if kind != "float":
                    raise ValueError("Only float32 vertex properties are supported")
                properties.append(name)
            if line == "end_header":
                break
        if vertex_count is None:
            raise ValueError("Missing vertex count")
        expected = vertex_count * len(properties) * 4
        body = stream.read()
        if len(body) != expected:
            raise ValueError(f"Incomplete/extra data: expected {expected} body bytes, got {len(body)}")
    data = np.frombuffer(body, dtype="<f4").reshape(vertex_count, len(properties))
    return {name: data[:, i] for i, name in enumerate(properties)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=ROOT / "models/gaussian/robot_hall.ply")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "outputs/realistic/scene-analysis")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--altitude", type=float, default=1.2)
    parser.add_argument("--half-band", type=float, default=0.25)
    parser.add_argument("--cell", type=float, default=0.15)
    args = parser.parse_args()
    fields = read_float_ply(args.scene)
    xyz = np.column_stack([fields[c] for c in "xyz"])
    alpha = 1.0 / (1.0 + np.exp(-np.clip(fields["opacity"], -40, 40)))
    scales = np.exp(np.clip(np.column_stack([fields[f"scale_{i}"] for i in range(3)]), -40, 20))
    color = np.clip(0.5 + 0.28209479177387814 * np.column_stack([fields[f"f_dc_{i}"] for i in range(3)]), 0, 1)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(scales).all(axis=1)
    selected = finite & (alpha >= args.alpha)
    points = xyz[selected]
    # Trim obvious isolated centres for legible plots, without discarding them
    # from descriptive statistics or implying that outliers cannot be occupied.
    bounds = np.quantile(points, [0.01, 0.99], axis=0)
    bounds[0] = np.floor(bounds[0])
    bounds[1] = np.ceil(bounds[1])
    xyz_bins = [np.arange(lo, hi + args.cell, args.cell) for lo, hi in zip(*bounds)]
    band_mask = selected & (np.abs(xyz[:, 2] - args.altitude) <= args.half_band)
    band = xyz[band_mask]

    fig = plt.figure(figsize=(18, 14), facecolor="#f7f7f4")
    grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 0.8))
    axes = [fig.add_subplot(grid[i, j]) for i in range(3) for j in range(2)]

    def density(ax, values, dimensions, title):
        i, j = dimensions
        counts, xb, yb = np.histogram2d(values[:, i], values[:, j], bins=(xyz_bins[i], xyz_bins[j]))
        drawn = ax.pcolormesh(xb, yb, np.ma.masked_less(counts.T, 1), cmap="magma", norm=LogNorm(vmin=1, vmax=max(2, counts.max())))
        fig.colorbar(drawn, ax=ax, shrink=0.7, label="Gaussian centre count / cell")
        ax.set(xlabel=f"{'xyz'[i]} (stored coordinates)", ylabel=f"{'xyz'[j]} (stored coordinates)", title=title)
        ax.set_aspect("equal")
        ax.grid(alpha=0.16)
        return counts

    density(axes[0], points, (0, 1), f"All-height centres; alpha ≥ {args.alpha}")
    band_counts = density(axes[1], band, (0, 1), f"Flight-height band z={args.altitude-args.half_band:.2f}…{args.altitude+args.half_band:.2f}")
    density(axes[2], points, (0, 2), "x–z elevation; centre density")
    density(axes[3], points, (1, 2), "y–z elevation; centre density")
    axes[4].hist(points[:, 2], bins=np.linspace(bounds[0, 2], bounds[1, 2], 100), color="#236b68")
    axes[4].axvspan(args.altitude-args.half_band, args.altitude+args.half_band, color="#d97a3b", alpha=0.3)
    axes[4].set(xlabel="z (stored coordinates)", ylabel="Gaussian centre count", title="Height distribution (centres, not surface area)")
    axes[5].scatter(band[:, 0], band[:, 1], s=3, c=color[band_mask], alpha=0.8, rasterized=True)
    axes[5].set(xlim=bounds[:, 0], ylim=bounds[:, 1], xlabel="x (stored coordinates)", ylabel="y (stored coordinates)", title="Flight-height centres colored by SH DC only")
    axes[5].set_aspect("equal")
    axes[5].set_facecolor("#303339")
    axes[5].grid(alpha=0.2)
    fig.suptitle(f"{args.scene.name}: descriptive 3DGS layout\nNot a collision map or measured depth; blank cells are not proven free space", fontsize=17)
    fig.tight_layout(rect=(0, 0.01, 1, 0.94))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=160)
    plt.close(fig)

    # Find dense cells as inspection candidates, not autonomous obstacle labels.
    order = np.argsort(band_counts.ravel())[::-1]
    candidates = []
    for flat in order:
        ix, iy = np.unravel_index(flat, band_counts.shape)
        count = int(band_counts[ix, iy])
        if count < 3 or len(candidates) >= 20:
            break
        center = np.array([(xyz_bins[0][ix] + xyz_bins[0][ix+1])/2, (xyz_bins[1][iy] + xyz_bins[1][iy+1])/2])
        if any(np.linalg.norm(center - np.asarray(c["xy"])) < 0.6 for c in candidates):
            continue
        candidates.append({"xy": center.tolist(), "count_in_cell": count})
    report = {
        "source": str(args.scene.resolve()), "sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        "vertices": len(xyz), "high_opacity_count": int(selected.sum()),
        "opacity_threshold": args.alpha,
        "alpha_conversion": "sigmoid(stored opacity)", "axis_scale_conversion": "exp(stored scale_i)",
        "coordinate_handling": "Stored coordinates used without registration. Dataset documentation describes z-up metres; this analysis does not independently verify that calibration.",
        "high_opacity_xyz_quantiles": {str(q): np.quantile(points, q, axis=0).tolist() for q in [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]},
        "linear_axis_scale_quantiles": {str(q): float(np.quantile(scales[selected], q)) for q in [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]},
        "plot_bounds_xyz": bounds.tolist(), "cell_width": args.cell,
        "flight_height": args.altitude, "height_band": [args.altitude - args.half_band, args.altitude + args.half_band],
        "height_band_high_opacity_count": len(band),
        "dense_cells_for_manual_inspection_only": candidates,
        "limitations": [
            "Gaussian centres and Gaussian support are not measured surfaces, solid occupancy, clearance or depth ground truth.",
            "Low density and empty cells can be missing reconstruction, not navigable free space.",
            "High opacity of an individual splat does not establish a solid obstacle.",
            "SH DC point colors are only a layout aid; use the Gaussian renderer for actual views.",
            "Collision proxies must be manually checked against rendered views and labeled approximate.",
            "Plot limits omit extreme one-percent tails; statistics retain them."
        ]
    }
    # This candidate is specific to the inspected robot_hall file. It is a
    # visible reddish compact cluster, not an identified or measured object.
    if args.scene.name == "robot_hall.ply":
        center = np.array([15.15, -2.95])
        start = np.array([13.55, -3.15])
        goal = np.array([16.75, -3.15])
        radius = 0.45
        region = selected & (xyz[:, 0] > 14.5) & (xyz[:, 0] < 15.7) & (xyz[:, 1] > -3.6) & (xyz[:, 1] < -2.0)
        candidate_band = xyz[region & band_mask]
        report["candidate_for_rendered_inspection"] = {
            "description": "Isolated reddish Gaussian cluster; object identity unknown until renderer inspection.",
            "status": "Candidate only; proxy unvalidated against rendered surfaces. Not certified free space.",
            "flight_band_count": len(candidate_band),
            "flight_band_xyz_min_max": [candidate_band.min(axis=0).tolist(), candidate_band.max(axis=0).tolist()],
            "flight_band_xyz_5_95_percentiles": np.quantile(candidate_band, [0.05, 0.95], axis=0).tolist(),
            "suggested_scene_start_xyz": [*start.tolist(), args.altitude],
            "suggested_scene_goal_xyz": [*goal.tolist(), args.altitude],
            "scene_translation_from_local_mujoco_xyz": [15.15, -3.15, 0.0],
            "rough_cylinder_proxy": {
                "scene_center_xy": center.tolist(), "local_center_xy": [0.0, 0.20],
                "radius": radius, "bottom_z": 0.0, "top_z": 1.55,
                "role": "Approximate conservative candidate for fixed-z=1.2 tests, pending rendered inspection; no depth truth claim."
            },
            "workflow": "Inspect rendered views from both route sides first. If shape/scale disagree, revise or reject candidate. Keep policy inputs RGB-derived; proxy only supplies labels/collision evaluation."
        }
        fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor="#f7f7f4")
        local_band = band_mask & (xyz[:, 0] > 12.5) & (xyz[:, 0] < 18.0) & (xyz[:, 1] > -5.1) & (xyz[:, 1] < -0.5)
        axes[0, 0].scatter(xyz[local_band, 0], xyz[local_band, 1], c=color[local_band], s=13)
        axes[0, 0].plot([start[0], goal[0]], [start[1], goal[1]], color="#23b6cf", lw=2, marker="o", label="Proposed start / goal; direct line crosses proxy")
        axes[0, 0].add_patch(plt.Circle(center, radius, fill=False, color="#ed8535", lw=2, label="Unvalidated approximate proxy"))
        axes[0, 0].set(xlim=(12.5, 18.0), ylim=(-5.1, -0.5), xlabel="x", ylabel="y", title="Candidate top-down; flight-height centres")
        axes[0, 0].legend(loc="upper left", fontsize=8)
        axes[0, 0].set_aspect("equal")
        for ax, dims, title in [(axes[0, 1], (0, 1), "Cluster XY at flight height"), (axes[1, 0], (0, 2), "Cluster XZ; z<2 only"), (axes[1, 1], (1, 2), "Cluster YZ; z<2 only")]:
            mask = region & (band_mask if dims == (0, 1) else ((xyz[:, 2] > -0.5) & (xyz[:, 2] < 2.0)))
            ax.scatter(xyz[mask, dims[0]], xyz[mask, dims[1]], c=color[mask], s=15)
            ax.set(xlabel="xyz"[dims[0]], ylabel="xyz"[dims[1]], title=title)
            ax.set_aspect("equal")
            if dims[1] == 2:
                ax.axhspan(args.altitude-args.half_band, args.altitude+args.half_band, color="#ed8535", alpha=0.13)
        for ax in axes.ravel():
            ax.set_facecolor("#34373d")
            ax.grid(alpha=0.2)
        fig.suptitle("Candidate for actual renderer inspection\nGaussian centre plots only — object identity, surface bounds and clearance unverified", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(args.output_prefix.with_name(args.output_prefix.name + "-candidate").with_suffix(".png"), dpi=160)
        plt.close(fig)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding='utf-8')
    print(json.dumps({"plot": str(args.output_prefix.with_suffix(".png")), "analysis": str(args.output_prefix.with_suffix(".json")), "vertices": len(xyz), "height_band_points": len(band)}))


if __name__ == "__main__":
    main()
