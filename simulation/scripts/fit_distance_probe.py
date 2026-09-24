#!/usr/bin/env python3
"""Decode approximate scene-proxy distance from frozen Flyvis features.

This trains diagnostic ridge readouts, never the existing navigation policy or
Flyvis. All model fitting and normalization use train trajectories; validation
selects the ridge coefficient, and test trajectories are evaluated once. The
labels are approximate collision-proxy clearances within one captured scene,
not measured physical depth. No goal, position, time or truth distance enters
the model input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MIN_DISTANCE = 0.05
MAX_DISTANCE = 6.0
NEAR_THRESHOLD = 0.8
DEFAULT_ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def validate_dataset(data):
    """Reject overlapping trajectories before fitting or reading split labels."""
    required = {"features", "velocity", "distance", "trajectory", "split"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Missing arrays: {sorted(missing)}")
    features = np.asarray(data["features"], dtype=np.float64)
    velocity = np.asarray(data["velocity"], dtype=np.float64)
    distance = np.asarray(data["distance"], dtype=np.float64)
    trajectory = np.asarray(data["trajectory"])
    split = np.asarray(data["split"]).astype(str)
    n = len(distance)
    if features.shape != (n, 72) or velocity.shape != (n, 2):
        raise ValueError("Expected features Nx72 and velocity Nx2")
    if distance.shape != (n,) or trajectory.shape != (n,) or split.shape != (n,):
        raise ValueError("distance, trajectory and split must be matching 1D arrays")
    if not np.issubdtype(trajectory.dtype, np.integer):
        raise ValueError("trajectory IDs must be integers")
    if not all(np.isfinite(x).all() for x in (features, velocity, distance)):
        raise ValueError("Features, velocity and distance must be finite")
    if np.any(distance < 0):
        raise ValueError("Distance labels must be nonnegative proxy surface clearance")
    if set(split) != {"train", "val", "test"}:
        raise ValueError("Require nonempty train, val and test splits only")
    for trajectory_id in np.unique(trajectory):
        if len(np.unique(split[trajectory == trajectory_id])) != 1:
            raise ValueError(f"Trajectory {trajectory_id} occurs in multiple splits")
    return {"features": features, "velocity": velocity, "distance": distance,
            "trajectory": trajectory.astype(np.int64), "split": split}


def fit_ridge(features, distance, alpha):
    """Fit on supplied training rows only, including train-only standardization."""
    x = np.asarray(features, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    normalized = (x - mean) / scale
    inverse = 1.0 / np.clip(distance, MIN_DISTANCE, MAX_DISTANCE)
    intercept = float(np.mean(inverse))
    matrix = normalized.T @ normalized + float(alpha) * np.eye(x.shape[1])
    coefficient = np.linalg.solve(matrix, normalized.T @ (inverse - intercept))
    return {"mean": mean, "scale": scale, "coefficient": coefficient,
            "intercept": intercept, "alpha": float(alpha)}


def predict_distance(model, features):
    x = (np.asarray(features, dtype=np.float64) - model["mean"]) / model["scale"]
    inverse = x @ model["coefficient"] + model["intercept"]
    # Clip the inverse before reciprocating: a negative unconstrained prediction
    # should map to the far limit, not falsely imply a very close obstacle.
    return 1.0 / np.clip(inverse, 1.0 / MAX_DISTANCE, 1.0 / MIN_DISTANCE)


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    error = prediction - truth
    near_true = truth <= NEAR_THRESHOLD
    near_pred = prediction <= NEAR_THRESHOLD
    tp = int(np.sum(near_true & near_pred))
    fn = int(np.sum(near_true & ~near_pred))
    fp = int(np.sum(~near_true & near_pred))
    tn = int(np.sum(~near_true & ~near_pred))
    return {
        "samples": len(truth), "mae_m": float(np.mean(np.abs(error))),
        "median_absolute_error_m": float(np.median(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "near_threshold_m": NEAR_THRESHOLD,
        "near_recall": tp / (tp + fn) if tp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "confusion": {"true_positive": tp, "false_negative": fn,
                      "false_positive": fp, "true_negative": tn},
    }


def select_ridge(x_train, y_train, x_val, y_val, alphas=DEFAULT_ALPHAS):
    """The selection API deliberately has no test arguments."""
    candidates = []
    chosen = None
    best = float("inf")
    for alpha in alphas:
        model = fit_ridge(x_train, y_train, alpha)
        score = metrics(y_val, predict_distance(model, x_val))["rmse_m"]
        candidates.append({"alpha": float(alpha), "validation_rmse_m": score})
        if score < best:
            best, chosen = score, model
    return chosen, candidates


def analyze(data, alphas=DEFAULT_ALPHAS):
    data = validate_dataset(data)
    split, truth = data["split"], data["distance"]
    train, val, test = (split == name for name in ("train", "val", "test"))
    inputs = {
        "flyvis_only": data["features"],
        "flyvis_velocity": np.column_stack((data["features"], data["velocity"])),
        "velocity_only": data["velocity"],
    }
    models, results, predictions = {}, {}, {}
    for name, x in inputs.items():
        model, candidates = select_ridge(x[train], truth[train], x[val], truth[val], alphas)
        models[name] = model
        prediction = predict_distance(model, x)
        predictions[name] = prediction
        results[name] = {
            "input_dimension": x.shape[1], "selected_alpha": model["alpha"],
            "validation_candidates": candidates,
            "validation": metrics(truth[val], prediction[val]),
            "test": metrics(truth[test], prediction[test]),
        }
    constant = float(np.clip(truth[train].mean(), MIN_DISTANCE, MAX_DISTANCE))
    predictions["mean_distance"] = np.full(len(truth), constant)
    results["mean_distance"] = {
        "training_mean_distance_m": float(truth[train].mean()),
        "clipped_constant_prediction_m": constant,
        "validation": metrics(truth[val], predictions["mean_distance"][val]),
        "test": metrics(truth[test], predictions["mean_distance"][test]),
    }
    for name in results:
        per_trajectory = []
        for trajectory_id in np.unique(data["trajectory"][test]):
            rows = test & (data["trajectory"] == trajectory_id)
            per_trajectory.append({"trajectory": int(trajectory_id),
                                   **metrics(truth[rows], predictions[name][rows])})
        results[name]["test_by_trajectory"] = per_trajectory
        results[name]["test_macro_trajectory_mae_m"] = float(np.mean([row["mae_m"] for row in per_trajectory]))
    summary = {name: {"samples": int(np.sum(split == name)),
                      "trajectory_ids": np.unique(data["trajectory"][split == name]).tolist(),
                      "distance_min_max_m": [float(truth[split == name].min()), float(truth[split == name].max())]}
               for name in ("train", "val", "test")}
    report = {
        "experiment": "Frozen Flyvis diagnostic inverse-distance ridge readout",
        "label": "Approximate manually fitted cylinder-proxy surface clearance; not measured physical depth",
        "scope": "Held-out whole trajectories within one 3DGS scene, not cross-scene generalization",
        "flyvis_weights_trained": False, "navigation_policy_changed": False,
        "ground_truth_in_model_input": False, "normalization": "Training-row mean and population standard deviation only; constant columns use scale 1",
        "selection": "Train-only fitting; ridge alpha selected by validation distance RMSE; no train+val refit",
        "ridge_objective": "Squared inverse-distance residual sum + alpha * coefficient squared norm; intercept unpenalized",
        "distance_prediction_range_m": [MIN_DISTANCE, MAX_DISTANCE],
        "target_transform": "1 / clip(proxy distance, 0.05, 6); metrics retain original nonnegative labels",
        "splits": summary, "results": results,
        "interpretation": "Successful readout supports decodable proximity information in frozen features for this scene. It does not establish that the existing M4 head estimates distance or that avoidance succeeds.",
        "limitations": [
            "Clearance labels inherit manual proxy registration, scale and shape errors.",
            "Feature-plus-velocity predictions have a motion/scale cue and are not single-image monocular metric depth.",
            "Features-only readout still sees temporal Flyvis state; it is not a single-frame estimator.",
            "Repeated frames within each trajectory are correlated; trajectory-level errors are included.",
            "A single fixed obstacle may allow scene-specific appearance/distance correlations.",
            "Whole-trajectory ID disjointness is enforced; the data collector must also ensure held-out paths rather than duplicate paths under new IDs."
        ],
    }
    return report, models, predictions, data


def save_plot(path, report, predictions, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    test = data["split"] == "test"
    truth = data["distance"][test]
    ids = data["trajectory"][test]
    colors = np.unique(ids, return_inverse=True)[1]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), facecolor="#f7f7f4")
    for ax, name in zip(axes.ravel(), ("flyvis_only", "flyvis_velocity", "velocity_only", "mean_distance")):
        prediction = predictions[name][test]
        ax.scatter(truth, prediction, c=colors, cmap="tab10", s=12, alpha=0.55)
        limit = max(1.0, min(MAX_DISTANCE, max(float(truth.max()), float(prediction.max())) + 0.15))
        ax.plot([0, limit], [0, limit], "--", color="#444444", lw=1)
        ax.axhline(NEAR_THRESHOLD, color="#b77335", alpha=0.4)
        ax.axvline(NEAR_THRESHOLD, color="#b77335", alpha=0.4)
        result = report["results"][name]["test"]
        ax.set(xlabel="Approximate proxy clearance (m)", ylabel="Readout prediction (m)",
               title=f"{name.replace('_', ' ')}\nMAE {result['mae_m']:.3f} m; RMSE {result['rmse_m']:.3f} m",
               xlim=(0, max(limit, float(truth.max()) + 0.1)), ylim=(0, limit))
        ax.grid(alpha=0.15)
    fig.suptitle("Held-out trajectory diagnostic · one 3DGS scene\nProxy labels, not measured depth; colors distinguish test trajectories", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/realistic/distance-sequences.npz")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "outputs/realistic/distance-probe")
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        report, models, predictions, data = analyze(archive)
    report["source_dataset"] = str(args.input.resolve())
    report["source_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, model in models.items():
        arrays.update({f"{name}__{key}": value for key, value in model.items()})
    arrays.update({f"prediction__{name}": prediction for name, prediction in predictions.items()})
    arrays.update({"distance_truth_for_evaluation_only": data["distance"],
                   "trajectory": data["trajectory"], "split": data["split"],
                   "constant_prediction": report["results"]["mean_distance"]["clipped_constant_prediction_m"]})
    np.savez_compressed(args.output_prefix.with_suffix(".npz"), **arrays)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding='utf-8')
    save_plot(args.output_prefix.with_suffix(".png"), report, predictions, data)
    print(json.dumps({"report": str(args.output_prefix.with_suffix(".json")),
                      "test_metrics": {name: result["test"] for name, result in report["results"].items()}}, indent=2))


if __name__ == "__main__":
    main()
