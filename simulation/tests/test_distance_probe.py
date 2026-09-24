"""Leakage protections for the frozen-feature distance diagnostic."""
from pathlib import Path
import importlib.util

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location("fit_distance_probe", Path(__file__).resolve().parents[1] / "scripts/fit_distance_probe.py")
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def sample_data():
    rng = np.random.default_rng(17)
    features = rng.normal(size=(90, 72))
    distance = 1 / (2.5 + 0.15 * features[:, 0])
    return {"features": features, "velocity": rng.normal(size=(90, 2)),
            "distance": distance, "trajectory": np.repeat(np.arange(9), 10),
            "split": np.repeat(["train", "val", "test"], [50, 20, 20])}


def test_trajectory_overlap_is_rejected():
    data = sample_data()
    data["trajectory"][50] = data["trajectory"][0]
    with pytest.raises(ValueError, match="multiple splits"):
        PROBE.validate_dataset(data)


def test_normalization_uses_training_rows_and_is_not_updated_at_prediction():
    train = np.array([[0.0, 7.0], [2.0, 7.0], [4.0, 7.0]])
    model = PROBE.fit_ridge(train, np.array([0.3, 0.5, 0.8]), 0.1)
    np.testing.assert_allclose(model["mean"], [2.0, 7.0])
    np.testing.assert_allclose(model["scale"], [np.sqrt(8/3), 1.0])
    snapshot = {key: np.array(value, copy=True) for key, value in model.items()}
    prediction = PROBE.predict_distance(model, np.array([[1000.0, -800.0]]))
    assert PROBE.MIN_DISTANCE <= prediction[0] <= PROBE.MAX_DISTANCE
    for key, original in snapshot.items():
        np.testing.assert_array_equal(model[key], original)


def test_test_labels_cannot_change_fit_or_hyperparameter_choice():
    first = sample_data()
    changed = {key: value.copy() for key, value in first.items()}
    changed["distance"][changed["split"] == "test"] = 5.5
    report_a, models_a, predictions_a, _ = PROBE.analyze(first, alphas=(0.1, 10.0))
    report_b, models_b, predictions_b, _ = PROBE.analyze(changed, alphas=(0.1, 10.0))
    for name in models_a:
        for key in models_a[name]:
            np.testing.assert_array_equal(models_a[name][key], models_b[name][key])
        np.testing.assert_array_equal(predictions_a[name], predictions_b[name])
        assert report_a["results"][name]["selected_alpha"] == report_b["results"][name]["selected_alpha"]
    assert report_a["results"]["flyvis_only"]["test"]["mae_m"] != report_b["results"]["flyvis_only"]["test"]["mae_m"]
