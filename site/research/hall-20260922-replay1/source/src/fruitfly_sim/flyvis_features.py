"""Streaming, frozen Flyvis features for a separate learned decision head.

Camera frames must be RGB uint8. They are center-square cropped, converted to
luminance and sampled by the official BoxEye onto 721 retinal hexals. The
pretrained visual network's FULL neural state survives each camera update;
call reset() only between episodes. This is not whole-brain learning.

Output order: T4a,b,c,d,T5a,b,c,d, each with 3 spatial rows × 3 columns.
Fixed normalization is tanh(mean activity - gray-warmup mean activity).
"""
from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class FlyvisFeatureExtractor:
    feature_dim = 72
    cell_types = tuple(f"T{layer}{direction}" for layer in (4, 5) for direction in "abcd")

    def __init__(self, model_dir=None, *, cpu_threads=2, neural_dt=0.02, camera_dt=0.1):
        started = time.perf_counter()
        if not 0 < neural_dt <= 0.02:
            raise ValueError("neural_dt must be > 0 and <= 0.02 seconds")
        substeps = round(camera_dt / neural_dt)
        if substeps < 1 or not np.isclose(substeps * neural_dt, camera_dt):
            raise ValueError("camera_dt must be an integer multiple of neural_dt")
        if cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        os.environ.setdefault("FLYVIS_ROOT_DIR", str(ROOT / "data" / "flyvis"))
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
        os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache" / "numba"))
        import torch
        torch.set_num_threads(cpu_threads)
        import flyvis
        from flyvis.datasets.rendering import BoxEye
        from flyvis.utils.hex_utils import hex_to_pixel

        self.torch = torch
        # Upstream Flyvis uses a global device during model construction.
        # CPU is the validated backend for this local macOS integration.
        flyvis.device = torch.device("cpu")
        torch.set_default_device("cpu")
        self.neural_dt = float(neural_dt)
        self.camera_dt = float(camera_dt)
        self.substeps = substeps
        self.model_dir = Path(model_dir or (ROOT / "data/flyvis/results/flow/0000/000"))
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Official pretrained Flyvis model missing: {self.model_dir}")
        view = flyvis.NetworkView(self.model_dir)
        self.network = view.init_network()
        self.network.eval()
        self.network.requires_grad_(False)
        self.eye = BoxEye(extent=15, kernel_size=13)
        self.side = int(self.eye.min_frame_size.max())
        if self.eye.hexals != 721:
            raise RuntimeError("Expected official 721-element retina")
        all_indices, all_bins = [], []
        nodes = self.network.connectome.nodes
        for type_index, name in enumerate(self.cell_types):
            indices = np.asarray(nodes.layer_index[name][:], dtype=np.int64)
            x, y = hex_to_pixel(nodes.u[:][indices], nodes.v[:][indices])
            cols = np.clip(np.floor(3 * (x - x.min()) / (np.ptp(x) + 1e-8)), 0, 2).astype(int)
            rows = np.clip(np.floor(3 * (y - y.min()) / (np.ptp(y) + 1e-8)), 0, 2).astype(int)
            all_indices.extend(indices.tolist())
            all_bins.extend((type_index * 9 + rows * 3 + cols).tolist())
        self._indices = torch.tensor(all_indices, dtype=torch.long)
        self._bins = torch.tensor(all_bins, dtype=torch.long)
        self._counts = torch.bincount(self._bins, minlength=72).to(torch.float32)
        if (self._counts == 0).any():
            raise RuntimeError("Spatial pooling produced an empty bin")
        self.state = None
        self.last_raw_features = np.zeros(72, dtype=np.float32)
        self.frames_processed = 0
        self.reset()
        self.init_seconds = time.perf_counter() - started

    def _pool(self, state):
        activity = state.nodes.activity[0]
        sums = self.torch.zeros(72, dtype=activity.dtype)
        sums.scatter_add_(0, self._bins, activity.index_select(0, self._indices))
        return sums / self._counts

    def reset(self):
        """Start an episode with 0.1 simulated seconds of uniform gray input."""
        with self.torch.no_grad():
            self.state = self.network.steady_state(0.1, self.neural_dt, batch_size=1, value=0.5)
            self._gray_baseline = self._pool(self.state).clone()
        self.frames_processed = 0
        self.last_raw_features = self._gray_baseline.cpu().numpy().copy()
        return np.zeros(72, dtype=np.float32)

    def retinal_input(self, rgb):
        """Map one RGB uint8 image to shape (1, 1, 1, 721) on CPU."""
        import cv2
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected an RGB uint8 array of shape (height, width, 3)")
        if min(rgb.shape[:2]) < 2:
            raise ValueError("Camera image is too small")
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        crop = min(height, width)
        y, x = (height - crop) // 2, (width - crop) // 2
        gray = cv2.resize(gray[y:y + crop, x:x + crop], (self.side, self.side), interpolation=cv2.INTER_AREA)
        sequence = self.torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None]
        with self.torch.no_grad():
            return self.eye(sequence)

    def transform(self, rgb):
        """Advance neural time by camera_dt, retaining state, and return 72 features."""
        movie = self.retinal_input(rgb).expand(1, self.substeps, 1, 721)
        with self.torch.no_grad():
            self.network.stimulus.zero(1, self.substeps)
            self.network.stimulus.add_input(movie)
            states = self.network(self.network.stimulus(), self.neural_dt,
                                  state=self.state, as_states=True)
            self.state = states[-1]
            raw = self._pool(self.state)
            features = self.torch.tanh(raw - self._gray_baseline).cpu().numpy().astype(np.float32)
            self.last_raw_features = raw.cpu().numpy().copy()
        if not np.all(np.isfinite(features)):
            raise RuntimeError("Flyvis produced non-finite features")
        self.frames_processed += 1
        return features

    __call__ = transform
