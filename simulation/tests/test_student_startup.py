"""Regression checks for a genuinely new student's empty workspace."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def test_empty_training_directory_starts_pending_and_preserves_updates(tmp_path, monkeypatch):
    import train_curriculum as training
    monkeypatch.setattr(training, 'OUT', tmp_path / 'outputs/training')
    monkeypatch.setattr(training, 'CKPT', tmp_path / 'checkpoints/training')
    live = training.Live()
    assert live.history == []
    assert [row['status'] for row in live.status['milestones']] == ['pending'] * 5
    live.update(message='中文測試 / English test')
    fresh = training.Live()
    assert fresh.status['message'] == '中文測試 / English test'
    assert json.loads((training.OUT / 'status.json').read_text(encoding='utf-8')) == fresh.status


def test_restore_verifies_before_writing_and_preserves_student_checkpoint(tmp_path, monkeypatch):
    import restore_public_evidence as restore
    import hashlib
    monkeypatch.setattr(restore, 'ROOT', tmp_path)
    source, target = tmp_path / 'published.pt', tmp_path / 'checkpoints/head.pt'
    source.write_bytes(b'published model')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='hash mismatch'):
        restore.restore(source, target, '0' * 64)
    assert not target.exists()
    restore.restore(source, target, digest)
    restore.restore(source, target, digest)
    target.write_bytes(b'student model')
    with pytest.raises(FileExistsError, match='overwrite'):
        restore.restore(source, target, digest)
    assert target.read_bytes() == b'student model'


def test_cpu_extractor_recovers_from_an_upstream_cuda_default(monkeypatch):
    if not (ROOT / 'data/flyvis/results/flow/0000/000').is_dir():
        pytest.skip('Official Flyvis model not downloaded')
    import torch
    from flyvis.network import initialization
    import numpy as np
    from fruitfly_sim.flyvis_features import FlyvisFeatureExtractor
    # Simulate an upstream device captured on a GPU-capable computer, even on
    # a CPU-only test runner. Without the reset model construction would fail.
    monkeypatch.setattr(initialization, 'device', torch.device('cuda'))
    extractor = FlyvisFeatureExtractor()
    values = extractor.transform(np.full((240, 320, 3), 128, dtype=np.uint8))
    assert values.shape == (72,) and np.isfinite(values).all()
