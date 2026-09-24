"""Narrow compatibility fix for Datamate 1.0's Windows HDF5 cache writer.

Datamate 1.0 opens mode='w', then tries to read a dataset that was just removed
by truncation. Its fallback unlinks the file before closing that handle, which
Windows rejects with WinError 32. Write the same single 'data' dataset within
one context manager instead. This changes cache I/O, not visual model weights.
Upstream fix: https://github.com/flyvis/datamate/commit/3b9792c3c90fb29d741f8185c7aca912aa0c0942
"""
from __future__ import annotations
from importlib.metadata import version
import os
from pathlib import Path
import sys
import tempfile


def write_cache_array(path, value):
    import h5py
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        path.rmdir()  # Only an empty directory, matching the upstream contract.
    descriptor, temporary = tempfile.mkstemp(prefix='.flyvis-', suffix='.h5', dir=path.parent)
    os.close(descriptor)
    try:
        with h5py.File(temporary, mode='w', libver='latest') as stream:
            stream.create_dataset('data', data=np.asarray(value))
            stream.swmr_mode = True
            stream.flush()
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def configure_flyvis_cache():
    if sys.platform != 'win32':
        return
    if version('datamate') != '1.0.0':
        raise RuntimeError('This Windows cache adapter requires pinned datamate==1.0.0. Reinstall requirements-vision.txt.')
    import datamate.io as io
    import datamate.directory as directory
    # directory imports the writer by value; both references must be updated.
    io._write_h5 = write_cache_array
    directory._write_h5 = write_cache_array
