"""Synchronous RGB-only camera bridge to the local Unity 3DGS renderer.

The simulator advances only after the matching camera response arrives. All
poses sent over the wire use MuJoCo's z-up world; the Unity process owns the
coordinate conversion. No depth, obstacle geometry or collision labels enter
the returned RGB image or Flyvis feature extractor.
"""
from __future__ import annotations

import json
import socket
import struct
import time

import cv2
import numpy as np


def camera_vectors(data, camera_id, offset=(0, 0, 0)):
    rotation = np.asarray(data.cam_xmat[camera_id]).reshape(3, 3)
    return (np.asarray(data.cam_xpos[camera_id]) + np.asarray(offset),
            -rotation[:, 2], rotation[:, 1])


def look_at(position, target):
    position, target = np.asarray(position, float), np.asarray(target, float)
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    if np.linalg.norm(right) < 1e-6:
        raise ValueError("Camera must not look exactly vertically")
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return position, forward, up


class UnitySplatRenderer:
    def __init__(self, host="127.0.0.1", port=8767, *, timeout=60., width=320, height=240):
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError("The rendering bridge is local-only")
        self.socket = socket.create_connection((host, port), timeout=timeout)
        self.socket.settimeout(timeout)
        self.width, self.height = width, height
        self.sequence = 0
        self.latencies = []
        self.last_header = None

    def _read_exact(self, count):
        chunks = bytearray()
        while len(chunks) < count:
            part = self.socket.recv(count - len(chunks))
            if not part:
                raise ConnectionError("Unity renderer disconnected before completing the image")
            chunks.extend(part)
        return bytes(chunks)

    def render(self, position, forward, up):
        vectors = [np.asarray(v, float) for v in (position, forward, up)]
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors):
            raise ValueError("Camera position and axes must each have three finite values")
        if min(np.linalg.norm(v) for v in vectors[1:]) < 1e-6 or np.linalg.norm(np.cross(*vectors[1:])) < 1e-6:
            raise ValueError("Camera axes must be nonzero and independent")
        self.sequence += 1
        request = dict(id=self.sequence, cam_pos=vectors[0].tolist(),
                       cam_forward=vectors[1].tolist(), cam_up=vectors[2].tolist())
        started = time.perf_counter()
        self.socket.sendall(json.dumps(request, allow_nan=False).encode() + b"\n")
        header_length, = struct.unpack("<I", self._read_exact(4))
        if not 1 <= header_length <= 65536:
            raise RuntimeError(f"Invalid renderer header size {header_length}")
        header = json.loads(self._read_exact(header_length))
        image_length, = struct.unpack("<I", self._read_exact(4))
        if image_length > 16 * 1024 * 1024:
            raise RuntimeError("Renderer image exceeds the protocol limit")
        encoded = self._read_exact(image_length)
        if header.get("id") != self.sequence:
            raise RuntimeError("Stale or mismatched camera response; refusing to evaluate policy")
        if header.get("error"):
            raise RuntimeError(f"Unity renderer: {header['error']}")
        if (header.get("width"), header.get("height"), header.get("format")) != (self.width, self.height, "png"):
            raise RuntimeError(f"Unexpected camera format: {header}")
        bgr = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (self.height, self.width, 3):
            raise RuntimeError("Unity returned an invalid camera image")
        self.latencies.append(time.perf_counter() - started)
        self.last_header = header
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
