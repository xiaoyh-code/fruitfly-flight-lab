"""Display-only MuJoCo drone observer, independent of the policy's FPV camera.

This renderer reads an existing environment's current state. It never steps or
forwards physics, modifies model geometry, or changes any named camera. Only the
offscreen framebuffer capacity may be enlarged to accommodate the display image.
"""
from __future__ import annotations

import math
import mujoco
import numpy as np


class DroneObserver:
    """Render a close follow view or a route-framed overview as RGB uint8.

    ``env`` must expose ``model`` and ``data`` (e.g. SplatNavigationEnv).
    Create one observer per environment; call close before disposing the env.
    The follow view is a MuJoCo observer, not captured 3DGS or policy input.
    """
    def __init__(self, env, width=960, height=540):
        if isinstance(width, bool) or isinstance(height, bool):
            raise ValueError("Observer dimensions must be positive integers")
        self.width, self.height = int(width), int(height)
        if self.width != width or self.height != height or min(self.width, self.height) < 1:
            raise ValueError("Observer dimensions must be positive integers")
        self.env = env
        self.model, self.data = env.model, env.data
        self.body_id = self.model.body("cf2").id
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), self.width)
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), self.height)
        self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self._camera)
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._camera.distance = .8
        self._camera.elevation = -25.
        self._overview_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self._overview_camera)
        self._overview_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._overview_camera.azimuth = 135.
        self._overview_camera.elevation = -55.
        self._closed = False

    def render(self, mode="follow"):
        if self._closed:
            raise RuntimeError("DroneObserver is closed")
        if mode == "follow":
            self._camera.lookat[:] = self.data.xpos[self.body_id]
            rotation = self.data.xmat[self.body_id].reshape(3, 3)
            heading = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
            self._camera.azimuth = heading + 145.
            self._renderer.update_scene(self.data, camera=self._camera)
        elif mode == "overview":
            points = [self.data.xpos[self.body_id, :2].copy()]
            for name in ("home", "goal", "obstacle_center"):
                value = getattr(self.env, name, None)
                if value is not None:
                    value = np.asarray(value).reshape(-1)
                    if len(value) >= 2 and np.isfinite(value[:2]).all():
                        points.append(value[:2])
            points = np.asarray(points)
            lower, upper = points.min(axis=0), points.max(axis=0)
            center = (lower + upper) / 2
            self._overview_camera.lookat[:] = [center[0], center[1], .7]
            # Fit the route, with ground/altitude margin, rather than reusing
            # the far-away named camera. A 3.2m hall route needs about 5.2m.
            self._overview_camera.distance = max(4.5, float(np.linalg.norm(upper - lower)) * 1.55 + .2)
            self._renderer.update_scene(self.data, camera=self._overview_camera)
        else:
            raise ValueError("Observer mode must be 'follow' or 'overview'")
        return self._renderer.render().copy()

    def close(self):
        if not self._closed:
            self._renderer.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
