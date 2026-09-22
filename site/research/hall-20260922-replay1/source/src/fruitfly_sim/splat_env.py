"""Captured-hall RGB + existing MuJoCo drone, with an approximate object proxy.

The only scene geometry is a manually fitted cylinder for a scanned red object.
It supplies evaluation/collision labels, NEVER policy observations. This pilot
does not have a measured, complete collision mesh for the hall.
"""
from __future__ import annotations
import json
import time
import uuid
from urllib.request import Request, urlopen

import cv2
import mujoco
import numpy as np

from .visual_env import VisualNavigationEnv
from .unity_renderer import camera_vectors


class WebSplatRenderer:
    """Actual WebGL 3DGS RGB; temporary backend while Unity activation is pending."""
    def __init__(self, port=8768):
        self.url = f"http://127.0.0.1:{int(port)}/render"
        self.sequence = 0
        self.session_id = uuid.uuid4().hex
        self.latencies = []

    def render(self, position, forward, up):
        self.sequence += 1
        request_id = f"{self.session_id}-{self.sequence}"
        data = dict(id=request_id, cam_pos=np.asarray(position).tolist(),
                    cam_forward=np.asarray(forward).tolist(), cam_up=np.asarray(up).tolist())
        request = Request(self.url, data=json.dumps(data,allow_nan=False).encode(),
                          headers={"Content-Type":"application/json"})
        started = time.perf_counter()
        with urlopen(request,timeout=55) as response:
            if response.headers.get("X-Frame-Id") != request_id:
                raise RuntimeError("Renderer response did not match the requested camera pose")
            encoded = response.read(16*1024*1024)
        bgr = cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (240,320,3):
            raise RuntimeError("Expected a 320x240 RGB observation from the 3DGS renderer")
        self.latencies.append(time.perf_counter()-started)
        return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)

    def close(self): pass


class SplatNavigationEnv(VisualNavigationEnv):
    OBSTACLE_RADIUS = 0.45
    SCENE_OFFSET = np.array([15.15,-3.15,0.0])
    PROXY_CENTER = np.array([0.,0.2])
    VEHICLE_RADIUS = 0.09  # Conservative horizontal envelope, used for clearance reporting.

    def __init__(self, extractor, renderer):
        self.scene_renderer = renderer
        self.latest_rgb = None
        self.minimum_clearance = float("inf")
        super().__init__(extractor)
        self.model.geom_size[self._obstacle_geom,:2] = [self.OBSTACLE_RADIUS,0.775]
        self.model.geom_rbound[self._obstacle_geom] = np.hypot(self.OBSTACLE_RADIUS,0.775)
        self.max_episode_steps = 120

    def reset(self, *, seed=None, options=None):
        # Reuse RNG initialization and bookkeeping without consuming a fake visual frame.
        from .learning_env import NavigationLearningEnv
        NavigationLearningEnv.reset(self,seed=seed,options=options)
        self.home = np.array([-1.6,self.np_random.uniform(-0.22,0.22)])
        self.goal = np.array([1.6,self.np_random.uniform(-0.22,0.22)])
        self.outbound_goal = self.goal.copy()
        self.obstacle_center = self.PROXY_CENTER.copy()
        self.base.reset(seed=seed,options={"position":[*self.home,self.ALTITUDE],
                                         "goal":[*self.goal,self.ALTITUDE]})
        self.data.mocap_pos[self._obstacle_mocap] = [*self.obstacle_center,0.775]
        self.model.geom_pos[self._home_geom,:2] = self.home
        self._update_goal_marker()
        mujoco.mj_forward(self.model,self.data)
        self.minimum_clearance = self.clearance()
        self.previous_action[:] = 0
        self.extractor.reset()
        self._plan_expert()
        return self.visual_observation(),self._info()

    def clearance(self):
        return float(np.linalg.norm(self.data.qpos[:2]-self.obstacle_center)-self.OBSTACLE_RADIUS-self.VEHICLE_RADIUS)

    def render(self, camera="fpv"):
        if camera != "fpv": return super().render(camera)
        cam_id = self.model.camera("fpv").id
        self.latest_rgb = self.scene_renderer.render(*camera_vectors(self.data,cam_id,self.SCENE_OFFSET))
        return self.latest_rgb.copy()

    def step(self, action):
        result=super().step(action)
        self.minimum_clearance=min(self.minimum_clearance,self.clearance())
        result[-1]["approx_proxy_envelope_clearance_m"]=self.minimum_clearance
        return result

    def close(self):
        super().close()
