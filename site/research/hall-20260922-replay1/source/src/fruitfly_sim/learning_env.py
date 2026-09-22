"""Small, explicitly state-based MuJoCo navigation curricula.

This baseline learns navigation above an existing attitude/altitude controller.
It is NOT vision-only, a Flyvis policy, or an identified real-drone model.
Observations (24 floats, all clipped to [-1, 1]):
  0:2 goal displacement / 4 m; 2:4 world velocity / 0.8 m/s;
  4:6 home displacement / 4 m; 6 altitude error / 0.5 m; 7 return phase;
  8:20 twelve evenly spaced synthetic horizontal range rays / 4 m;
  20:22 obstacle displacement / 4 m; 22 obstacle radius / 1 m; 23 active.
Range rays and obstacle coordinates use simulator geometry, not camera inference.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .env import FruitFlyDroneEnv, _scene_xml


class NavigationLearningEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}
    SPEED = 0.8
    ALTITUDE = 1.2
    SUCCESS_RADIUS = 0.25
    OBSTACLE_RADIUS = 0.42
    RANGE = 4.0

    def __init__(self, stage="reach", render_mode=None):
        super().__init__()
        if stage not in {"reach", "avoid", "return"}:
            raise ValueError("stage must be 'reach', 'avoid', or 'return'")
        self.stage = stage
        self.render_mode = render_mode
        self.base = FruitFlyDroneEnv(render_mode=render_mode, episode_seconds=40)
        # Build an uncluttered training arena in memory. The upstream XML and
        # the original demonstration environment are never changed.
        root = ET.fromstring(_scene_xml(self.base.model_path, 320, 240))
        world = root.find("worldbody")
        for geom in list(world.findall("geom")):
            if geom.get("name") not in {"ground", "home_marker"}:
                world.remove(geom)
        obstacle = ET.SubElement(world, "body", name="training_obstacle", mocap="true", pos="0 0 0.9")
        ET.SubElement(obstacle, "geom", name="training_obstacle_geom", type="cylinder",
                      size=f"{self.OBSTACLE_RADIUS} 0.9", rgba="0.85 0.28 0.12 1")
        ET.SubElement(world, "geom", name="goal_marker", type="cylinder", size="0.25 0.012",
                      pos="0 0 0.02", rgba="0.1 0.85 0.6 1", contype="0", conaffinity="0")
        self.base.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.base.data = mujoco.MjData(self.base.model)
        self.base.body_id = self.base.model.body("cf2").id
        self.base.mass = float(self.base.model.body_mass[self.base.body_id])
        self.base.inertia = self.base.model.body_inertia[self.base.body_id].copy()
        self._obstacle_geom = self.base.model.geom("training_obstacle_geom").id
        self._obstacle_mocap = self.base.model.body_mocapid[self.base.model.body("training_obstacle").id]
        self._home_geom = self.base.model.geom("home_marker").id
        self._goal_geom = self.base.model.geom("goal_marker").id
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, (24,), dtype=np.float32)
        self.dt = self.base.dt * 5
        self.max_episode_steps = 200 if stage != "return" else 350
        self.home = np.zeros(2)
        self.goal = np.zeros(2)
        self.outbound_goal = np.zeros(2)
        self.obstacle_center = np.zeros(2)
        self.obstacle_active = stage != "reach"
        self.phase = 0
        self.episode_steps = 0
        self._done = True
        self._expert_waypoints = []
        self._expert_index = 0
        self._last_action = np.zeros(2)

    @property
    def model(self):
        return self.base.model

    @property
    def data(self):
        return self.base.data

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        angle = self.np_random.uniform(-math.pi, math.pi)
        direction = np.array([math.cos(angle), math.sin(angle)])
        lateral = np.array([-direction[1], direction[0]])
        length = self.np_random.uniform(2.0, 3.6)
        center = self.np_random.uniform(-0.35, 0.35, 2)
        self.home = center - direction * length / 2
        self.outbound_goal = center + direction * length / 2
        self.goal = self.outbound_goal.copy()
        self.obstacle_center = center + lateral * self.np_random.uniform(-0.24, 0.24)
        self.phase = 0
        self.episode_steps = 0
        self._done = False
        self._last_action[:] = 0
        self.base.reset(seed=seed, options={"position": [*self.home, self.ALTITUDE],
                                           "goal": [*self.goal, self.ALTITUDE]})
        self.model.geom_contype[self._obstacle_geom] = int(self.obstacle_active)
        self.model.geom_conaffinity[self._obstacle_geom] = int(self.obstacle_active)
        self.model.geom_rgba[self._obstacle_geom, 3] = float(self.obstacle_active)
        self.data.mocap_pos[self._obstacle_mocap] = [*self.obstacle_center, 0.9]
        self.model.geom_pos[self._home_geom, :2] = self.home
        self._update_goal_marker()
        mujoco.mj_forward(self.model, self.data)
        self._plan_expert()
        return self.state_to_features(), self._info()

    def _update_goal_marker(self):
        self.model.geom_pos[self._goal_geom, :2] = self.goal
        self.base.set_goal([*self.goal, self.ALTITUDE])

    def _ranges(self):
        rays = np.ones(12)
        if not self.obstacle_active:
            return rays
        offset = self.obstacle_center - self.data.qpos[:2]
        dist2 = float(offset @ offset)
        if dist2 < self.OBSTACLE_RADIUS ** 2:
            return np.zeros(12)
        for i, angle in enumerate(np.arange(12) * (2 * math.pi / 12)):
            along = float(offset @ [math.cos(angle), math.sin(angle)])
            discriminant = self.OBSTACLE_RADIUS ** 2 - (dist2 - along * along)
            if along > 0 and discriminant >= 0:
                rays[i] = np.clip((along - math.sqrt(discriminant)) / self.RANGE, 0, 1)
        return rays

    def state_to_features(self):
        position = self.data.qpos[:2]
        obstacle = (self.obstacle_center - position) / 4 if self.obstacle_active else np.zeros(2)
        values = np.concatenate(((self.goal - position) / 4, self.data.qvel[:2] / self.SPEED,
                                 (self.home - position) / 4,
                                 [(self.data.qpos[2] - self.ALTITUDE) / 0.5, float(self.phase)],
                                 self._ranges(), obstacle,
                                 [self.OBSTACLE_RADIUS if self.obstacle_active else 0,
                                  float(self.obstacle_active)]))
        return np.clip(values, -1, 1).astype(np.float32)

    def _info(self, **extra):
        info = {"is_success": False, "collision": False, "phase": self.phase,
                "distance_to_goal": float(np.linalg.norm(self.goal - self.data.qpos[:2])),
                "episode_steps": self.episode_steps, "stage": self.stage,
                "time": float(self.data.time), "goal": self.goal.copy(),
                "position": self.data.qpos[:3].copy(), "obstacle_active": self.obstacle_active}
        info.update(extra)
        return info

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset() before stepping a completed episode")
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("action must contain two finite normalized velocity commands")
        action = np.clip(action, -1, 1)
        action = action / max(1.0, float(np.linalg.norm(action)))
        before = float(np.linalg.norm(self.goal - self.data.qpos[:2]))
        collided = False
        physics_ended = False
        for _ in range(5):
            vz = np.clip(2 * (self.ALTITUDE - self.data.qpos[2]), -0.5, 0.5)
            _, _, terminated, truncated, info = self.base.step([*(action * self.SPEED), vz, 0])
            collided = collided or info["collision"]
            if terminated or truncated:
                physics_ended = True
                break
        self.episode_steps += 1
        after = float(np.linalg.norm(self.goal - self.data.qpos[:2]))
        reward = 4.0 * (before - after) - 0.025
        reward -= 0.003 * float(np.sum((action - self._last_action) ** 2))
        self._last_action = action.copy()
        if self.obstacle_active:
            clearance = float(np.linalg.norm(self.data.qpos[:2] - self.obstacle_center)) - self.OBSTACLE_RADIUS
            reward -= 0.08 * max(0.0, (0.35 - clearance) / 0.35)
        escaped = np.linalg.norm(self.data.qpos[:2]) > 4.0
        terminated = bool(collided or physics_ended or escaped)
        success = False
        reached_outbound = False
        if terminated:
            reward -= 12.0
        elif after < self.SUCCESS_RADIUS:
            if self.stage == "return" and self.phase == 0:
                self.phase = 1
                self.goal = self.home.copy()
                self._update_goal_marker()
                self._plan_expert()
                reached_outbound = True
                reward += 8.0
            else:
                reward += 20.0
                terminated = True
                success = True
        truncated = bool(self.episode_steps >= self.max_episode_steps and not terminated)
        self._done = terminated or truncated
        return self.state_to_features(), float(reward), terminated, truncated, self._info(
            is_success=success, collision=collided, reached_outbound=reached_outbound,
            out_of_bounds=bool(escaped))

    def _plan_expert(self):
        """Demonstrator only: a geometry-aware waypoint around one cylinder."""
        start = self.data.qpos[:2].copy()
        direction = self.goal - start
        direction /= max(np.linalg.norm(direction), 1e-8)
        lateral = np.array([-direction[1], direction[0]])
        # Pick a fixed world-side tie break, making demonstrations unambiguous.
        if lateral[1] < 0 or (abs(lateral[1]) < 1e-8 and lateral[0] < 0):
            lateral = -lateral
        if self.obstacle_active:
            clearance = self.OBSTACLE_RADIUS + 0.62
            self._expert_waypoints = [self.obstacle_center - direction * 0.7 + lateral * clearance,
                                      self.obstacle_center + direction * 0.7 + lateral * clearance,
                                      self.goal.copy()]
        else:
            self._expert_waypoints = [self.goal.copy()]
        self._expert_index = 0

    def expert_action(self):
        """Return a privileged demonstrator action; never used inside step()."""
        position = self.data.qpos[:2]
        while self._expert_index < len(self._expert_waypoints) - 1:
            if np.linalg.norm(self._expert_waypoints[self._expert_index] - position) >= 0.3:
                break
            self._expert_index += 1
        offset = self._expert_waypoints[self._expert_index] - position
        velocity = 1.6 * offset - 0.25 * self.data.qvel[:2]
        velocity *= min(1.0, self.SPEED / max(np.linalg.norm(velocity), 1e-9))
        return (velocity / self.SPEED).astype(np.float32)

    def render(self, camera="fpv"):
        return self.base.render(camera=camera)

    def close(self):
        self.base.close()
