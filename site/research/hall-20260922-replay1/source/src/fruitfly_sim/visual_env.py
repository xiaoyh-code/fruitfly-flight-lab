"""Small forward-facing visual navigation task with GPS/velocity assistance.

Policy observations contain only frozen Flyvis camera features, relative goal,
velocity and previous command. Obstacle geometry is available solely to the
explicit demonstration teacher and success/collision evaluator.
"""
from __future__ import annotations
import numpy as np
import mujoco
from gymnasium import spaces
from .learning_env import NavigationLearningEnv


class VisualNavigationEnv(NavigationLearningEnv):
    def __init__(self, extractor, render_mode="rgb_array"):
        super().__init__(stage="avoid", render_mode=render_mode)
        self.extractor = extractor
        self.observation_space = spaces.Box(-1, 1, shape=(78,), dtype=np.float32)
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.latest_features = np.zeros(72, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.home = np.array([-1.6, self.np_random.uniform(-0.35, 0.35)])
        self.goal = np.array([1.6, self.np_random.uniform(-0.35, 0.35)])
        self.outbound_goal = self.goal.copy()
        obstacle_x = self.np_random.uniform(-0.45, 0.45)
        centerline_y = self.home[1] + (self.goal[1] - self.home[1]) * ((obstacle_x + 1.6) / 3.2)
        # The offset is independently randomized and excluded from observations.
        # Avoid near-exact ties so the demonstrator's choice is visually resolvable.
        offset = self.np_random.choice([-1, 1]) * self.np_random.uniform(0.20, 0.65)
        self.obstacle_center = np.array([obstacle_x, centerline_y + offset])
        self.base.reset(seed=seed, options={"position": [*self.home, self.ALTITUDE],
                                           "goal": [*self.goal, self.ALTITUDE]})
        self.data.mocap_pos[self._obstacle_mocap] = [*self.obstacle_center, 0.9]
        self.model.geom_pos[self._home_geom, :2] = self.home
        self._update_goal_marker()
        mujoco.mj_forward(self.model, self.data)
        self.previous_action[:] = 0
        self.extractor.reset()
        self._plan_expert()
        return self.visual_observation(), self._info()

    def visual_observation(self):
        self.latest_features = self.extractor.transform(self.render("fpv"))
        sensors = np.concatenate(((self.goal - self.data.qpos[:2]) / 4,
                                  self.data.qvel[:2] / self.SPEED,
                                  self.previous_action))
        return np.concatenate((self.latest_features, np.clip(sensors, -1, 1))).astype(np.float32)

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)
        self.previous_action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        return self.visual_observation(), reward, terminated, truncated, info

    def _plan_expert(self):
        start = self.data.qpos[:2].copy()
        direction = self.goal - start
        length = max(np.linalg.norm(direction), 1e-8)
        direction = direction / length
        lateral = np.array([-direction[1], direction[0]])
        signed_offset = float((self.obstacle_center - start) @ lateral)
        # Choose the shorter, visually indicated side. This privileged geometry
        # teacher supplies labels during training only, never policy evaluation.
        side = -1.0 if signed_offset >= 0 else 1.0
        clearance = self.OBSTACLE_RADIUS + 0.48
        self._expert_waypoints = [self.obstacle_center - direction * 0.65 + side * lateral * clearance,
                                  self.obstacle_center + direction * 0.65 + side * lateral * clearance,
                                  self.goal.copy()]
        self._expert_index = 0
