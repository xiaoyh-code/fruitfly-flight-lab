"""Separate, privileged Gymnasium/MuJoCo landing and multi-obstacle baselines.

These tasks do not use Flyvis or 3DGS. A learned policy commands world xyz
velocity above the existing attitude controller. Observations have 52 floats:
relative target xyz / 6, velocity xyz / SPEED, body-up xyz, previous action xyz,
eight [relative obstacle x/6, y/6, radius, active] slots, three phase flags
(approach, descend, settling), relative pad xyz / 6, pad radius, landing flag.
All values are clipped to [-1, 1]. Obstacle and pad coordinates are privileged
simulator geometry. expert_action() is a separate geometry-aware demonstrator;
step() never calls it. Existing M1--M4 environments/checkpoints are unaffected.

A landing requires actual pad-top contact, a safe *pre-impact* speed/attitude,
full conservative horizontal footprint containment, motors off, and 0.5 seconds
of stable contact. This is an uncalibrated rigid-body baseline, not flight safety
validation. Pad height, dimensions and thresholds are engineering task choices.
min_clearance is a horizontal obstacle-envelope metric; 10 m is a sentinel when
there are no obstacles (landing task), not a measured distance to ground/pad.
"""
from __future__ import annotations

import heapq
import math
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .env import FruitFlyDroneEnv, _look_at, _scene_xml


class MissionEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}
    OBSERVATION_SIZE = 52
    MAX_OBSTACLES = 8
    SPEED = np.array([0.8, 0.8, 0.4])
    ALTITUDE = 1.2
    PAD_RADIUS = 0.55
    PAD_TOP = 0.10
    # Measured level collision-mesh bottom in the installed Menagerie model.
    FOOT_OFFSET = 0.012517
    VEHICLE_RADIUS = 0.09
    TOUCHDOWN_VERTICAL_LIMIT = 0.2
    TOUCHDOWN_HORIZONTAL_LIMIT = 0.15
    TOUCHDOWN_TILT_LIMIT = 10.0
    STABLE_CONTACT_SECONDS = 0.5

    def __init__(self, task="landing", obstacle_count=3, render_mode=None):
        super().__init__()
        if task not in {"landing", "obstacles", "obstacle_landing"}:
            raise ValueError("task must be landing, obstacles, or obstacle_landing")
        if obstacle_count not in {3, 5, 8}:
            raise ValueError("obstacle_count must be 3, 5, or 8")
        if render_mode not in {None, "rgb_array"}:
            raise ValueError("render_mode must be None or rgb_array")
        self.task = task
        self.obstacle_count = 0 if task == "landing" else obstacle_count
        self.landing_enabled = task != "obstacles"
        self.render_mode = render_mode
        self.base = FruitFlyDroneEnv(episode_seconds=60, render_mode=render_mode)
        root = ET.fromstring(_scene_xml(self.base.model_path, 320, 240))
        world = root.find("worldbody")
        for geom in list(world.findall("geom")):
            if geom.get("name") != "ground":
                world.remove(geom)
        overview = world.find("camera[@name='overview']")
        overview.set("pos", "7 -9 8")
        overview.set("xyaxes", _look_at([7, -9, 8], [0, 0, 0.6]))
        drone = world.find("body[@name='cf2']")
        ET.SubElement(drone, "camera", name="down", pos="0 0 -0.010",
                      xyaxes="1 0 0 0 1 0", fovy="85")
        for index in range(self.MAX_OBSTACLES):
            body = ET.SubElement(world, "body", name=f"mission_obstacle_{index}",
                                 mocap="true", pos="0 0 1")
            ET.SubElement(body, "geom", name=f"mission_obstacle_geom_{index}",
                          type="cylinder", size="0.3 1", rgba="0.8 0.25 0.12 1")
        pad = ET.SubElement(world, "body", name="landing_pad", mocap="true", pos="0 0 0.05")
        ET.SubElement(pad, "geom", name="landing_pad_geom", type="cylinder",
                      size=f"{self.PAD_RADIUS} {self.PAD_TOP / 2}",
                      rgba="0.1 0.62 0.35 1", friction="1 0.005 0.0001")
        # White, non-colliding centre markings are part of the synthetic image.
        for name, size in [("pad_mark_x", "0.22 0.025 0.001"),
                           ("pad_mark_y", "0.025 0.22 0.001")]:
            ET.SubElement(pad, "geom", name=name, type="box", size=size,
                          pos=f"0 0 {self.PAD_TOP / 2 + 0.001}", rgba="0.95 0.95 0.95 1",
                          contype="0", conaffinity="0")
        self.base.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.base.data = mujoco.MjData(self.base.model)
        self.base.body_id = self.model.body("cf2").id
        self.base.mass = float(self.model.body_mass[self.base.body_id])
        self.base.inertia = self.model.body_inertia[self.base.body_id].copy()
        self._obstacle_geoms = [self.model.geom(f"mission_obstacle_geom_{i}").id
                               for i in range(self.MAX_OBSTACLES)]
        self._obstacle_mocaps = [self.model.body_mocapid[self.model.body(f"mission_obstacle_{i}").id]
                                for i in range(self.MAX_OBSTACLES)]
        self._pad_geom = self.model.geom("landing_pad_geom").id
        self._pad_mocap = self.model.body_mocapid[self.model.body("landing_pad").id]
        self._ground_geom = self.model.geom("ground").id
        self.action_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-1, 1, shape=(self.OBSERVATION_SIZE,), dtype=np.float32)
        self.dt = 0.1
        self.frame_skip = 50
        self.max_episode_steps = 500
        self.obstacles = np.zeros((self.MAX_OBSTACLES, 3))  # x, y, radius
        self.goal = np.array([2.8, 0, self.PAD_TOP + self.FOOT_OFFSET])
        self.home = np.array([-2.8, 0, self.ALTITUDE])
        self.phase = "approach"
        self.previous_action = np.zeros(3)
        self._done = True
        self._path = []
        self._path_index = 0
        self._reset_metrics()

    @property
    def model(self):
        return self.base.model

    @property
    def data(self):
        return self.base.data

    def _reset_metrics(self):
        self.episode_steps = 0
        self.stable_contact_seconds = 0.0
        self.touchdown_speed = None
        self.touchdown_horizontal_speed = None
        self.touchdown_tilt_deg = None
        self.min_clearance = 10.0
        self.failure_reason = ""
        self._contact_seen = False
        self._collision = False
        self._success = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.home = np.array([-2.8, self.np_random.uniform(-0.45, 0.45),
                              self.np_random.uniform(1.0, 1.5)])
        self.goal = np.array([2.8, self.np_random.uniform(-0.45, 0.45),
                              self.PAD_TOP + self.FOOT_OFFSET if self.landing_enabled else self.ALTITUDE])
        if self.task == "landing":
            self.home[:2] = self.goal[:2] + self.np_random.uniform(-0.9, 0.9, 2)
        if "position" in options:
            position = np.asarray(options["position"], dtype=float)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError("position must contain three finite values")
            self.home = position.copy()
        if "goal_xy" in options:
            goal_xy = np.asarray(options["goal_xy"], dtype=float)
            if goal_xy.shape != (2,) or not np.all(np.isfinite(goal_xy)):
                raise ValueError("goal_xy must contain two finite values")
            self.goal[:2] = goal_xy
        self.obstacles[:] = 0
        # Rejection sampling keeps cylinders separate; a grid path is checked
        # before accepting a layout. Every seed owns its own complete layout.
        for _ in range(100):
            proposed = []
            for _attempt in range(1000):
                if len(proposed) == self.obstacle_count:
                    break
                candidate = np.array([self.np_random.uniform(-1.85, 1.85),
                                      self.np_random.uniform(-1.1, 1.1),
                                      self.np_random.uniform(0.22, 0.33)])
                if all(np.linalg.norm(candidate[:2] - old[:2]) > candidate[2] + old[2] + 0.16
                       for old in proposed):
                    proposed.append(candidate)
            if len(proposed) != self.obstacle_count:
                continue
            if proposed:
                self.obstacles[:self.obstacle_count] = proposed
            try:
                self._path = self._plan_path(self.home[:2], self.goal[:2])
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError("Could not sample a connected obstacle layout")
        self.base.reset(seed=seed, options={"position": self.home, "goal": self.goal})
        for index, (gid, mid) in enumerate(zip(self._obstacle_geoms, self._obstacle_mocaps)):
            active = index < self.obstacle_count
            self.model.geom_contype[gid] = int(active)
            self.model.geom_conaffinity[gid] = int(active)
            self.model.geom_rgba[gid, 3] = float(active)
            self.model.geom_size[gid, 0] = self.obstacles[index, 2] if active else 0.3
            self.model.geom_rbound[gid] = math.hypot(self.model.geom_size[gid, 0], 1.0)
            # Runtime radius changes also need broad-phase bounds. Each mocap
            # body owns exactly this one axis-aligned cylinder / BVH leaf.
            self.model.geom_aabb[gid, 3:5] = self.model.geom_size[gid, 0]
            body = self.model.geom_bodyid[gid]
            leaf = self.model.body_bvhadr[body]
            self.model.bvh_aabb[leaf, 3:5] = self.model.geom_size[gid, 0]
            self.data.mocap_pos[mid] = [*self.obstacles[index, :2], 1.0] if active else [0, 0, -3]
        self.data.mocap_pos[self._pad_mocap] = [*self.goal[:2], self.PAD_TOP / 2]
        self.phase = str(options.get("phase", "approach"))
        if self.phase not in {"approach", "descend"}:
            raise ValueError("reset phase must be approach or descend")
        self.previous_action[:] = 0
        self._reset_metrics()
        self._path_index = 0
        self._done = False
        mujoco.mj_forward(self.model, self.data)
        self.min_clearance = self._clearance()
        return self._observation(), self._info()

    def _target(self):
        target = self.goal.copy()
        if self.phase == "approach":
            target[2] = self.ALTITUDE
        return target

    def _observation(self):
        obstacle_slots = np.zeros((self.MAX_OBSTACLES, 4))
        for index in range(self.obstacle_count):
            obstacle_slots[index] = [*((self.obstacles[index, :2] - self.data.qpos[:2]) / 6),
                                      self.obstacles[index, 2], 1.0]
        phases = [float(self.phase == p) for p in ("approach", "descend", "settling")]
        values = np.concatenate(((self._target() - self.data.qpos[:3]) / 6,
                                 self.data.qvel[:3] / self.SPEED,
                                 self.data.xmat[self.base.body_id].reshape(3, 3)[:, 2],
                                 self.previous_action, obstacle_slots.ravel(), phases,
                                 (self.goal - self.data.qpos[:3]) / 6,
                                 [self.PAD_RADIUS, float(self.landing_enabled)]))
        return np.clip(values, -1, 1).astype(np.float32)

    def _tilt(self):
        up_z = self.data.xmat[self.base.body_id].reshape(3, 3)[2, 2]
        return math.degrees(math.acos(float(np.clip(up_z, -1, 1))))

    def _clearance(self):
        if not self.obstacle_count:
            return 10.0
        return float(np.min(np.linalg.norm(self.obstacles[:self.obstacle_count, :2] - self.data.qpos[:2], axis=1)
                            - self.obstacles[:self.obstacle_count, 2] - self.VEHICLE_RADIUS))

    def _drone_contacts(self):
        contacts = []
        for contact in self.data.contact:
            b1, b2 = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if (b1 == self.base.body_id) != (b2 == self.base.body_id):
                other = contact.geom2 if b1 == self.base.body_id else contact.geom1
                contacts.append((int(other), contact))
        return contacts

    def _fail(self, reason, collision=False):
        self.failure_reason = reason
        self._collision = bool(collision)
        self._done = True

    def _process_contacts(self, contacts, before_velocity, before_tilt):
        if not contacts:
            self.stable_contact_seconds = 0.0
            return
        if any(gid != self._pad_geom for gid, _ in contacts) or not self.landing_enabled:
            self._fail("obstacle_or_ground_contact", collision=True)
            return
        contained = np.linalg.norm(self.data.qpos[:2] - self.goal[:2]) + self.VEHICLE_RADIUS <= self.PAD_RADIUS
        top_only = all(c.pos[2] >= self.PAD_TOP - 0.005 and abs(c.frame[2]) >= 0.9
                       for _, c in contacts)
        if not contained or not top_only:
            self._fail("pad_edge_or_side_contact", collision=True)
            return
        if not self._contact_seen:
            self._contact_seen = True
            self.touchdown_speed = max(0.0, -float(before_velocity[2]))
            self.touchdown_horizontal_speed = float(np.linalg.norm(before_velocity[:2]))
            self.touchdown_tilt_deg = float(before_tilt)
            if (self.touchdown_speed > self.TOUCHDOWN_VERTICAL_LIMIT or
                    self.touchdown_horizontal_speed > self.TOUCHDOWN_HORIZONTAL_LIMIT or
                    self.touchdown_tilt_deg > self.TOUCHDOWN_TILT_LIMIT):
                self._fail("unsafe_touchdown", collision=True)
                return
            self.phase = "settling"
        if (np.linalg.norm(self.data.qvel[:2]) <= 0.05 and abs(self.data.qvel[2]) <= 0.05
                and self._tilt() <= self.TOUCHDOWN_TILT_LIMIT):
            self.stable_contact_seconds += self.model.opt.timestep
        else:
            self.stable_contact_seconds = 0.0
        if self.stable_contact_seconds + 1e-9 >= self.STABLE_CONTACT_SECONDS:
            self._success = True
            self._done = True

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset before stepping a completed mission")
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("action must contain three finite normalized xyz velocities")
        action = np.clip(action, -1, 1)
        action[:2] /= max(1.0, float(np.linalg.norm(action[:2])))
        before_distance = float(np.linalg.norm(self._target() - self.data.qpos[:3]))
        for _ in range(self.frame_skip):
            # Contacts exposed by mj_step are evaluated before integration. Save
            # velocities BEFORE stepping; solver impulses must not erase impact.
            before_velocity = self.data.qvel[:3].copy()
            before_tilt = self._tilt()
            if self.phase == "settling":
                self.data.ctrl[:] = 0.0
            else:
                self.base._control([*(action * self.SPEED), 0.0])
            mujoco.mj_step(self.model, self.data)
            self._process_contacts(self._drone_contacts(), before_velocity, before_tilt)
            self.min_clearance = min(self.min_clearance, self._clearance())
            if self._done:
                break
            if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
                self._fail("unstable")
                break
            if abs(self.data.qpos[0]) > 4.2 or abs(self.data.qpos[1]) > 3.1 or self.data.qpos[2] > 3 or self.data.qpos[2] < -0.1:
                self._fail("out_of_bounds")
                break
        mujoco.mj_forward(self.model, self.data)
        self.episode_steps += 1
        after_distance = float(np.linalg.norm(self._target() - self.data.qpos[:3]))
        reward = 5.0 * (before_distance - after_distance) - 0.02
        reward -= 0.004 * float(np.sum((action - self.previous_action) ** 2))
        reward -= 0.08 * max(0, (0.25 - self._clearance()) / 0.25)
        # Phase transition is a task state machine, not a hidden motion teacher.
        if not self._done and self.phase == "approach":
            horizontal_error = np.linalg.norm(self.goal[:2] - self.data.qpos[:2])
            if horizontal_error < 0.10 and np.linalg.norm(self.data.qvel[:2]) < 0.12:
                if self.landing_enabled:
                    self.phase = "descend"
                    reward += 5.0
                elif abs(self.data.qpos[2] - self.ALTITUDE) < 0.15:
                    self._success = True
                    self._done = True
        if self._success:
            reward += 30.0
        elif self._done:
            reward -= 20.0
        terminated = bool(self._done)
        truncated = bool(self.episode_steps >= self.max_episode_steps and not terminated)
        if truncated:
            self.failure_reason = "timeout"
            self._done = True
        self.previous_action = action.copy()
        return self._observation(), float(reward), terminated, truncated, self._info()

    def _info(self):
        return {"is_success": bool(self._success), "collision": bool(self._collision),
                "failure_reason": self.failure_reason, "position": self.data.qpos[:3].copy(),
                "goal": self.goal.copy(), "distance_to_goal": float(np.linalg.norm(self.goal - self.data.qpos[:3])),
                "min_clearance": float(self.min_clearance), "phase": self.phase,
                "touchdown_speed": self.touchdown_speed,
                "touchdown_horizontal_speed": self.touchdown_horizontal_speed,
                "touchdown_tilt_deg": self.touchdown_tilt_deg,
                "stable_contact_seconds": float(self.stable_contact_seconds),
                "episode_steps": self.episode_steps, "time": float(self.data.time),
                "task": self.task, "obstacle_count": self.obstacle_count,
                "observation_mode": "privileged simulator geometry/state; no Flyvis or 3DGS",
                "obstacles": self.obstacles[:self.obstacle_count].copy()}

    def _segment_clear(self, start, end, inflation=0.25):
        delta = np.asarray(end) - start
        den = max(float(delta @ delta), 1e-12)
        for x, y, radius in self.obstacles[:self.obstacle_count]:
            centre = np.array([x, y])
            t = np.clip(float((centre - start) @ delta) / den, 0, 1)
            if np.linalg.norm(centre - (start + t * delta)) < radius + inflation:
                return False
        return True

    def _plan_path(self, start, goal):
        """A* over a 0.15 m grid, followed by visibility-based path shortening."""
        start, goal = np.asarray(start), np.asarray(goal)
        if self._segment_clear(start, goal):
            return [goal.copy()]
        spacing = 0.15
        lo = np.array([-3.3, -2.55])
        shape = (45, 35)
        to_index = lambda p: tuple(np.clip(np.rint((p - lo) / spacing).astype(int), [0, 0], np.array(shape) - 1))
        to_world = lambda i: lo + np.asarray(i) * spacing
        first, last = to_index(start), to_index(goal)
        queue = [(0.0, first)]
        costs = {first: 0.0}
        parents = {}
        found = False
        while queue:
            _, node = heapq.heappop(queue)
            if node == last:
                found = True
                break
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                neighbour = (node[0] + dx, node[1] + dy)
                if not (0 <= neighbour[0] < shape[0] and 0 <= neighbour[1] < shape[1]):
                    continue
                if not self._segment_clear(to_world(node), to_world(neighbour)):
                    continue
                new_cost = costs[node] + math.hypot(dx, dy)
                if new_cost >= costs.get(neighbour, math.inf):
                    continue
                costs[neighbour] = new_cost
                parents[neighbour] = node
                priority = new_cost + np.linalg.norm(np.asarray(last) - neighbour)
                heapq.heappush(queue, (priority, neighbour))
        if not found:
            raise RuntimeError("No connected inflated obstacle path")
        path = [goal.copy()]
        current = last
        while current != first:
            path.append(to_world(current))
            current = parents[current]
        path.append(start.copy())
        path.reverse()
        shortened = []
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self._segment_clear(path[i], path[j]):
                j -= 1
            shortened.append(path[j])
            i = j
        return shortened

    def expert_action(self):
        """Privileged A* waypoint / soft-descent teacher; absent from step()."""
        if self.phase == "settling":
            return np.zeros(3, dtype=np.float32)
        position = self.data.qpos[:3]
        if self.phase == "approach":
            while self._path_index < len(self._path) - 1:
                if np.linalg.norm(self._path[self._path_index] - position[:2]) > 0.16:
                    break
                self._path_index += 1
            delta = self._path[self._path_index] - position[:2]
            velocity_xy = 1.6 * delta - 0.35 * self.data.qvel[:2]
            speed = min(0.55, max(0.15, np.linalg.norm(delta)))
            velocity_xy *= min(1, speed / max(np.linalg.norm(velocity_xy), 1e-9))
            velocity_z = np.clip(1.5 * (self.ALTITUDE - position[2]), -0.3, 0.3)
        else:
            velocity_xy = 1.4 * (self.goal[:2] - position[:2]) - 0.4 * self.data.qvel[:2]
            # Continue slightly below the exact support height so actual contact
            # occurs; a height-only termination would incorrectly count a hover.
            velocity_z = np.clip(1.2 * (self.goal[2] - 0.03 - position[2]), -0.16, 0.1)
        return np.clip(np.array([*velocity_xy, velocity_z]) / self.SPEED, -1, 1).astype(np.float32)

    def render(self, camera="fpv"):
        if camera not in {"fpv", "overview", "down"}:
            raise ValueError("camera must be fpv, overview, or down")
        return self.base.render(camera)

    def close(self):
        self.base.close()
