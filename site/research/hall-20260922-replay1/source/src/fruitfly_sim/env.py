"""Gymnasium interface around the official Menagerie Crazyflie rigid-body model.

The action is WORLD-frame [vx, vy, vz, yaw_rate], in m/s and rad/s. A built-in
geometric PD controller converts it to collective thrust (N) and body moments
(N m). This is a convenient navigation baseline, not a calibrated racing-drone
motor/aerodynamic simulation. Its controller and simplified scene are custom.

Observations contain simulation ground truth in ``state``. Optional ``rgb`` is
the actual body-mounted camera image. Ground truth is for baseline/evaluation;
do not give it to a policy when claiming vision-only navigation. Neither Flyvis
nor Gaussian Splatting is part of this baseline.
"""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "bitcraze_crazyflie_2" / "cf2.xml"


def _numbers(values) -> str:
    return " ".join(str(float(x)) for x in values)


def _look_at(position, target) -> str:
    direction = np.asarray(target, float) - np.asarray(position, float)
    direction /= np.linalg.norm(direction)
    right = np.cross(direction, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, direction)
    return _numbers(np.concatenate((right, up)))


def _scene_xml(model_path: Path, width: int, height: int) -> str:
    """Adapt XML in memory, leaving the downloaded upstream model untouched."""
    root = ET.parse(model_path).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str(model_path.parent / "assets"))
    option = root.find("option")
    option.set("timestep", "0.002")
    option.set("gravity", "0 0 -9.81")
    # Disable the uncalibrated mesh-fluid model for a transparent rigid-body
    # baseline. Wind, rotor dynamics and aerodynamic identification are future work.
    option.set("density", "0")
    option.set("viscosity", "0")
    keyframes = root.find("keyframe")
    if keyframes is not None:
        root.remove(keyframes)
    # Menagerie's moment gears are -1e-5 and ctrlrange is explicitly arbitrary.
    # Use SI moments directly and document this prototype's controller limits.
    actuators = root.find("actuator")
    for index, axis in enumerate("xyz"):
        actuator = actuators.find(f"motor[@name='{axis}_moment']")
        gear = [0] * 6
        gear[3 + index] = 1
        actuator.set("gear", _numbers(gear))
        actuator.set("ctrlrange", "-0.002 0.002")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(width), offheight=str(height))
    ET.SubElement(visual, "headlight", diffuse="0.65 0.65 0.65", ambient="0.3 0.3 0.3")
    ET.SubElement(visual, "rgba", haze="0.65 0.78 0.9 1")
    assets = root.find("asset")
    ET.SubElement(assets, "texture", type="skybox", builtin="gradient",
                  rgb1="0.28 0.50 0.74", rgb2="0.8 0.89 0.97", width="512", height="2048")
    ET.SubElement(assets, "texture", name="ground_tex", type="2d", builtin="checker",
                  rgb1="0.31 0.42 0.26", rgb2="0.38 0.49 0.31", width="512", height="512")
    ET.SubElement(assets, "material", name="ground_mat", texture="ground_tex",
                  texrepeat="24 24", texuniform="true", reflectance="0.02")
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="2 -3 10", dir="-0.2 0.3 -1", directional="true")
    ET.SubElement(world, "geom", name="ground", type="plane", size="20 20 0.1",
                  material="ground_mat")
    ET.SubElement(world, "camera", name="overview", pos="7 -8 7",
                  xyaxes=_look_at([7, -8, 7], [1.0, 1.0, 0.9]), fovy="48")
    drone = world.find("body[@name='cf2']")
    angle = math.radians(8)
    ET.SubElement(drone, "camera", name="fpv", pos="0.055 0 0.025",
                  xyaxes=_numbers([0, -1, 0, math.sin(angle), 0, math.cos(angle)]), fovy="85")
    # A visible but non-colliding home marker and route landmarks.
    ET.SubElement(world, "geom", name="home_marker", type="cylinder", size="0.32 0.008",
                  pos="0 0 0.009", rgba="0.95 0.65 0.15 1", contype="0", conaffinity="0")
    for name, pos, size, color in [
        ("red_building", [5.5, 2.0, 1.0], [0.8, 1.0, 1.0], [0.65, 0.24, 0.18, 1]),
        ("blue_building", [1.3, 5.0, 0.7], [1.0, 0.5, 0.7], [0.20, 0.39, 0.58, 1]),
        ("white_building", [-3.0, 3.0, 1.3], [0.6, 1.0, 1.3], [0.82, 0.78, 0.63, 1]),
        ("small_obstacle", [1.5, 1.5, 0.35], [0.3, 0.3, 0.35], [0.80, 0.52, 0.20, 1]),
    ]:
        ET.SubElement(world, "geom", name=name, type="box", pos=_numbers(pos),
                      size=_numbers(size), rgba=_numbers(color))
    for index, (x, y) in enumerate([(-2, -2), (5, -2), (6, 5), (-3, 6), (-5, 0)]):
        ET.SubElement(world, "geom", name=f"tree_trunk_{index}", type="cylinder",
                      pos=f"{x} {y} 0.6", size="0.09 0.6", rgba="0.35 0.23 0.12 1")
        ET.SubElement(world, "geom", name=f"tree_crown_{index}", type="ellipsoid",
                      pos=f"{x} {y} 1.6", size="0.65 0.65 0.9", rgba="0.16 0.37 0.16 1")
    return ET.tostring(root, encoding="unicode")


class FruitFlyDroneEnv(gym.Env):
    """Navigation actions over a MuJoCo Crazyflie with an optional RGB camera.

    ``state`` order: position xyz (world m), quaternion wxyz (body-to-world),
    linear velocity xyz (world m/s), angular velocity xyz (body rad/s).
    ``goal`` is a world-frame target used by the baseline reward only.

    Action limits: vx/vy ±2 m/s, vz ±1 m/s, yaw rate ±1.5 rad/s. Commands are
    clipped to these limits. A step is 20 ms (50 Hz); physics runs at 500 Hz.
    Reward is negative target distance times dt. Collision/escape terminates;
    the configured episode time truncates. Reaching a goal is reported in info
    but does not terminate, allowing a multi-waypoint route in one episode.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, model_path=None, *, rgb_observation=False, width=320,
                 height=240, episode_seconds=30.0, render_mode=None):
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'; use the demo's --viewer for GUI")
        if width < 16 or height < 16 or episode_seconds <= 0:
            raise ValueError("Image dimensions must be >=16 and episode_seconds must be positive")
        self.model_path = Path(model_path or DEFAULT_MODEL).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Missing official Crazyflie model: {self.model_path}")
        self.width, self.height = int(width), int(height)
        self.rgb_observation = bool(rgb_observation)
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_string(_scene_xml(self.model_path, self.width, self.height))
        self.data = mujoco.MjData(self.model)
        self.body_id = self.model.body("cf2").id
        self.mass = float(self.model.body_mass[self.body_id])
        self.inertia = self.model.body_inertia[self.body_id].copy()
        self.frame_skip = 10
        self.dt = self.model.opt.timestep * self.frame_skip
        self.max_steps = math.ceil(episode_seconds / self.dt)
        self.action_space = spaces.Box(np.array([-2, -2, -1, -1.5], dtype=np.float32),
                                       np.array([2, 2, 1, 1.5], dtype=np.float32))
        observations = {
            "state": spaces.Box(-np.inf, np.inf, shape=(13,), dtype=np.float32),
            "goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        }
        if self.rgb_observation:
            observations["rgb"] = spaces.Box(0, 255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict(observations)
        self.goal = np.array([0, 0, 1.0], dtype=float)
        self._renderer = None
        self._steps = 0
        self._finished = True
        self._desired_yaw = 0.0

    def set_goal(self, goal):
        goal = np.asarray(goal, dtype=float)
        if goal.shape != (3,) or not np.all(np.isfinite(goal)):
            raise ValueError("goal must contain three finite world coordinates")
        self.goal = goal.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        position = np.asarray(options.get("position", [0, 0, 1.0]), dtype=float)
        yaw = float(options.get("yaw", 0))
        if position.shape != (3,) or not np.all(np.isfinite(position)) or not np.isfinite(yaw):
            raise ValueError("reset position and yaw must be finite")
        jitter = float(options.get("position_jitter", 0.0))
        if not np.isfinite(jitter) or jitter < 0:
            raise ValueError("position_jitter must be finite and nonnegative")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = position + self.np_random.uniform(-jitter, jitter, size=3)
        self.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.data.ctrl[0] = self.mass * 9.81
        self._desired_yaw = yaw
        self._steps = 0
        self._finished = False
        self.set_goal(options.get("goal", [0, 0, 1.0]))
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), self._info()

    def _control(self, action):
        rotation = self.data.xmat[self.body_id].reshape(3, 3)
        acceleration = np.clip(2.5 * (action[:3] - self.data.qvel[:3]), -2.0, 2.0)
        force_world = self.mass * (acceleration + [0, 0, 9.81])
        desired_z = force_world / np.linalg.norm(force_world)
        heading = np.array([math.cos(self._desired_yaw), math.sin(self._desired_yaw), 0.0])
        desired_y = np.cross(desired_z, heading)
        desired_y /= np.linalg.norm(desired_y)
        desired_x = np.cross(desired_y, desired_z)
        desired_rotation = np.column_stack([desired_x, desired_y, desired_z])
        skew_error = desired_rotation.T @ rotation - rotation.T @ desired_rotation
        attitude_error = 0.5 * np.array([skew_error[2, 1], skew_error[0, 2], skew_error[1, 0]])
        omega = self.data.qvel[3:6]
        target_omega = rotation.T @ np.array([0, 0, action[3]])
        moment = -0.002 * attitude_error - 0.0005 * (omega - target_omega)
        moment += np.cross(omega, self.inertia * omega)
        self.data.ctrl[0] = np.clip(force_world @ rotation[:, 2], 0, 0.35)
        self.data.ctrl[1:4] = np.clip(moment, -0.002, 0.002)

    def step(self, action):
        if self._finished:
            raise RuntimeError("Call reset() before stepping a new or completed episode")
        action = np.asarray(action, dtype=float)
        if action.shape != (4,) or not np.all(np.isfinite(action)):
            raise ValueError("action must contain four finite values: vx, vy, vz, yaw_rate")
        command = np.clip(action, self.action_space.low, self.action_space.high)
        collided = False
        for _ in range(self.frame_skip):
            self._desired_yaw += command[3] * self.model.opt.timestep
            self._control(command)
            mujoco.mj_step(self.model, self.data)
            if self._has_collision():
                collided = True
                break
        self._steps += 1
        # Refresh Cartesian state/camera poses at the new integration state.
        mujoco.mj_forward(self.model, self.data)
        position = self.data.qpos[:3]
        unstable = not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel))
        escaped = np.linalg.norm(position[:2]) > 15 or position[2] > 8 or position[2] < 0.03
        terminated = bool(collided or unstable or escaped)
        truncated = bool(self._steps >= self.max_steps and not terminated)
        self._finished = terminated or truncated
        info = self._info()
        info.update(collision=collided, out_of_bounds=bool(escaped), unstable=unstable,
                    applied_action=command.astype(np.float32))
        reward = -info["distance_to_goal"] * self.dt - (10.0 if terminated else 0.0)
        return self._observation(), float(reward), terminated, truncated, info

    def _has_collision(self):
        for contact in self.data.contact:
            bodies = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if (bodies[0] == self.body_id) != (bodies[1] == self.body_id):
                return True
        return False

    def _info(self):
        distance = float(np.linalg.norm(self.goal - self.data.qpos[:3]))
        return {"time": float(self.data.time), "distance_to_goal": distance,
                "goal_reached": distance < 0.15, "collision": self._has_collision(),
                "body_thrust_N": float(self.data.ctrl[0]),
                "body_moment_Nm": self.data.ctrl[1:4].copy()}

    def _observation(self):
        observation = {"state": np.concatenate((self.data.qpos[:7], self.data.qvel[:6])).astype(np.float32),
                       "goal": self.goal.astype(np.float32).copy()}
        if self.rgb_observation:
            observation["rgb"] = self.render()
        return observation

    def render(self, camera="fpv"):
        """Return a copy of an RGB frame from ``fpv``, ``overview`` or ``track``."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def waypoint_velocity(state, goal, *, speed=0.75, face_direction=True):
    """Ground-truth waypoint baseline; not visual path planning or learning."""
    state = np.asarray(state)
    displacement = np.asarray(goal) - state[:3]
    velocity = 1.3 * displacement
    norm = np.linalg.norm(velocity)
    if norm > speed:
        velocity *= speed / norm
    yaw_rate = 0.0
    if face_direction and np.linalg.norm(displacement[:2]) > 0.2:
        w, x, y, z = state[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        target_yaw = math.atan2(displacement[1], displacement[0])
        error = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
        yaw_rate = np.clip(1.5 * error, -1.0, 1.0)
    return np.asarray([*velocity, yaw_rate], dtype=np.float32)
