"""Physical mission gates: these test the environment, not learned policies."""
import math

import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from fruitfly_sim.mission_env import MissionEnv


def run_teacher(env, seed):
    env.reset(seed=seed)
    for _ in range(env.max_episode_steps):
        _, _, terminated, truncated, info = env.step(env.expert_action())
        if terminated or truncated:
            return info
    pytest.fail("Mission did not terminate or truncate")


def test_gym_interface_seed_and_camera_configuration():
    env = MissionEnv("obstacles", 8)
    try:
        check_env(env, skip_render_check=True)
        first, _ = env.reset(seed=121)
        layout = env.obstacles.copy()
        again, _ = env.reset(seed=121)
        np.testing.assert_array_equal(first, again)
        np.testing.assert_array_equal(layout, env.obstacles)
        other, _ = env.reset(seed=122)
        assert not np.array_equal(first, other)
        assert first.shape == (52,) and first.dtype == np.float32
        assert env.action_space.shape == (3,)
        assert env.model.camera("down").id >= 0
    finally:
        env.close()


@pytest.mark.parametrize("task,count", [("landing", 3)] + [(task, count)
    for task in ("obstacles", "obstacle_landing") for count in (3, 5, 8)])
def test_five_distinct_layouts_have_physical_teacher_solution(task, count):
    env = MissionEnv(task, count)
    try:
        for seed in (3, 17, 41, 106, 251):
            info = run_teacher(env, seed)
            assert info["is_success"], (task, count, seed, info)
            assert not info["collision"]
            assert info["min_clearance"] > 0
            if task != "obstacles":
                assert info["phase"] == "settling"
                assert info["stable_contact_seconds"] >= 0.5
                assert info["touchdown_speed"] <= 0.2
                assert info["touchdown_horizontal_speed"] <= 0.15
                assert info["touchdown_tilt_deg"] <= 10
                assert env._drone_contacts(), "Height proximity alone is not landing"
                np.testing.assert_array_equal(env.data.ctrl, 0)
            with pytest.raises(RuntimeError):
                env.step(np.zeros(3))
    finally:
        env.close()


def make_landing(position, velocity=(0, 0, 0), tilt=0):
    env = MissionEnv("landing")
    env.reset(seed=2, options={"goal_xy": [0, 0], "position": position, "phase": "descend"})
    env.data.qvel[:3] = velocity
    env.data.qpos[3:7] = [math.cos(math.radians(tilt) / 2), 0, math.sin(math.radians(tilt) / 2), 0]
    mujoco.mj_forward(env.model, env.data)
    return env


@pytest.mark.parametrize("velocity,tilt", [((0, 0, -0.8), 0), ((0.5, 0, -0.1), 0), ((0, 0, -0.1), 20)])
def test_unsafe_touchdown_uses_pre_impact_state(velocity, tilt):
    env = make_landing([0, 0, 0.118 if not tilt else 0.13], velocity, tilt)
    try:
        for _ in range(10):
            _, _, terminated, truncated, info = env.step([0, 0, -0.4])
            if terminated or truncated:
                break
        assert terminated and info["collision"] and not info["is_success"], info
        assert info["failure_reason"] == "unsafe_touchdown", info
        assert (info["touchdown_speed"] > 0.2 or info["touchdown_horizontal_speed"] > 0.15
                or info["touchdown_tilt_deg"] > 10)
    finally:
        env.close()


def test_pad_side_and_ground_contact_are_not_landing():
    for position in ([0.59, 0, 0.09], [0.50, 0, 0.112], [1, 0, 0.012]):
        env = make_landing(position, (0, 0, -0.1))
        try:
            _, _, terminated, _, info = env.step([0, 0, 0])
            assert terminated and info["collision"] and not info["is_success"], info
            assert info["failure_reason"] in {"pad_edge_or_side_contact", "obstacle_or_ground_contact"}
        finally:
            env.close()


def test_hover_near_pad_is_not_landing():
    env = make_landing([0, 0, 0.18])
    try:
        for _ in range(15):
            _, _, terminated, truncated, info = env.step([0, 0, 0])
            assert not terminated and not truncated
        assert not info["is_success"] and info["touchdown_speed"] is None
        assert info["stable_contact_seconds"] == 0
    finally:
        env.close()


def test_each_active_obstacle_has_physical_collision():
    env = MissionEnv("obstacles", 8)
    try:
        for index in range(8):
            env.reset(seed=88)
            x, y, radius = env.obstacles[index]
            env.data.qpos[:3] = [x + radius + 0.01, y, 1.2]
            mujoco.mj_forward(env.model, env.data)
            _, _, terminated, _, info = env.step([0, 0, 0])
            assert terminated and info["collision"] and not info["is_success"], (index, info)
    finally:
        env.close()


def test_teacher_is_not_called_by_step():
    env = MissionEnv("obstacles", 3)
    try:
        env.reset(seed=55)
        def fail():
            raise AssertionError("Teacher must never be called during policy step")
        env.expert_action = fail
        env.step([0, 0, 0])
    finally:
        env.close()


def test_soft_touchdown_needs_dwell_and_motors_off():
    env = make_landing([0, 0, 0.114], (0, 0, -0.08))
    try:
        _, _, terminated, _, info = env.step([0, 0, -0.2])
        assert info["touchdown_speed"] is not None, info
        assert not terminated and not info["is_success"]
        assert info["stable_contact_seconds"] < 0.5
        np.testing.assert_array_equal(env.data.ctrl, 0)
        for _ in range(10):
            _, _, terminated, _, info = env.step([1, 1, 1])
            if terminated:
                break
        assert info["is_success"] and info["stable_contact_seconds"] >= 0.5, info
        # The explicit landed state cuts power regardless of subsequent commands.
        np.testing.assert_array_equal(env.data.ctrl, 0)
    finally:
        env.close()
