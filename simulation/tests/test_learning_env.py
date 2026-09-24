"""Small physical checks for the training curriculum, not training-score tests."""
import numpy as np
import pytest
import mujoco
from gymnasium.utils.env_checker import check_env
from fruitfly_sim.learning_env import NavigationLearningEnv


def test_gym_interface_and_seed_variation():
    env = NavigationLearningEnv("reach")
    try:
        check_env(env, skip_render_check=True)
        first, _ = env.reset(seed=15)
        again, _ = env.reset(seed=15)
        other, _ = env.reset(seed=16)
        np.testing.assert_array_equal(first, again)
        assert not np.array_equal(first, other)
        assert first.shape == (24,) and first.dtype == np.float32
    finally:
        env.close()


@pytest.mark.parametrize("stage", ["reach", "avoid", "return"])
def test_expert_reaches_tasks(stage):
    env = NavigationLearningEnv(stage)
    try:
        env.reset(seed=7)
        phases = set()
        for _ in range(env.max_episode_steps):
            _, _, terminated, truncated, info = env.step(env.expert_action())
            phases.add(info["phase"])
            if terminated or truncated:
                break
        assert info["is_success"], info
        assert not info["collision"]
        assert abs(env.data.qpos[2] - env.ALTITUDE) < 0.05
        if stage == "return":
            assert phases == {0, 1}
        with pytest.raises(RuntimeError):
            env.step(np.zeros(2))
    finally:
        env.close()


def test_obstacle_has_real_collision_geometry():
    env = NavigationLearningEnv("avoid")
    try:
        env.reset(seed=3)
        env.data.qpos[:2] = env.obstacle_center + [env.OBSTACLE_RADIUS, 0]
        mujoco.mj_forward(env.model, env.data)
        _, reward, terminated, _, info = env.step(np.zeros(2))
        assert terminated and info["collision"] and not info["is_success"]
        assert reward < 0
    finally:
        env.close()
