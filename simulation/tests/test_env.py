"""Meaningful deterministic physics and Gymnasium-interface smoke checks.

Run: .venv/bin/python -m unittest discover -s tests -v
Set FRUITFLY_RENDER_TEST=1 to also exercise the native OpenGL camera.
"""
import os
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fruitfly_sim import FruitFlyDroneEnv, waypoint_velocity


class DroneEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = FruitFlyDroneEnv(episode_seconds=12)
        self.addCleanup(self.env.close)

    def test_hover_and_api(self):
        obs, _ = self.env.reset(seed=7)
        self.assertTrue(self.env.observation_space.contains(obs))
        for _ in range(150):
            obs, reward, terminated, truncated, info = self.env.step(np.zeros(4))
            self.assertFalse(terminated or truncated)
        np.testing.assert_allclose(obs["state"][:3], [0, 0, 1], atol=0.01)
        self.assertAlmostEqual(info["body_thrust_N"], self.env.mass * 9.81, places=5)

    def test_waypoint_and_return(self):
        obs, _ = self.env.reset(seed=0)
        for goal in ([1, 0, 1.2], [0, 0, 1]):
            self.env.set_goal(goal)
            reached = False
            for _ in range(250):
                obs, _, terminated, truncated, info = self.env.step(waypoint_velocity(obs["state"], goal))
                self.assertFalse(terminated or truncated)
                if np.linalg.norm(obs["state"][:3] - goal) < 0.13:
                    reached = True
                    break
            self.assertTrue(reached, f"Failed to reach {goal}: {obs['state'][:3]}")

    def test_seed_and_time_limit(self):
        first, _ = self.env.reset(seed=9, options={"position_jitter": 0.03})
        second, _ = self.env.reset(seed=9, options={"position_jitter": 0.03})
        np.testing.assert_array_equal(first["state"], second["state"])
        self.env.max_steps = 2
        self.env.step(np.zeros(4))
        _, _, terminated, truncated, _ = self.env.step(np.zeros(4))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        with self.assertRaises(RuntimeError):
            self.env.step(np.zeros(4))

    def test_collision_terminates(self):
        self.env.reset(options={"position": [5.5, 2.0, 1.0]})
        _, _, terminated, _, info = self.env.step(np.zeros(4))
        self.assertTrue(terminated)
        self.assertTrue(info["collision"])

    @unittest.skipUnless(os.environ.get("FRUITFLY_RENDER_TEST") == "1", "explicit OpenGL render check")
    def test_fpv_camera(self):
        self.env.reset()
        first = self.env.render()
        self.assertEqual(first.shape, (240, 320, 3))
        self.assertEqual(first.dtype, np.uint8)
        self.assertGreater(float(first.std()), 10)
        self.env.reset(options={"yaw": 1.5})
        second = self.env.render()
        self.assertGreater(np.mean(np.abs(first.astype(float) - second.astype(float))), 2)


if __name__ == "__main__":
    unittest.main()
