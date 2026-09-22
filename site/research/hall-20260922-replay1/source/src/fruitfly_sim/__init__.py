"""Small, native-MuJoCo flight simulation for visual navigation experiments."""

from .env import FruitFlyDroneEnv, waypoint_velocity

__all__ = ["FruitFlyDroneEnv", "waypoint_velocity"]
