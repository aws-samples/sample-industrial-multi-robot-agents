"""Base driver interface for robot execution backends."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseDriver(ABC):
    """Abstract robot driver. Implement for each execution backend."""

    @abstractmethod
    def execute_trajectory(
        self,
        trajectory: list[dict],
        joint_names: list[str],
        velocity_scale: float = 1.0,
    ) -> dict:
        """
        Execute a precomputed trajectory.

        Args:
            trajectory:     [{"positions": [j1..j6], "time_s": float}]
            joint_names:    ROS 2 joint name list
            velocity_scale: Speed scaling (0.1–1.0)

        Returns:
            {"success": bool, "elapsed_s": float, "error": str|None}
        """

    @abstractmethod
    def go_home(self) -> dict:
        """Move robot to home configuration."""

    @abstractmethod
    def get_joint_positions(self) -> list[float]:
        """Return current joint positions [j1..j6]."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Driver identifier string."""
