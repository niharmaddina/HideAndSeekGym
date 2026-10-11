"""HideAndSeekEnv: a minimal 3D hide-and-seek environment following the Gymnasium API.

The agent controls the hider, a point-like body that moves in fixed increments inside a cubic stage.
The seeker is placed at random and does not move. Each step the hider earns its distance from the
seeker divided by the stage size (so roughly 0 to 1.7 per step); walking into the seeker ends the
episode with a penalty of ``catch_penalty``; otherwise the episode is truncated after ``max_steps``.

Two interchangeable backends hold the bodies:

* ``backend="numpy"`` (default): positions are plain arrays. No dependencies beyond NumPy, fast enough
  for training (hundreds of thousands of steps per minute).
* ``backend="pybullet"``: the same logic with the bodies loaded into PyBullet, so ``render_mode="human"``
  shows the scene in the PyBullet GUI. Requires ``pip install pybullet``.

Physics is not used by either backend: positions are set directly, which is what the task needs.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# action index -> (axis, direction)
_MOVES = {0: (1, +1), 1: (1, -1), 2: (0, -1), 3: (0, +1), 4: (2, +1), 5: (2, -1)}


class _NumpyWorld:
    def __init__(self):
        self.hider = np.zeros(3)
        self.seeker = np.zeros(3)

    def reset(self, hider: np.ndarray, seeker: np.ndarray) -> None:
        self.hider, self.seeker = hider.copy(), seeker.copy()

    def move_hider(self, pos: np.ndarray) -> None:
        self.hider = pos.copy()

    def positions(self) -> tuple[np.ndarray, np.ndarray]:
        return self.hider, self.seeker

    def close(self) -> None:
        pass


class _PyBulletWorld:
    def __init__(self, gui: bool):
        import pybullet as p
        import pybullet_data
        self.p = p
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        self.hider = self.seeker = None

    def reset(self, hider: np.ndarray, seeker: np.ndarray) -> None:
        p = self.p
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.hider = p.loadURDF("r2d2.urdf", hider.tolist(), physicsClientId=self.client)
        self.seeker = p.loadURDF("r2d2.urdf", seeker.tolist(), physicsClientId=self.client)

    def move_hider(self, pos: np.ndarray) -> None:
        self.p.resetBasePositionAndOrientation(self.hider, pos.tolist(), [0, 0, 0, 1], physicsClientId=self.client)
        self.p.stepSimulation(physicsClientId=self.client)

    def positions(self) -> tuple[np.ndarray, np.ndarray]:
        h, _ = self.p.getBasePositionAndOrientation(self.hider, physicsClientId=self.client)
        s, _ = self.p.getBasePositionAndOrientation(self.seeker, physicsClientId=self.client)
        return np.asarray(h), np.asarray(s)

    def close(self) -> None:
        if self.p.isConnected(self.client):
            self.p.disconnect(self.client)


class HideAndSeekEnv(gym.Env):
    """Observation: hider xyz then seeker xyz, 6 floats in [0, stage_size].
    Actions: 0 forward (+y), 1 backward (-y), 2 left (-x), 3 right (+x), 4 up (+z), 5 down (-z), one ``increment`` each.
    Reward: Euclidean distance between hider and seeker after the move, divided by stage_size;
    minus ``catch_penalty`` on the step the hider is caught (distance < catch_radius).

    The 2024 version of this file used ``reward = -distance``, which trains the hider to walk INTO the
    seeker (the least negative return is the shortest episode). The sign was fixed in October 2026."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, stage_size: float = 10.0, increment: float = 0.1, catch_radius: float = 1.0,
                 max_steps: int = 500, catch_penalty: float = 10.0, backend: str = "numpy",
                 render_mode: str | None = None):
        super().__init__()
        if backend not in ("numpy", "pybullet"):
            raise ValueError("backend must be 'numpy' or 'pybullet'")
        if render_mode == "human" and backend != "pybullet":
            raise ValueError("render_mode='human' needs backend='pybullet'")
        self.stage_size = float(stage_size)
        self.increment = float(increment)
        self.catch_radius = float(catch_radius)
        self.max_steps = int(max_steps)
        self.catch_penalty = float(catch_penalty)
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(_MOVES))
        self.observation_space = spaces.Box(low=0.0, high=self.stage_size, shape=(6,), dtype=np.float32)
        self._world = _PyBulletWorld(gui=render_mode == "human") if backend == "pybullet" else _NumpyWorld()
        self._hider_pos = np.zeros(3)
        self._steps = 0

    # ------------------------------------------------------------------ gym API
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._hider_pos = self._random_position()
        seeker_pos = self._random_position()
        while np.linalg.norm(self._hider_pos - seeker_pos) < self.catch_radius:
            seeker_pos = self._random_position()
        self._world.reset(self._hider_pos, seeker_pos)
        self._steps = 0
        return self._observation(), {"distance": self._distance()}

    def step(self, action: int):
        axis, direction = _MOVES[int(action)]
        self._hider_pos[axis] += direction * self.increment
        self._hider_pos = np.clip(self._hider_pos, 0.0, self.stage_size)
        self._world.move_hider(self._hider_pos)
        self._steps += 1
        distance = self._distance()
        terminated = bool(distance < self.catch_radius)
        truncated = self._steps >= self.max_steps and not terminated
        reward = distance / self.stage_size - (self.catch_penalty if terminated else 0.0)
        return self._observation(), float(reward), terminated, truncated, {"distance": distance}

    def render(self):
        return None  # the PyBullet GUI client draws itself when render_mode == "human"

    def close(self):
        self._world.close()

    # ------------------------------------------------------------------ helpers
    def _random_position(self) -> np.ndarray:
        return self.np_random.uniform(0.0, self.stage_size, size=3)

    def _observation(self) -> np.ndarray:
        hider, seeker = self._world.positions()
        return np.concatenate([hider, seeker]).astype(np.float32)

    def _distance(self) -> float:
        hider, seeker = self._world.positions()
        return float(np.linalg.norm(hider - seeker))
