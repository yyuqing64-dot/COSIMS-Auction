from __future__ import annotations

import numpy as np

from sim.task import Task
from sim.env import Env


class RoundRobin:
    name = "rr"

    def __init__(self, config: dict, rng: np.random.Generator):
        self.n = int(config["n_nodes"])
        self.idx = 0

    def select_dest(self, task: Task, env: Env) -> int:
        """Cycle through destinations 0..n-1 regardless of load."""
        dest = self.idx
        self.idx = (self.idx + 1) % self.n
        return dest

