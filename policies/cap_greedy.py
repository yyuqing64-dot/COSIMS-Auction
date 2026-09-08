from __future__ import annotations
from typing import Optional
import numpy as np
from sim.task import Task
from sim.env import Env

class CapAwareGreedy:
    name = "cap_greedy"

    def __init__(self, config, rng):
        self.cloud_enabled = bool(config["cloud"]["enabled"])

    def select_dest(self, task: Task, env: Env) -> Optional[int]:
        # choose destination with minimal estimated finish time
        n = env.n
        best_dest = 0
        best_finish = float("inf")

        for j in range(n):
            f = env.est_finish_time_edge(task, j)
            if f < best_finish:
                best_finish = f
                best_dest = j

        if self.cloud_enabled:
            cloud_finish = env.est_finish_time_cloud(task)
            if cloud_finish < best_finish:
                return None

        return int(best_dest)
