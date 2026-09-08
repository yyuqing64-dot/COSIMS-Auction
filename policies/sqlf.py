from __future__ import annotations

import heapq
from typing import List

import numpy as np

from sim.env import Env
from sim.task import Task


class ShortestQueueFirst:
    name = "sqlf"

    def __init__(self, config: dict, rng: np.random.Generator):
        self.n = int(config["n_nodes"])
        self.allow_cloud = bool(config.get("sqlf", {}).get("allow_cloud", False))
        self._finish_heaps: List[List[float]] = [[] for _ in range(self.n)]
        self._tie_break = np.arange(self.n, dtype=int)

    def _cleanup(self, now: float):
        for h in self._finish_heaps:
            while h and h[0] <= now:
                heapq.heappop(h)

    def select_dest(self, task: Task, env: Env):
        now = float(task.arrival_t)
        self._cleanup(now)
        qlens = np.array([len(h) for h in self._finish_heaps], dtype=int)
        min_q = int(qlens.min())
        cands = self._tie_break[qlens == min_q]
        if cands.size == 1:
            dest = int(cands[0])
        else:
            best_dest = int(cands[0])
            best_finish = float("inf")
            for j in cands.tolist():
                f = float(env.est_finish_time_edge(task, int(j)))
                if f < best_finish:
                    best_finish = f
                    best_dest = int(j)
            dest = best_dest
        edge_finish = float(env.est_finish_time_edge(task, dest))
        if self.allow_cloud and bool(env.cloud_enabled):
            cloud_finish = float(env.est_finish_time_cloud(task))
            if cloud_finish < edge_finish:
                return None
        heapq.heappush(self._finish_heaps[dest], edge_finish)
        return dest
