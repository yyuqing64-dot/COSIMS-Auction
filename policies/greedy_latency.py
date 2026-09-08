from __future__ import annotations
from typing import Optional
from sim.task import Task
from sim.env import Env

class GreedyLatency:
    name = "greedy_latency"

    def __init__(self, config, rng):
        self.cloud_enabled = bool(config["cloud"]["enabled"])

    def select_dest(self, task: Task, env: Env) -> Optional[int]:
        # choose minimal network latency edge node
        lat_row = env.net.lat_ms[task.src]  # ms
        dest = int(lat_row.argmin())

        if self.cloud_enabled:
            # compare to cloud as an option (rough)
            best_edge_finish = env.est_finish_time_edge(task, dest)
            cloud_finish = env.est_finish_time_cloud(task)
            if cloud_finish < best_edge_finish:
                return None

        return dest
