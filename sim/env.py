from __future__ import annotations
import numpy as np
from typing import List, Optional, Protocol, Any, Dict

from sim.task import Task
from sim.network import Network

class Policy(Protocol):
    name: str
    def select_dest(self, task: Task, env: "Env") -> Optional[int]:
        """Return dest node id [0..n-1], or None meaning cloud."""

class Env:
    def __init__(self, config: Dict[str, Any], rng: np.random.Generator, policy: Policy):
        self.cfg = config
        self.rng = rng
        self.policy = policy

        self.n = int(config["n_nodes"])
        self.sim_time = float(config["sim_time"])
        self.warmup_time = float(config["warmup_time"])

        xi_list = config["capacity"]["xi"]
        assert len(xi_list) == self.n
        self.xi = np.array(xi_list, dtype=float)  # "work units per second"

        net_cfg = config["network"]
        self.net = Network.random_latency_matrix(
            n=self.n,
            rng=self.rng,
            ms_min=float(net_cfg["latency_ms_min"]),
            ms_max=float(net_cfg["latency_ms_max"]),
            diagonal_ms=float(net_cfg.get("diagonal_ms", 1.0)),
            symmetric=bool(net_cfg.get("symmetric", True)),
        )

        cloud_cfg = config["cloud"]
        self.cloud_enabled = bool(cloud_cfg["enabled"])
        self.cloud_latency_s = float(cloud_cfg["cloud_latency_ms"]) / 1000.0
        self.cloud_xi = float(cloud_cfg["cloud_capacity_xi"])  # faster than edge by default
        self.cloud_available_time = 0.0

        # Each edge node has an "available time" (single-server queue)
        self.available_time = np.zeros(self.n, dtype=float)

        # Arrival
        self.base_total_rate = float(config["arrival"]["base_total_rate"])  # tasks/s
        self.bursts = []
        arr_cfg = config.get("arrival", {})
        for b in arr_cfg.get("burst", []):
            try:
                self.bursts.append(
                    (
                        float(b.get("start", 0.0)),
                        float(b.get("end", 0.0)),
                        float(b.get("factor", 1.0)),
                    )
                )
            except Exception:
                pass
        # distribute arrivals across sources (can be skewed)
        self.src_probs = self._default_src_distribution(self.n)

        # Task distribution
        tcfg = config["task"]
        self.cpu_mu = float(tcfg["cpu_logn_mu"])
        self.cpu_sigma = float(tcfg["cpu_logn_sigma"])
        self.data_min = float(tcfg["data_mb_min"])
        self.data_max = float(tcfg["data_mb_max"])

        self.now = 0.0
        self._tid = 0

    @property
    def cost_matrix(self) -> np.ndarray:
        """Expose latency matrix (ms) as generic cost matrix for OT-based policies."""
        return self.net.lat_matrix_ms

    @staticmethod
    def _default_src_distribution(n: int) -> np.ndarray:
        # slightly skewed to create hotspots (you can change later)
        w = np.ones(n, dtype=float)
        w[0] = 3.0
        w[1] = 2.0
        w = w / w.sum()
        return w

    def est_finish_time_edge(self, task: Task, dest: int) -> float:
        net = self.net.latency_s(task.src, dest)
        start = max(task.arrival_t + net, float(self.available_time[dest]))
        exec_t = task.cpu_demand / float(self.xi[dest])
        return start + exec_t

    def est_finish_time_cloud(self, task: Task) -> float:
        start = max(task.arrival_t + self.cloud_latency_s, float(self.cloud_available_time))
        exec_t = task.cpu_demand / float(self.cloud_xi)
        return start + exec_t

    def _sample_task(self, arrival_t: float) -> Task:
        src = int(self.rng.choice(self.n, p=self.src_probs))
        # lognormal "work"
        cpu = float(self.rng.lognormal(mean=self.cpu_mu, sigma=self.cpu_sigma))
        data_mb = float(self.rng.uniform(self.data_min, self.data_max))
        t = Task(
            tid=self._tid,
            src=src,
            arrival_t=arrival_t,
            cpu_demand=cpu,
            data_mb=data_mb,
        )
        self._tid += 1
        return t

    def run(self, load_scale: float) -> List[Task]:
        """
        Generate arrivals as Poisson process with rate = base_total_rate * load_scale.
        For each arrival, schedule immediately (event-driven queueing).
        """
        def rate_factor(now: float) -> float:
            f = 1.0
            for s, e, k in self.bursts:
                if s <= now <= e:
                    f *= k
            return f
        lam_base = self.base_total_rate * float(load_scale)
        tasks: List[Task] = []

        t = 0.0
        while t < self.sim_time:
            # exponential inter-arrival
            lam_now = lam_base * rate_factor(t)
            dt = float(self.rng.exponential(1.0 / max(lam_now, 1e-12)))
            t += dt
            self.now = t
            if t >= self.sim_time:
                break

            task = self._sample_task(arrival_t=t)

            W = np.maximum(np.asarray(self.available_time, dtype=float) - t, 0.0)
            task.backlog_time_total = float(W.sum())
            task.backlog_work_total = float(np.dot(W, self.xi))

            # choose destination
            dest = self.policy.select_dest(task, self)

            if dest is None:
                # cloud
                task.is_cloud = True
                task.dest = None
                start = max(task.arrival_t + self.cloud_latency_s, self.cloud_available_time)
                exec_t = task.cpu_demand / self.cloud_xi
                finish = start + exec_t
                self.cloud_available_time = finish
            else:
                task.is_cloud = False
                task.dest = int(dest)
                net = self.net.latency_s(task.src, task.dest)
                start = max(task.arrival_t + net, float(self.available_time[task.dest]))
                exec_t = task.cpu_demand / float(self.xi[task.dest])
                finish = start + exec_t
                self.available_time[task.dest] = finish

            task.start_t = float(start)
            task.finish_t = float(finish)

            tasks.append(task)

        return tasks
