# policies/cosims_sinkhorn.py
from __future__ import annotations
import numpy as np
import time


def _safe_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, eps)
    return x / x.sum()


def sinkhorn_transport(M: np.ndarray,
                       r: np.ndarray,
                       o: np.ndarray,
                       lam: float = 0.1,
                       max_iter: int = 35,
                       eps: float = 1e-12) -> np.ndarray:
    """
    Entropic OT via Sinkhorn matrix scaling.
    M: (n,n) cost matrix (>=0)
    r,o: (n,) marginals, sum to 1
    returns P: (n,n) transport matrix, approx row=r col=o
    """
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    assert M.shape == (n, n)

    r = _safe_normalize(r, eps)
    o = _safe_normalize(o, eps)

    # Kernel
    K = np.exp(-(M / max(lam, eps)))
    K = np.maximum(K, eps)

    u = np.ones(n) / n
    v = np.ones(n) / n

    for _ in range(max_iter):
        u = r / (K @ v + eps)
        v = o / (K.T @ u + eps)

    P = (u[:, None] * K) * v[None, :]
    P = P / (P.sum() + eps)
    return P


class COSIMSSinkhorn:
    """
    COSIMS-style OT offloading baseline:
      - update P every epoch using arrival counts -> r, capacities -> o, cost matrix -> M
      - per task from src=i, sample dest ~ Categorical(P[i,:])
    """
    name = "cosims_sinkhorn"

    def __init__(
        self,
        config: dict,
        rng: np.random.Generator,
        epoch_sec: float | None = None,
        lam: float | None = None,
        max_iter: int | None = None,
    ):
        # hyperparams from config or defaults
        cosims_cfg = config.get("cosims", {})
        self.epoch_sec = float(epoch_sec or cosims_cfg.get("epoch_sec", 1.0))
        self.lam = float(lam or cosims_cfg.get("lam", 0.1))
        self.max_iter = int(max_iter or cosims_cfg.get("max_iter", 35))
        self.ot_demand_mode = str(cosims_cfg.get("ot_demand_mode", "count")).lower()
        self.ot_cost_mode = str(cosims_cfg.get("ot_cost_mode", "prop")).lower()
        self.cost_service_weight = float(cosims_cfg.get("ot_cost_service_weight", 1.0))
        self.cost_queue_weight = float(cosims_cfg.get("ot_cost_queue_weight", 1.0))
        self.rng = rng

        # lazy init from env on first call to select_dest
        self.initialized = False
        self.n = None
        self.o = None
        self.M = None
        self.base_M = None
        self.epoch_start_t = 0.0
        self.counts = None
        self.task_counts = None
        self.cpu_sums = None
        self.P = None
        self._ot_time_sum = 0.0
        self._ot_updates = 0
        self._decision_time_sum = 0.0
        self._decision_calls = 0

    def _init_from_env(self, env):
        self.n = int(getattr(env, "n_nodes", getattr(env, "n", None)))
        if self.n is None:
            raise AttributeError("Env must have n_nodes or n")

        caps = None
        for name in ("capacities", "xi", "edge_capacities", "edge_caps"):
            val = getattr(env, name, None)
            if val is not None:
                caps = val
                break
        if caps is None:
            raise AttributeError("Env must provide capacities vector (capacities/xi/edge_capacities/edge_caps)")
        self.o = _safe_normalize(np.asarray(caps, dtype=float))

        M = getattr(env, "cost_matrix", None)
        if callable(M):
            M = M()
        if M is None:
            M = (
                getattr(env, "M", None)
                or getattr(env, "latency_matrix", None)
                or getattr(env, "lat_matrix", None)
                or getattr(env, "net_lat", None)
                or getattr(env, "cost", None)
            )
        if M is None:
            raise AttributeError("Env must provide cost matrix (cost_matrix()/M/latency_matrix/lat_matrix/net_lat)")
        self.base_M = np.asarray(M, dtype=float)
        self.M = self.base_M.copy()
        assert self.M.shape == (self.n, self.n), f"Cost matrix shape must be (n,n), got {self.M.shape}"

        self.counts = np.zeros(self.n, dtype=float)
        self.task_counts = np.zeros(self.n, dtype=float)
        self.cpu_sums = np.zeros(self.n, dtype=float)
        self.P = np.ones((self.n, self.n), dtype=float) / self.n
        self.initialized = True
        self.epoch_start_t = 0.0

    def _demand_increment(self, task) -> float:
        if self.ot_demand_mode in {"workload", "weighted", "cpu", "cpu_demand"}:
            return float(max(getattr(task, "cpu_demand", 1.0), 1e-12))
        return 1.0

    def _minmax01(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            return np.zeros_like(x, dtype=float)
        return (x - lo) / (hi - lo)

    def _scale_like_base_cost(self, x01: np.ndarray) -> np.ndarray:
        base = np.asarray(self.base_M, dtype=float)
        lo = float(np.nanmin(base))
        hi = float(np.nanmax(base))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            return np.asarray(x01, dtype=float)
        return lo + np.asarray(x01, dtype=float) * (hi - lo)

    def _current_cost_matrix(self, env, now: float) -> np.ndarray:
        if self.ot_cost_mode not in {"comp", "completion", "completion_aware", "resource", "resource_aware"}:
            return self.base_M
        counts = np.asarray(self.task_counts, dtype=float)
        cpu = np.asarray(self.cpu_sums, dtype=float)
        positive = counts > 0
        global_avg = float(cpu[positive].sum() / max(counts[positive].sum(), 1e-12)) if np.any(positive) else 1.0
        avg_cpu = np.where(positive, cpu / np.maximum(counts, 1e-12), global_avg)
        xi = np.asarray(getattr(env, "xi"), dtype=float)
        service = avg_cpu[:, None] / np.maximum(xi[None, :], 1e-12)
        backlog = np.maximum(np.asarray(getattr(env, "available_time"), dtype=float) - now, 0.0)
        backlog_matrix = np.broadcast_to(backlog[None, :], self.base_M.shape)

        combined = (
            self._minmax01(self.base_M)
            + self.cost_service_weight * self._minmax01(service)
            + self.cost_queue_weight * self._minmax01(backlog_matrix)
        )
        return self._scale_like_base_cost(self._minmax01(combined))

    def _maybe_update_epoch(self, now: float, env=None):
        if now - self.epoch_start_t < self.epoch_sec:
            return
        total = float(self.counts.sum())
        if total > 0:
            r = self.counts.astype(float) / total
            t0 = time.perf_counter()
            M = self._current_cost_matrix(env, now) if env is not None else self.base_M
            self.P = sinkhorn_transport(M, r, self.o, lam=self.lam, max_iter=self.max_iter)
            t1 = time.perf_counter()
            self._ot_time_sum += (t1 - t0)
            self._ot_updates += 1
            # numerical safety: ensure rows sum to 1
            row_sums = self.P.sum(axis=1, keepdims=True)
            self.P = self.P / np.maximum(row_sums, 1e-12)
        # reset
        self.counts[:] = 0
        self.task_counts[:] = 0
        self.cpu_sums[:] = 0
        self.epoch_start_t = now

    def select_dest(self, task, env):
        """
        Return destination node id (int).
        If your framework uses cloud as None/-1, you can adapt here.
        """
        if not self.initialized:
            self._init_from_env(env)

        now = float(getattr(env, "now", getattr(env, "t", getattr(env, "time", 0.0))))

        src = int(getattr(task, "src", getattr(task, "src_id", getattr(task, "source", None))))
        if src is None:
            raise AttributeError("Task must have src/src_id/source")

        self.counts[src] += self._demand_increment(task)
        self.task_counts[src] += 1.0
        self.cpu_sums[src] += float(getattr(task, "cpu_demand", 1.0))
        t0 = time.perf_counter()
        self._maybe_update_epoch(now, env)

        p = self.P[src]
        p = p / np.maximum(p.sum(), 1e-12)
        dest = int(self.rng.choice(self.n, p=p))
        t1 = time.perf_counter()
        self._decision_time_sum += (t1 - t0)
        self._decision_calls += 1
        return dest

    def get_cost_stats(self) -> dict:
        ot_mean = (self._ot_time_sum / self._ot_updates) if self._ot_updates > 0 else 0.0
        dec_mean = (self._decision_time_sum / self._decision_calls) if self._decision_calls > 0 else 0.0
        return {
            "ot_solve_ms_mean": float(ot_mean * 1000.0),
            "ot_update_count": int(self._ot_updates),
            "decision_time_ms_mean": float(dec_mean * 1000.0),
        }
