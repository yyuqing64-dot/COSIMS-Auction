from __future__ import annotations
import numpy as np
import time

from policies.cosims_sinkhorn import sinkhorn_transport, _safe_normalize


class COSIMSAuctionA:
    name = "cosims_auction_A"

    def __init__(
        self,
        config: dict,
        rng: np.random.Generator,
        epoch_sec: float | None = None,
        lam: float | None = None,
        max_iter: int | None = None,
        topk: int | None = None,
        alpha: float | None = None,
        eta: float | None = None,
        eps_noise: float | None = None,
    ):
        cosims_cfg = config.get("cosims", {})
        self.epoch_sec = float(epoch_sec or cosims_cfg.get("epoch_sec", 1.0))
        self.lam = float(lam or cosims_cfg.get("lam", 0.1))
        self.max_iter = int(max_iter or cosims_cfg.get("max_iter", 35))
        self.topk = int(topk or cosims_cfg.get("topk", 5))
        self.alpha = float(alpha or cosims_cfg.get("alpha", 1.0))
        self.eta = float(eta or cosims_cfg.get("eta", 1.0))
        self.eps_noise = float(eps_noise or cosims_cfg.get("eps_noise", 0.0))
        self.cloud_enabled = bool(config["cloud"]["enabled"])
        self.rng = rng

        self.initialized = False
        self.n = None
        self.o = None
        self.M = None
        self.epoch_start_t = 0.0
        self.counts = None
        self.P = None
        self._ot_time_sum = 0.0
        self._ot_updates = 0
        self._decision_time_sum = 0.0
        self._decision_calls = 0
        self._auction_time_sum = 0.0
        self._auction_calls = 0

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
            # fallback to network ms matrix if available
            net = getattr(env, "net", None)
            if net is not None and hasattr(net, "lat_ms"):
                M = net.lat_ms
        if M is None:
            raise AttributeError("Env must provide cost matrix (cost_matrix()/M/latency_matrix/lat_matrix/net_lat)")
        self.M = np.asarray(M, dtype=float)
        assert self.M.shape == (self.n, self.n), f"Cost matrix shape must be (n,n), got {self.M.shape}"

        self.counts = np.zeros(self.n, dtype=int)
        self.P = np.ones((self.n, self.n), dtype=float) / self.n
        self.initialized = True
        self.epoch_start_t = 0.0

    def _maybe_update_epoch(self, now: float):
        if now - self.epoch_start_t < self.epoch_sec:
            return
        total = int(self.counts.sum())
        if total > 0:
            r = self.counts.astype(float) / float(total)
            t0 = time.perf_counter()
            self.P = sinkhorn_transport(self.M, r, self.o, lam=self.lam, max_iter=self.max_iter)
            t1 = time.perf_counter()
            self._ot_time_sum += (t1 - t0)
            self._ot_updates += 1
            row_sums = self.P.sum(axis=1, keepdims=True)
            self.P = self.P / np.maximum(row_sums, 1e-12)
        self.counts[:] = 0
        self.epoch_start_t = now

    def _topk_indices(self, probs_row: np.ndarray, k: int) -> np.ndarray:
        k = max(1, min(k, probs_row.size))
        # argsort descending by probability
        idx = np.argpartition(-probs_row, kth=k - 1)[:k]
        # sort by actual value descending to keep stable ordering
        idx = idx[np.argsort(-probs_row[idx])]
        return idx

    def _backlog_time(self, env, j: int, now: float) -> float:
        # single-server queue modeled by available_time
        return float(max(0.0, float(env.available_time[j]) - now))

    def _bid(self, task, env, j: int, now: float) -> float:
        # components
        L_ij = float(env.net.latency_s(task.src, j))
        mu_j = float(env.xi[j])
        W_j = self._backlog_time(env, j, now)
        s_ij = float(task.cpu_demand) / max(mu_j, 1e-12)
        # congestion price: eta * backlog_time + eps_noise
        p_j = self.eta * W_j + self.eps_noise
        bid = self.alpha * L_ij + W_j + s_ij + p_j
        return float(bid)

    def select_dest(self, task, env):
        if not self.initialized:
            self._init_from_env(env)

        now = float(getattr(env, "now", getattr(env, "t", getattr(env, "time", 0.0))))
        src = int(getattr(task, "src", getattr(task, "src_id", getattr(task, "source", None))))
        if src is None:
            raise AttributeError("Task must have src/src_id/source")

        self.counts[src] += 1
        t0 = time.perf_counter()
        self._maybe_update_epoch(now)

        p_row = self.P[src]
        p_row = p_row / np.maximum(p_row.sum(), 1e-12)
        cand = self._topk_indices(p_row, self.topk)

        # compute bids for candidates, choose argmin
        best_j = int(cand[0])
        best_bid = float("inf")
        tb0 = time.perf_counter()
        for j in cand:
            b = self._bid(task, env, int(j), now)
            if b < best_bid:
                best_bid = b
                best_j = int(j)
        tb1 = time.perf_counter()
        self._auction_time_sum += (tb1 - tb0)
        self._auction_calls += 1

        # optional: compare with cloud
        if self.cloud_enabled:
            edge_finish = env.est_finish_time_edge(task, best_j)
            cloud_finish = env.est_finish_time_cloud(task)
            if cloud_finish < edge_finish:
                return None

        t1 = time.perf_counter()
        self._decision_time_sum += (t1 - t0)
        self._decision_calls += 1
        return int(best_j)

    def get_cost_stats(self) -> dict:
        ot_mean = (self._ot_time_sum / self._ot_updates) if self._ot_updates > 0 else 0.0
        dec_mean = (self._decision_time_sum / self._decision_calls) if self._decision_calls > 0 else 0.0
        auction_mean = (self._auction_time_sum / self._auction_calls) if self._auction_calls > 0 else 0.0
        return {
            "ot_solve_ms_mean": float(ot_mean * 1000.0),
            "ot_update_count": int(self._ot_updates),
            "decision_time_ms_mean": float(dec_mean * 1000.0),
            "auction_time_ms_mean": float(auction_mean * 1000.0),
            "topk": int(self.topk),
        }
