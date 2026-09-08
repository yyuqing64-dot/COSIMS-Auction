from __future__ import annotations
import numpy as np
from collections import deque
import time

from policies.cosims_sinkhorn import sinkhorn_transport, _safe_normalize


class COSIMSAuctionB:
    name = "cosims_auction_B"

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
        window_sec: float | None = None,
        rate_factor: float | None = None,
        slope_th: float | None = None,
        wait_tau: float | None = None,
        misfit_delta: float | None = None,
    ):
        cosims_cfg = config.get("cosims", {})
        event_cfg = cosims_cfg.get("event", {})
        self.epoch_sec = float(epoch_sec or cosims_cfg.get("epoch_sec", 1.0))
        self.lam = float(lam or cosims_cfg.get("lam", 0.1))
        self.max_iter = int(max_iter or cosims_cfg.get("max_iter", 35))
        self.ot_demand_mode = str(cosims_cfg.get("ot_demand_mode", "count")).lower()
        self.ot_cost_mode = str(cosims_cfg.get("ot_cost_mode", "prop")).lower()
        self.cost_service_weight = float(cosims_cfg.get("ot_cost_service_weight", 1.0))
        self.cost_queue_weight = float(cosims_cfg.get("ot_cost_queue_weight", 1.0))
        self.topk = int(topk or cosims_cfg.get("topk", 5))
        self.alpha = float(alpha or cosims_cfg.get("alpha", 1.0))
        self.eta = float(eta or cosims_cfg.get("eta", 1.0))
        self.beta = float(cosims_cfg.get("beta", 0.0))
        self.eps_noise = float(eps_noise or cosims_cfg.get("eps_noise", 0.0))
        self.window_sec = float(window_sec or event_cfg.get("window_sec", 5.0))
        self.rate_factor = float(rate_factor or event_cfg.get("rate_factor", 1.5))
        self.slope_th = float(slope_th or event_cfg.get("slope_th", 0.5))
        self.wait_tau = float(wait_tau or event_cfg.get("wait_tau", 1.0))
        self.misfit_delta = float(misfit_delta or event_cfg.get("misfit_delta", 0.2))
        self.slope_dt_min = float(event_cfg.get("slope_dt_min", 0.0))
        self.slope_dt_max = float(event_cfg.get("slope_dt_max", 1e9))
        self.default_light_compare = bool(event_cfg.get("default_light_compare", True))
        self.cloud_enabled = bool(config["cloud"]["enabled"])
        self.rng = rng

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

        self.arrivals = deque()
        self.prev_rate = None
        self.last_rate_t = 0.0
        self.prev_backlog = None
        self.last_backlog_t = 0.0
        self._ot_time_sum = 0.0
        self._ot_updates = 0
        self._decision_time_sum = 0.0
        self._decision_calls = 0
        self._auction_time_sum = 0.0
        self._auction_calls = 0
        self._trig_arr_time_sum = 0.0
        self._trig_backlog_time_sum = 0.0
        self._trig_misfit_time_sum = 0.0
        self._trig_eval_calls = 0
        self._trig_fired_calls = 0

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
            net = getattr(env, "net", None)
            if net is not None and hasattr(net, "lat_ms"):
                M = net.lat_ms
        if M is None:
            raise AttributeError("Env must provide cost matrix (cost_matrix()/M/latency_matrix/lat_matrix/net_lat)")
        self.base_M = np.asarray(M, dtype=float)
        self.M = self.base_M.copy()
        assert self.M.shape == (self.n, self.n)

        self.counts = np.zeros(self.n, dtype=float)
        self.task_counts = np.zeros(self.n, dtype=float)
        self.cpu_sums = np.zeros(self.n, dtype=float)
        self.P = np.ones((self.n, self.n), dtype=float) / self.n
        self.prev_backlog = np.zeros(self.n, dtype=float)
        self.initialized = True
        self.epoch_start_t = 0.0
        self.last_backlog_t = 0.0
        self.last_rate_t = 0.0
        self.prev_rate = 0.0
        self.arrivals.clear()

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
            row_sums = self.P.sum(axis=1, keepdims=True)
            self.P = self.P / np.maximum(row_sums, 1e-12)
        self.counts[:] = 0
        self.task_counts[:] = 0
        self.cpu_sums[:] = 0
        self.epoch_start_t = now

    def _topk_indices(self, probs_row: np.ndarray, k: int) -> np.ndarray:
        k = max(1, min(k, probs_row.size))
        idx = np.argpartition(-probs_row, kth=k - 1)[:k]
        idx = idx[np.argsort(-probs_row[idx])]
        return idx

    def _backlog_time(self, env, j: int, now: float) -> float:
        return float(max(0.0, float(env.available_time[j]) - now))

    def _bid(self, task, env, j: int, now: float) -> float:
        L_ij = float(env.net.latency_s(task.src, j))
        mu_j = float(env.xi[j])
        W_j = self._backlog_time(env, j, now)
        s_ij = float(task.cpu_demand) / max(mu_j, 1e-12)
        dt_raw = float(max(now - self.last_backlog_t, 1e-9))
        dt = float(np.clip(dt_raw, self.slope_dt_min if self.slope_dt_min > 0 else dt_raw,
                           self.slope_dt_max if self.slope_dt_max > 0 else dt_raw))
        slope_j = float(max(0.0, (W_j - float(self.prev_backlog[j])) / dt))
        p_j = self.eta * W_j + self.beta * slope_j + self.eps_noise
        return float(self.alpha * L_ij + W_j + s_ij + p_j)

    def _light_score(self, task, env, j: int, now: float) -> float:
        L_ij = float(env.net.latency_s(task.src, j))
        mu_j = float(env.xi[j])
        W_j = self._backlog_time(env, j, now)
        s_ij = float(task.cpu_demand) / max(mu_j, 1e-12)
        return float(self.alpha * L_ij + W_j + s_ij)

    def _arrival_trigger(self, now: float) -> bool:
        t0 = time.perf_counter()
        self.arrivals.append(now)
        while self.arrivals and now - self.arrivals[0] > self.window_sec:
            self.arrivals.popleft()
        cur_rate = float(len(self.arrivals)) / max(self.window_sec, 1e-12)
        prev_rate = float(self.prev_rate or cur_rate)
        triggered = cur_rate > self.rate_factor * max(prev_rate, 1e-12)
        if now - self.last_rate_t >= self.window_sec:
            self.prev_rate = cur_rate
            self.last_rate_t = now
        t1 = time.perf_counter()
        self._trig_arr_time_sum += (t1 - t0)
        return bool(triggered)

    def _backlog_triggers(self, env, now: float, top1: int) -> bool:
        t0 = time.perf_counter()
        W = np.maximum(np.asarray(env.available_time, dtype=float) - now, 0.0)
        dt_raw = float(max(now - self.last_backlog_t, 1e-9))
        dt = float(np.clip(dt_raw, self.slope_dt_min if self.slope_dt_min > 0 else dt_raw,
                           self.slope_dt_max if self.slope_dt_max > 0 else dt_raw))
        slope = (W - self.prev_backlog) / dt
        self.prev_backlog = W
        self.last_backlog_t = now
        cong = bool(W[int(top1)] > self.wait_tau)
        abrupt = bool(np.any(slope > self.slope_th))
        t1 = time.perf_counter()
        self._trig_backlog_time_sum += (t1 - t0)
        return bool(cong or abrupt)

    def _misfit_trigger(self, task, env, now: float, p_row: np.ndarray, top1: int) -> bool:
        t0 = time.perf_counter()
        cand = self._topk_indices(p_row, self.topk)
        bids = [self._bid(task, env, int(j), now) for j in cand]
        b_top1 = float(self._bid(task, env, int(top1), now))
        b_best = float(np.min(bids))
        t1 = time.perf_counter()
        self._trig_misfit_time_sum += (t1 - t0)
        return bool(b_top1 - b_best > self.misfit_delta)

    def select_dest(self, task, env):
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
        p_row = self.P[src]
        p_row = p_row / np.maximum(p_row.sum(), 1e-12)
        top1 = int(np.argmax(p_row))
        ta0 = time.perf_counter()
        trig_arr = self._arrival_trigger(now)
        trig_backlog = self._backlog_triggers(env, now, top1)
        trig_misfit = self._misfit_trigger(task, env, now, p_row, top1)
        ta1 = time.perf_counter()
        self._trig_eval_calls += 1
        if trig_arr or trig_backlog or trig_misfit:
            cand = self._topk_indices(p_row, self.topk)
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
            self._trig_fired_calls += 1
            dest = int(best_j)
        else:
            if self.default_light_compare and p_row.size >= 2:
                cand2 = self._topk_indices(p_row, 2)
                j1 = int(cand2[0])
                j2 = int(cand2[1])
                s1 = self._light_score(task, env, j1, now)
                s2 = self._light_score(task, env, j2, now)
                dest = int(j1 if s1 <= s2 else j2)
            else:
                dest = int(top1)
        if self.cloud_enabled:
            edge_finish = env.est_finish_time_edge(task, dest)
            cloud_finish = env.est_finish_time_cloud(task)
            if cloud_finish < edge_finish:
                t1 = time.perf_counter()
                self._decision_time_sum += (t1 - t0)
                self._decision_calls += 1
                return None
        t1 = time.perf_counter()
        self._decision_time_sum += (t1 - t0)
        self._decision_calls += 1
        return int(dest)

    def get_cost_stats(self) -> dict:
        ot_mean = (self._ot_time_sum / self._ot_updates) if self._ot_updates > 0 else 0.0
        dec_mean = (self._decision_time_sum / self._decision_calls) if self._decision_calls > 0 else 0.0
        auction_mean = (self._auction_time_sum / self._auction_calls) if self._auction_calls > 0 else 0.0
        trig_arr_mean = (self._trig_arr_time_sum / self._trig_eval_calls) if self._trig_eval_calls > 0 else 0.0
        trig_backlog_mean = (self._trig_backlog_time_sum / self._trig_eval_calls) if self._trig_eval_calls > 0 else 0.0
        trig_misfit_mean = (self._trig_misfit_time_sum / self._trig_eval_calls) if self._trig_eval_calls > 0 else 0.0
        auction_call_rate = float(self._auction_calls / max(self._trig_eval_calls, 1))
        return {
            "ot_solve_ms_mean": float(ot_mean * 1000.0),
            "ot_update_count": int(self._ot_updates),
            "decision_time_ms_mean": float(dec_mean * 1000.0),
            "auction_time_ms_mean": float(auction_mean * 1000.0),
            "trigger_eval_arrival_ms_mean": float(trig_arr_mean * 1000.0),
            "trigger_eval_backlog_ms_mean": float(trig_backlog_mean * 1000.0),
            "trigger_eval_misfit_ms_mean": float(trig_misfit_mean * 1000.0),
            "trigger_fired_rate": auction_call_rate,
            "auction_call_rate": auction_call_rate,
            "topk": int(self.topk),
        }
