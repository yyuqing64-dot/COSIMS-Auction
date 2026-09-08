from __future__ import annotations
import numpy as np
from typing import List, Dict
from sim.task import Task

def compute_metrics(tasks: List[Task], warmup: float, sim_time: float) -> Dict[str, float]:
    # filter by arrival time after warmup
    filt = [t for t in tasks if t.arrival_t >= warmup]
    if len(filt) == 0:
        return {
            "throughput": 0.0,
            "avg_latency": 0.0,
            "p95_latency": 0.0,
            "cloud_rate": 0.0,
            "completion": 0.0,
        }

    lat = np.array([t.finish_t - t.arrival_t for t in filt], dtype=float)
    avg_latency = float(lat.mean())
    p95_latency = float(np.quantile(lat, 0.95))
    cloud_rate = float(np.mean([1.0 if t.is_cloud else 0.0 for t in filt]))

    # In this minimal simulator, tasks always complete (no deadline drops).
    # completion is defined as completed/arrived.
    completion = 1.0

    completed = len(filt)
    horizon = max(sim_time - warmup, 1e-9)
    throughput = float(completed / horizon)

    return {
        "throughput": throughput,
        "avg_latency": avg_latency,
        "p95_latency": p95_latency,
        "cloud_rate": cloud_rate,
        "completion": completion,
    }

def compute_timeseries(tasks: List[Task], warmup: float, sim_time: float, window_s: float) -> Dict[str, np.ndarray]:
    if window_s <= 0:
        return {}
    t0 = float(warmup)
    t1 = float(sim_time)
    edges = np.arange(t0, t1 + 1e-9, window_s)
    centers = (edges[:-1] + edges[1:]) / 2.0
    p95 = np.full(centers.shape, np.nan, dtype=float)
    thr = np.full(centers.shape, np.nan, dtype=float)
    cloud = np.full(centers.shape, np.nan, dtype=float)
    back_time = np.full(centers.shape, np.nan, dtype=float)
    back_work = np.full(centers.shape, np.nan, dtype=float)
    arr = np.array([t.arrival_t for t in tasks], dtype=float)
    lat = np.array([t.finish_t - t.arrival_t for t in tasks], dtype=float)
    is_cloud = np.array([1.0 if t.is_cloud else 0.0 for t in tasks], dtype=float)
    btime = np.array([getattr(t, "backlog_time_total", 0.0) for t in tasks], dtype=float)
    bwork = np.array([getattr(t, "backlog_work_total", 0.0) for t in tasks], dtype=float)
    for i in range(len(centers)):
        a = edges[i]
        b = edges[i + 1]
        mask = (arr >= a) & (arr < b)
        cnt = int(mask.sum())
        if cnt <= 0:
            continue
        p95[i] = float(np.quantile(lat[mask], 0.95))
        thr[i] = float(cnt / max(b - a, 1e-9))
        cloud[i] = float(is_cloud[mask].mean())
        back_time[i] = float(btime[mask].mean())
        back_work[i] = float(bwork[mask].mean())
    return {
        "time": centers,
        "p95_latency_ts": p95,
        "throughput_ts": thr,
        "cloud_rate_ts": cloud,
        "backlog_time_total_ts": back_time,
        "backlog_work_total_ts": back_work,
    }
