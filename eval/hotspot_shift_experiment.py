from __future__ import annotations

import argparse
import json
import os
import time
import hashlib
import platform
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(__file__)))

from sim.env import Env
from sim.task import Task
from policies.greedy_latency import GreedyLatency
from policies.cap_greedy import CapAwareGreedy
from policies.sqlf import ShortestQueueFirst
from policies.cosims_sinkhorn import COSIMSSinkhorn
from policies.cosims_auction_B import COSIMSAuctionB


POLICIES = {
    "greedy_latency": GreedyLatency,
    "cap_greedy": CapAwareGreedy,
    "sqlf": ShortestQueueFirst,
    "cosims_sinkhorn": COSIMSSinkhorn,
    "cosims_auction_B": COSIMSAuctionB,
}


@dataclass
class RunArtifacts:
    summary_row: Dict[str, float]
    ts_df: pd.DataFrame
    stage_df: pd.DataFrame


def parse_seeds(text: str) -> List[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_loads(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def apply_event_overrides(
    config: dict,
    misfit_delta: float | None,
    wait_tau: float | None,
    slope_th: float | None,
    rate_factor: float | None,
):
    cosims = config.setdefault("cosims", {})
    event = cosims.setdefault("event", {})
    if misfit_delta is not None:
        event["misfit_delta"] = float(misfit_delta)
    if wait_tau is not None:
        event["wait_tau"] = float(wait_tau)
    if slope_th is not None:
        event["slope_th"] = float(slope_th)
    if rate_factor is not None:
        event["rate_factor"] = float(rate_factor)
    cosims["event"] = event
    config["cosims"] = cosims


def build_groups(n: int) -> List[np.ndarray]:
    parts = np.array_split(np.arange(n, dtype=int), 3)
    return [p.astype(int) for p in parts]


def build_region_latency_matrix(base_ms: np.ndarray, groups: List[np.ndarray], near_ms: float, far_ms: float) -> np.ndarray:
    n = int(base_ms.shape[0])
    out = np.array(base_ms, dtype=float, copy=True)
    src_to_gid = np.zeros(n, dtype=int)
    for gid, g in enumerate(groups):
        src_to_gid[g] = gid
    for i in range(n):
        gid = int(src_to_gid[i])
        for j in range(n):
            same = int(j in set(groups[gid].tolist()))
            if same == 1:
                out[i, j] = min(out[i, j], near_ms)
            else:
                out[i, j] = max(out[i, j], far_ms)
    np.fill_diagonal(out, min(near_ms * 0.4, 1.0))
    return out


def stage_group_probs() -> List[np.ndarray]:
    return [
        np.array([0.7, 0.2, 0.1], dtype=float),
        np.array([0.1, 0.7, 0.2], dtype=float),
        np.array([0.2, 0.1, 0.7], dtype=float),
    ]


def stage_index(t: float, sim_time: float) -> int:
    if t < sim_time / 3.0:
        return 0
    if t < 2.0 * sim_time / 3.0:
        return 1
    return 2


def sample_src(rng: np.random.Generator, groups: List[np.ndarray], probs: np.ndarray) -> int:
    gid = int(rng.choice(3, p=probs))
    bucket = groups[gid]
    return int(rng.choice(bucket))


def run_hotspot_once(config: dict, policy_name: str, seed: int, load: float, window_s: float, near_ms: float, far_ms: float) -> RunArtifacts:
    rng = np.random.default_rng(seed)
    policy_cls = POLICIES[policy_name]
    policy = policy_cls(config, rng)
    env = Env(config, rng=rng, policy=policy)
    if hasattr(policy, "on_env_init"):
        policy.on_env_init(env)

    groups = build_groups(env.n)
    env.net.lat_matrix_ms = build_region_latency_matrix(env.net.lat_matrix_ms, groups, near_ms=near_ms, far_ms=far_ms)

    warmup = float(config["warmup_time"])
    sim_time = float(config["sim_time"])
    lam = float(config["arrival"]["base_total_rate"]) * float(load)
    t = 0.0
    tids = 0
    probs_list = stage_group_probs()

    arrivals: List[float] = []
    lats: List[float] = []
    clouds: List[float] = []
    backlog_time_total: List[float] = []
    backlog_work_total: List[float] = []
    queue_imbalance: List[float] = []
    queue_gap: List[float] = []
    queue_peak: List[float] = []
    stage_ids: List[int] = []
    trig_eval_cum: List[int] = []
    trig_fire_cum: List[int] = []
    queue_rows: List[np.ndarray] = []

    while t < sim_time:
        dt = float(rng.exponential(1.0 / max(lam, 1e-12)))
        t += dt
        if t >= sim_time:
            break
        env.now = t
        sid = stage_index(t, sim_time)
        src = sample_src(rng, groups, probs_list[sid])
        cpu = float(rng.lognormal(mean=float(config["task"]["cpu_logn_mu"]), sigma=float(config["task"]["cpu_logn_sigma"])))
        data_mb = float(rng.uniform(float(config["task"]["data_mb_min"]), float(config["task"]["data_mb_max"])))
        task = Task(
            tid=tids,
            src=src,
            arrival_t=t,
            cpu_demand=cpu,
            data_mb=data_mb,
        )
        tids += 1
        w = np.maximum(np.asarray(env.available_time, dtype=float) - t, 0.0)
        task.backlog_time_total = float(w.sum())
        task.backlog_work_total = float(np.dot(w, env.xi))
        dest = env.policy.select_dest(task, env)
        if dest is None:
            task.is_cloud = True
            task.dest = None
            start = max(task.arrival_t + env.cloud_latency_s, env.cloud_available_time)
            exec_t = task.cpu_demand / env.cloud_xi
            finish = start + exec_t
            env.cloud_available_time = finish
        else:
            task.is_cloud = False
            task.dest = int(dest)
            net = env.net.latency_s(task.src, task.dest)
            start = max(task.arrival_t + net, float(env.available_time[task.dest]))
            exec_t = task.cpu_demand / float(env.xi[task.dest])
            finish = start + exec_t
            env.available_time[task.dest] = finish
        task.start_t = float(start)
        task.finish_t = float(finish)

        arrivals.append(float(task.arrival_t))
        lats.append(float(task.finish_t - task.arrival_t))
        clouds.append(1.0 if task.is_cloud else 0.0)
        backlog_time_total.append(float(task.backlog_time_total))
        backlog_work_total.append(float(task.backlog_work_total))
        queue_imbalance.append(float(np.var(w)))
        queue_gap.append(float(np.max(w) - np.min(w)))
        queue_peak.append(float(np.max(w)))
        stage_ids.append(int(sid))
        queue_rows.append(w.copy())
        trig_eval_cum.append(int(getattr(policy, "_trig_eval_calls", 0)))
        trig_fire_cum.append(int(getattr(policy, "_auction_calls", 0)))

    arr = np.array(arrivals, dtype=float)
    lat = np.array(lats, dtype=float)
    cloud = np.array(clouds, dtype=float)
    btime = np.array(backlog_time_total, dtype=float)
    bwork = np.array(backlog_work_total, dtype=float)
    qimb = np.array(queue_imbalance, dtype=float)
    qgap = np.array(queue_gap, dtype=float)
    qpeak = np.array(queue_peak, dtype=float)
    sid_arr = np.array(stage_ids, dtype=int)
    trig_eval_cum_arr = np.array(trig_eval_cum, dtype=int)
    trig_fire_cum_arr = np.array(trig_fire_cum, dtype=int)
    qmat = np.vstack(queue_rows) if len(queue_rows) > 0 else np.zeros((0, env.n), dtype=float)

    filt = arr >= warmup
    if int(filt.sum()) > 0:
        avg_latency = float(lat[filt].mean())
        p95_latency = float(np.quantile(lat[filt], 0.95))
        cloud_rate = float(cloud[filt].mean())
        throughput = float(filt.sum() / max(sim_time - warmup, 1e-12))
        queue_peak_global = float(qpeak[filt].max())
        queue_imbalance_mean = float(qimb[filt].mean())
        queue_gap_mean = float(qgap[filt].mean())
    else:
        avg_latency = 0.0
        p95_latency = 0.0
        cloud_rate = 0.0
        throughput = 0.0
        queue_peak_global = 0.0
        queue_imbalance_mean = 0.0
        queue_gap_mean = 0.0

    stage_rows = []
    for sid in [0, 1, 2]:
        m = (sid_arr == sid) & (arr >= warmup)
        if int(m.sum()) == 0:
            continue
        stage_rows.append({
            "policy": policy_name,
            "seed": int(seed),
            "load": float(load),
            "stage": int(sid),
            "avg_latency": float(lat[m].mean()),
            "p95_latency": float(np.quantile(lat[m], 0.95)),
            "cloud_rate": float(cloud[m].mean()),
            "queue_imbalance_mean": float(qimb[m].mean()),
            "queue_gap_mean": float(qgap[m].mean()),
        })
    stage_df = pd.DataFrame(stage_rows)

    edges = np.arange(warmup, sim_time + 1e-9, window_s)
    if edges.size < 2:
        edges = np.array([warmup, sim_time], dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2.0

    ts_rows = []
    for i in range(len(centers)):
        a = float(edges[i])
        b = float(edges[i + 1])
        m = (arr >= a) & (arr < b)
        if int(m.sum()) == 0:
            continue
        sid_w = stage_index(float(centers[i]), sim_time)
        avg_l = float(lat[m].mean())
        p95_l = float(np.quantile(lat[m], 0.95))
        cloud_r = float(cloud[m].mean())
        qimb_m = float(qimb[m].mean())
        qgap_m = float(qgap[m].mean())
        qpk_m = float(qpeak[m].max())
        btime_m = float(btime[m].mean())
        bwork_m = float(bwork[m].mean())
        trig_rate = float("nan")
        if policy_name == "cosims_auction_B":
            idx = np.where(m)[0]
            i0 = int(idx[0])
            i1 = int(idx[-1])
            eval0 = int(trig_eval_cum_arr[i0 - 1]) if i0 > 0 else 0
            fire0 = int(trig_fire_cum_arr[i0 - 1]) if i0 > 0 else 0
            eval1 = int(trig_eval_cum_arr[i1])
            fire1 = int(trig_fire_cum_arr[i1])
            den = max(eval1 - eval0, 1)
            trig_rate = float((fire1 - fire0) / den)
        row = {
            "policy": policy_name,
            "seed": int(seed),
            "load": float(load),
            "time": float(centers[i]),
            "stage": int(sid_w),
            "avg_latency_ts": avg_l,
            "p95_latency_ts": p95_l,
            "cloud_rate_ts": cloud_r,
            "queue_imbalance_ts": qimb_m,
            "queue_gap_ts": qgap_m,
            "queue_peak_ts": qpk_m,
            "backlog_time_total_ts": btime_m,
            "backlog_work_total_ts": bwork_m,
            "trigger_rate_ts": trig_rate,
        }
        if qmat.shape[0] > 0:
            q_mean = qmat[m].mean(axis=0)
            for j in range(env.n):
                row[f"queue_node_{j}_mean"] = float(q_mean[j])
        ts_rows.append(row)
    ts_df = pd.DataFrame(ts_rows)

    recovery_q_ab = compute_recovery_time(ts_df, switch_t=sim_time / 3.0, end_t=2.0 * sim_time / 3.0, col="queue_imbalance_ts")
    recovery_q_bc = compute_recovery_time(ts_df, switch_t=2.0 * sim_time / 3.0, end_t=sim_time, col="queue_imbalance_ts")
    recovery_p95_ab = compute_recovery_time(ts_df, switch_t=sim_time / 3.0, end_t=2.0 * sim_time / 3.0, col="p95_latency_ts")
    recovery_p95_bc = compute_recovery_time(ts_df, switch_t=2.0 * sim_time / 3.0, end_t=sim_time, col="p95_latency_ts")

    cost_stats = {}
    if hasattr(policy, "get_cost_stats"):
        try:
            cost_stats = policy.get_cost_stats()
        except Exception:
            cost_stats = {}

    summary_row = {
        "policy": policy_name,
        "seed": int(seed),
        "load": float(load),
        "throughput": throughput,
        "avg_latency": avg_latency,
        "p95_latency": p95_latency,
        "cloud_rate": cloud_rate,
        "completion": 1.0,
        "queue_peak": queue_peak_global,
        "queue_imbalance_mean": queue_imbalance_mean,
        "queue_gap_mean": queue_gap_mean,
        "recovery_q_ab": recovery_q_ab,
        "recovery_q_bc": recovery_q_bc,
        "recovery_p95_ab": recovery_p95_ab,
        "recovery_p95_bc": recovery_p95_bc,
    }
    for k, v in cost_stats.items():
        summary_row[k] = v

    return RunArtifacts(summary_row=summary_row, ts_df=ts_df, stage_df=stage_df)


def compute_recovery_time(ts_df: pd.DataFrame, switch_t: float, end_t: float, col: str) -> float:
    if ts_df.empty or col not in ts_df.columns:
        return float("nan")
    d = ts_df[(ts_df["time"] >= switch_t) & (ts_df["time"] < end_t)][["time", col]].dropna().sort_values("time")
    if d.empty:
        return float("nan")
    val = d[col].to_numpy(dtype=float)
    tim = d["time"].to_numpy(dtype=float)
    peak = float(np.max(val))
    k = max(1, int(0.3 * len(val)))
    steady = float(np.mean(val[-k:]))
    target = float(steady + 0.1 * abs(peak - steady))
    if peak >= steady:
        idx = np.where(val <= target)[0]
    else:
        idx = np.where(val >= target)[0]
    if idx.size == 0:
        return float(end_t - switch_t)
    return float(max(tim[int(idx[0])] - switch_t, 0.0))


def agg_mean_std(df: pd.DataFrame, keys: List[str], group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for group, g in df.groupby(group_cols):
        if not isinstance(group, tuple):
            group = (group,)
        row = {c: v for c, v in zip(group_cols, group)}
        for k in keys:
            s = pd.to_numeric(g[k], errors="coerce").dropna()
            if s.empty:
                row[f"{k}_mean"] = np.nan
                row[f"{k}_std"] = np.nan
            else:
                row[f"{k}_mean"] = float(s.mean())
                row[f"{k}_std"] = float(s.std(ddof=1)) if s.size > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(group_cols).reset_index(drop=True)


def make_plots(ts_mean_std: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for load in sorted(ts_mean_std["load"].dropna().unique().tolist()):
        d = ts_mean_std[ts_mean_std["load"] == load].copy()

        plt.figure(figsize=(8, 4))
        for p, g in d.groupby("policy"):
            g = g.sort_values("time")
            plt.plot(g["time"], g["queue_imbalance_ts_mean"], label=str(p))
        plt.xlabel("Time (s)")
        plt.ylabel("Queue Imbalance")
        plt.title(f"Hotspot Shift Queue Imbalance (load={load})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"queue_imbalance_over_time_load{load}.png"))
        plt.close()

        plt.figure(figsize=(8, 4))
        for p, g in d.groupby("policy"):
            g = g.sort_values("time")
            plt.plot(g["time"], g["p95_latency_ts_mean"], label=str(p))
        plt.xlabel("Time (s)")
        plt.ylabel("Rolling P95 Latency (s)")
        plt.title(f"Hotspot Shift Rolling P95 (load={load})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"rolling_p95_over_time_load{load}.png"))
        plt.close()

        b = d[d["policy"] == "cosims_auction_B"].sort_values("time")
        if not b.empty:
            plt.figure(figsize=(8, 4))
            plt.plot(b["time"], b["trigger_rate_ts_mean"])
            plt.xlabel("Time (s)")
            plt.ylabel("Trigger Rate")
            plt.title(f"Hotspot Shift Trigger Rate (B, load={load})")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"trigger_rate_over_time_B_load{load}.png"))
            plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/auction_B_recommended.json")
    ap.add_argument("--out_dir", type=str, default="outputs/core_scenarios/hotspot_shift")
    ap.add_argument("--loads", type=str, default="1.0,1.2")
    ap.add_argument("--seeds", type=str, default="0-4")
    ap.add_argument("--policies", type=str, default="cap_greedy,sqlf,cosims_auction_B,cosims_sinkhorn,greedy_latency")
    ap.add_argument("--window_s", type=float, default=20.0)
    ap.add_argument("--near_ms", type=float, default=8.0)
    ap.add_argument("--far_ms", type=float, default=35.0)
    ap.add_argument("--misfit_delta", type=float, default=None)
    ap.add_argument("--wait_tau", type=float, default=None)
    ap.add_argument("--slope_th", type=float, default=None)
    ap.add_argument("--rate_factor", type=float, default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    apply_event_overrides(
        config=config,
        misfit_delta=args.misfit_delta,
        wait_tau=args.wait_tau,
        slope_th=args.slope_th,
        rate_factor=args.rate_factor,
    )

    loads = parse_loads(args.loads)
    seeds = parse_seeds(args.seeds)
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    for p in policies:
        if p not in POLICIES:
            raise ValueError(f"Unknown policy: {p}")

    os.makedirs(args.out_dir, exist_ok=True)
    raw_dir = os.path.join(args.out_dir, "raw")
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    summary_rows = []
    ts_frames = []
    stage_frames = []

    for load in loads:
        for seed in seeds:
            for policy in policies:
                art = run_hotspot_once(
                    config=config,
                    policy_name=policy,
                    seed=seed,
                    load=load,
                    window_s=float(args.window_s),
                    near_ms=float(args.near_ms),
                    far_ms=float(args.far_ms),
                )
                summary_rows.append(art.summary_row)
                if not art.ts_df.empty:
                    ts_frames.append(art.ts_df)
                if not art.stage_df.empty:
                    stage_frames.append(art.stage_df)
                print(f"[OK] load={load} seed={seed} policy={policy}")

    df_summary_runs = pd.DataFrame(summary_rows).sort_values(["load", "policy", "seed"])
    df_summary_runs.to_csv(os.path.join(raw_dir, "summary_runs.csv"), index=False)

    summary_keys = [c for c in df_summary_runs.columns if c not in {"load", "policy", "seed"}]
    df_summary_mean_std = agg_mean_std(df_summary_runs, keys=summary_keys, group_cols=["load", "policy"])
    df_summary_mean_std.to_csv(os.path.join(args.out_dir, "summary_mean_std.csv"), index=False)

    df_ts_runs = pd.concat(ts_frames, axis=0, ignore_index=True) if len(ts_frames) > 0 else pd.DataFrame()
    if not df_ts_runs.empty:
        df_ts_runs = df_ts_runs.sort_values(["load", "policy", "seed", "time"])
        df_ts_runs.to_csv(os.path.join(raw_dir, "timeseries_runs.csv"), index=False)
        ts_keys = [c for c in df_ts_runs.columns if c not in {"load", "policy", "seed", "time", "stage"}]
        df_ts_mean_std = agg_mean_std(df_ts_runs, keys=ts_keys, group_cols=["load", "policy", "time", "stage"])
        df_ts_mean_std.to_csv(os.path.join(args.out_dir, "timeseries_mean_std.csv"), index=False)
        make_plots(df_ts_mean_std, fig_dir)
    else:
        df_ts_mean_std = pd.DataFrame()

    df_stage_runs = pd.concat(stage_frames, axis=0, ignore_index=True) if len(stage_frames) > 0 else pd.DataFrame()
    if not df_stage_runs.empty:
        df_stage_runs = df_stage_runs.sort_values(["load", "policy", "seed", "stage"])
        df_stage_runs.to_csv(os.path.join(raw_dir, "stage_runs.csv"), index=False)
        stage_keys = [c for c in df_stage_runs.columns if c not in {"load", "policy", "seed", "stage"}]
        df_stage_mean_std = agg_mean_std(df_stage_runs, keys=stage_keys, group_cols=["load", "policy", "stage"])
        df_stage_mean_std.to_csv(os.path.join(args.out_dir, "stage_mean_std.csv"), index=False)
    else:
        df_stage_mean_std = pd.DataFrame()

    rec_cols = ["recovery_q_ab", "recovery_q_bc", "recovery_p95_ab", "recovery_p95_bc"]
    df_recovery = df_summary_runs[["load", "policy", "seed"] + rec_cols].copy()
    df_recovery.to_csv(os.path.join(raw_dir, "recovery_runs.csv"), index=False)
    df_recovery_mean_std = agg_mean_std(df_recovery, keys=rec_cols, group_cols=["load", "policy"])
    df_recovery_mean_std.to_csv(os.path.join(args.out_dir, "recovery_mean_std.csv"), index=False)

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "config_path": args.config,
        "loads": loads,
        "seeds": seeds,
        "policies": policies,
        "window_s": float(args.window_s),
        "near_ms": float(args.near_ms),
        "far_ms": float(args.far_ms),
        "event_overrides": {
            "misfit_delta": args.misfit_delta,
            "wait_tau": args.wait_tau,
            "slope_th": args.slope_th,
            "rate_factor": args.rate_factor,
        },
        "env": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": plt.matplotlib.__version__,
            "platform": platform.platform(),
        },
    }
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_text = f.read()
    meta["config_sha256"] = hashlib.sha256(cfg_text.encode("utf-8")).hexdigest()
    with open(os.path.join(args.out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
        f.write(cfg_text)

    print(f"[OK] summary: {os.path.join(args.out_dir, 'summary_mean_std.csv')}")
    print(f"[OK] timeseries: {os.path.join(args.out_dir, 'timeseries_mean_std.csv')}")
    print(f"[OK] recovery: {os.path.join(args.out_dir, 'recovery_mean_std.csv')}")


if __name__ == "__main__":
    main()
