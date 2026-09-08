from __future__ import annotations

import argparse
import json
import os
import time
import hashlib
import platform
import sys
from dataclasses import dataclass
from typing import Dict, List

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


def parse_floats(text: str) -> List[float]:
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


def stage_index(t: float, sim_time: float, phase_ratios: List[float]) -> int:
    b1 = phase_ratios[0] * sim_time
    b2 = (phase_ratios[0] + phase_ratios[1]) * sim_time
    if t < b1:
        return 0
    if t < b2:
        return 1
    return 2


def run_burst_once(
    config: dict,
    policy_name: str,
    seed: int,
    base_load: float,
    burst_load: float,
    phase_ratios: List[float],
    window_s: float,
    profile_name: str,
) -> RunArtifacts:
    rng = np.random.default_rng(seed)
    policy_cls = POLICIES[policy_name]
    policy = policy_cls(config, rng)
    env = Env(config, rng=rng, policy=policy)
    if hasattr(policy, "on_env_init"):
        policy.on_env_init(env)

    warmup = float(config["warmup_time"])
    sim_time = float(config["sim_time"])
    lam_base = float(config["arrival"]["base_total_rate"])
    stage_loads = [float(base_load), float(burst_load), float(base_load)]

    t = 0.0
    tids = 0

    arrivals: List[float] = []
    lats: List[float] = []
    clouds: List[float] = []
    backlog_time_total: List[float] = []
    backlog_work_total: List[float] = []
    queue_mean: List[float] = []
    queue_peak: List[float] = []
    queue_var: List[float] = []
    stage_ids: List[int] = []
    trig_eval_cum: List[int] = []
    trig_fire_cum: List[int] = []

    while t < sim_time:
        sid = stage_index(t, sim_time, phase_ratios)
        lam_now = lam_base * stage_loads[sid]
        dt = float(rng.exponential(1.0 / max(lam_now, 1e-12)))
        t += dt
        if t >= sim_time:
            break
        env.now = t
        task = env._sample_task(arrival_t=t)
        task.tid = tids
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

        sid_now = stage_index(t, sim_time, phase_ratios)
        arrivals.append(float(task.arrival_t))
        lats.append(float(task.finish_t - task.arrival_t))
        clouds.append(1.0 if task.is_cloud else 0.0)
        backlog_time_total.append(float(task.backlog_time_total))
        backlog_work_total.append(float(task.backlog_work_total))
        queue_mean.append(float(np.mean(w)))
        queue_peak.append(float(np.max(w)))
        queue_var.append(float(np.var(w)))
        stage_ids.append(int(sid_now))
        trig_eval_cum.append(int(getattr(policy, "_trig_eval_calls", 0)))
        trig_fire_cum.append(int(getattr(policy, "_auction_calls", 0)))

    arr = np.array(arrivals, dtype=float)
    lat = np.array(lats, dtype=float)
    cloud = np.array(clouds, dtype=float)
    btime = np.array(backlog_time_total, dtype=float)
    bwork = np.array(backlog_work_total, dtype=float)
    qmean = np.array(queue_mean, dtype=float)
    qpeak = np.array(queue_peak, dtype=float)
    qvar = np.array(queue_var, dtype=float)
    sid_arr = np.array(stage_ids, dtype=int)
    trig_eval_cum_arr = np.array(trig_eval_cum, dtype=int)
    trig_fire_cum_arr = np.array(trig_fire_cum, dtype=int)

    filt = arr >= warmup
    if int(filt.sum()) > 0:
        avg_latency = float(lat[filt].mean())
        p95_latency = float(np.quantile(lat[filt], 0.95))
        cloud_rate = float(cloud[filt].mean())
        throughput = float(filt.sum() / max(sim_time - warmup, 1e-12))
        queue_peak_global = float(qpeak[filt].max())
        queue_mean_global = float(qmean[filt].mean())
        queue_var_global = float(qvar[filt].mean())
    else:
        avg_latency = 0.0
        p95_latency = 0.0
        cloud_rate = 0.0
        throughput = 0.0
        queue_peak_global = 0.0
        queue_mean_global = 0.0
        queue_var_global = 0.0

    stage_rows = []
    for sid in [0, 1, 2]:
        m = (sid_arr == sid) & (arr >= warmup)
        if int(m.sum()) == 0:
            continue
        stage_rows.append({
            "profile": profile_name,
            "policy": policy_name,
            "seed": int(seed),
            "base_load": float(base_load),
            "burst_load": float(burst_load),
            "stage": int(sid),
            "avg_latency": float(lat[m].mean()),
            "p95_latency": float(np.quantile(lat[m], 0.95)),
            "cloud_rate": float(cloud[m].mean()),
            "queue_mean": float(qmean[m].mean()),
            "queue_peak": float(qpeak[m].max()),
            "queue_var": float(qvar[m].mean()),
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
        sid_w = stage_index(float(centers[i]), sim_time, phase_ratios)
        trig_rate = float("nan")
        if policy_name == "cosims_auction_B":
            idx = np.where(m)[0]
            i0 = int(idx[0])
            i1 = int(idx[-1])
            eval0 = int(trig_eval_cum_arr[i0 - 1]) if i0 > 0 else 0
            fire0 = int(trig_fire_cum_arr[i0 - 1]) if i0 > 0 else 0
            eval1 = int(trig_eval_cum_arr[i1])
            fire1 = int(trig_fire_cum_arr[i1])
            trig_rate = float((fire1 - fire0) / max(eval1 - eval0, 1))
        ts_rows.append({
            "profile": profile_name,
            "policy": policy_name,
            "seed": int(seed),
            "base_load": float(base_load),
            "burst_load": float(burst_load),
            "time": float(centers[i]),
            "stage": int(sid_w),
            "avg_latency_ts": float(lat[m].mean()),
            "p95_latency_ts": float(np.quantile(lat[m], 0.95)),
            "cloud_rate_ts": float(cloud[m].mean()),
            "queue_mean_ts": float(qmean[m].mean()),
            "queue_peak_ts": float(qpeak[m].max()),
            "queue_var_ts": float(qvar[m].mean()),
            "backlog_time_total_ts": float(btime[m].mean()),
            "backlog_work_total_ts": float(bwork[m].mean()),
            "trigger_rate_ts": trig_rate,
        })
    ts_df = pd.DataFrame(ts_rows)

    recovery_p95 = compute_recovery_time_burst(ts_df, metric_col="p95_latency_ts", decay_ratio=0.2)
    recovery_queue = compute_recovery_time_burst(ts_df, metric_col="queue_peak_ts", decay_ratio=0.2)
    burst_mask = (sid_arr == 1) & (arr >= warmup)
    burst_p95 = float(np.quantile(lat[burst_mask], 0.95)) if int(burst_mask.sum()) > 0 else float("nan")
    burst_cloud_rate = float(cloud[burst_mask].mean()) if int(burst_mask.sum()) > 0 else float("nan")
    pre_mask = (sid_arr == 0) & (arr >= warmup)
    rec_mask = (sid_arr == 2) & (arr >= warmup)
    pre_p95 = float(np.quantile(lat[pre_mask], 0.95)) if int(pre_mask.sum()) > 0 else float("nan")
    rec_p95 = float(np.quantile(lat[rec_mask], 0.95)) if int(rec_mask.sum()) > 0 else float("nan")
    burst_peak_queue = float(qpeak[burst_mask].max()) if int(burst_mask.sum()) > 0 else float("nan")
    pre_queue_mean = float(qmean[pre_mask].mean()) if int(pre_mask.sum()) > 0 else float("nan")
    rec_queue_mean = float(qmean[rec_mask].mean()) if int(rec_mask.sum()) > 0 else float("nan")
    p95_shock_ratio = float(burst_p95 / max(pre_p95, 1e-12)) if np.isfinite(burst_p95) and np.isfinite(pre_p95) else float("nan")
    queue_shock_ratio = float(burst_peak_queue / max(pre_queue_mean, 1e-12)) if np.isfinite(burst_peak_queue) and np.isfinite(pre_queue_mean) else float("nan")

    cost_stats = {}
    if hasattr(policy, "get_cost_stats"):
        try:
            cost_stats = policy.get_cost_stats()
        except Exception:
            cost_stats = {}

    summary_row = {
        "profile": profile_name,
        "policy": policy_name,
        "seed": int(seed),
        "base_load": float(base_load),
        "burst_load": float(burst_load),
        "throughput": throughput,
        "avg_latency": avg_latency,
        "p95_latency": p95_latency,
        "cloud_rate": cloud_rate,
        "completion": 1.0,
        "queue_peak": queue_peak_global,
        "queue_mean": queue_mean_global,
        "queue_var": queue_var_global,
        "burst_stage_p95": burst_p95,
        "burst_stage_cloud_rate": burst_cloud_rate,
        "pre_stage_p95": pre_p95,
        "recovery_stage_p95": rec_p95,
        "burst_peak_queue": burst_peak_queue,
        "pre_stage_queue_mean": pre_queue_mean,
        "recovery_stage_queue_mean": rec_queue_mean,
        "p95_shock_ratio": p95_shock_ratio,
        "queue_shock_ratio": queue_shock_ratio,
        "recovery_time_p95": recovery_p95,
        "recovery_time_queue": recovery_queue,
    }
    for k, v in cost_stats.items():
        summary_row[k] = v

    return RunArtifacts(summary_row=summary_row, ts_df=ts_df, stage_df=stage_df)


def compute_recovery_time_burst(ts_df: pd.DataFrame, metric_col: str, decay_ratio: float) -> float:
    if ts_df.empty or metric_col not in ts_df.columns:
        return float("nan")
    burst = ts_df[ts_df["stage"] == 1][["time", metric_col]].dropna().sort_values("time")
    rec = ts_df[ts_df["stage"] == 2][["time", metric_col]].dropna().sort_values("time")
    if burst.empty or rec.empty:
        return float("nan")
    peak = float(max(burst[metric_col].max(), rec[metric_col].max()))
    rec_vals = rec[metric_col].to_numpy(dtype=float)
    rec_tim = rec["time"].to_numpy(dtype=float)
    tail_n = max(1, int(0.3 * rec_vals.size))
    steady = float(np.mean(rec_vals[-tail_n:]))
    threshold = float(steady + decay_ratio * max(peak - steady, 0.0))
    t0 = float(rec_tim.min())
    vals = rec_vals
    tim = rec_tim
    idx = np.where(vals <= threshold)[0]
    if idx.size == 0:
        return float(tim[-1] - t0)
    return float(max(tim[int(idx[0])] - t0, 0.0))


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
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def make_plots(ts_mean_std: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    if ts_mean_std.empty:
        return
    for profile in sorted(ts_mean_std["profile"].dropna().unique().tolist()):
        d0 = ts_mean_std[ts_mean_std["profile"] == profile].copy()

        plt.figure(figsize=(8, 4))
        for p, g in d0.groupby("policy"):
            g = g.sort_values("time")
            plt.plot(g["time"], g["p95_latency_ts_mean"], label=str(p))
        plt.xlabel("Time (s)")
        plt.ylabel("Rolling P95 Latency (s)")
        plt.title(f"Burst Rolling P95 ({profile})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"rolling_p95_over_time_{profile}.png"))
        plt.close()

        plt.figure(figsize=(8, 4))
        for p, g in d0.groupby("policy"):
            g = g.sort_values("time")
            plt.plot(g["time"], g["queue_peak_ts_mean"], label=str(p))
        plt.xlabel("Time (s)")
        plt.ylabel("Queue Peak")
        plt.title(f"Burst Queue Peak ({profile})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"queue_peak_over_time_{profile}.png"))
        plt.close()

        b = d0[d0["policy"] == "cosims_auction_B"].sort_values("time")
        if not b.empty:
            plt.figure(figsize=(8, 4))
            plt.plot(b["time"], b["trigger_rate_ts_mean"])
            plt.xlabel("Time (s)")
            plt.ylabel("Trigger Rate")
            plt.title(f"Burst Trigger Rate (B, {profile})")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"trigger_rate_over_time_B_{profile}.png"))
            plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/auction_B_recommended.json")
    ap.add_argument("--out_dir", type=str, default="outputs/core_scenarios/burst")
    ap.add_argument("--base_load", type=float, default=1.0)
    ap.add_argument("--burst_loads", type=str, default="1.4,1.6")
    ap.add_argument("--phase_ratios", type=str, default="0.3,0.4,0.3")
    ap.add_argument("--seeds", type=str, default="0-4")
    ap.add_argument("--policies", type=str, default="cap_greedy,sqlf,cosims_auction_B,cosims_sinkhorn,greedy_latency")
    ap.add_argument("--window_s", type=float, default=20.0)
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

    burst_loads = parse_floats(args.burst_loads)
    phase_ratios = parse_floats(args.phase_ratios)
    if len(phase_ratios) != 3 or abs(sum(phase_ratios) - 1.0) > 1e-6:
        raise ValueError("phase_ratios must be 3 numbers summing to 1.0")
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

    for burst_load in burst_loads:
        profile = f"burst_{args.base_load}_{burst_load}_{args.base_load}"
        for seed in seeds:
            for policy in policies:
                art = run_burst_once(
                    config=config,
                    policy_name=policy,
                    seed=seed,
                    base_load=float(args.base_load),
                    burst_load=float(burst_load),
                    phase_ratios=phase_ratios,
                    window_s=float(args.window_s),
                    profile_name=profile,
                )
                summary_rows.append(art.summary_row)
                if not art.ts_df.empty:
                    ts_frames.append(art.ts_df)
                if not art.stage_df.empty:
                    stage_frames.append(art.stage_df)
                print(f"[OK] profile={profile} seed={seed} policy={policy}")

    df_summary_runs = pd.DataFrame(summary_rows).sort_values(["profile", "policy", "seed"])
    df_summary_runs.to_csv(os.path.join(raw_dir, "summary_runs.csv"), index=False)

    summary_keys = [c for c in df_summary_runs.columns if c not in {"profile", "policy", "seed", "base_load", "burst_load"}]
    df_summary_mean_std = agg_mean_std(df_summary_runs, keys=summary_keys, group_cols=["profile", "base_load", "burst_load", "policy"])
    df_summary_mean_std.to_csv(os.path.join(args.out_dir, "summary_mean_std.csv"), index=False)

    df_ts_runs = pd.concat(ts_frames, axis=0, ignore_index=True) if len(ts_frames) > 0 else pd.DataFrame()
    if not df_ts_runs.empty:
        df_ts_runs = df_ts_runs.sort_values(["profile", "policy", "seed", "time"])
        df_ts_runs.to_csv(os.path.join(raw_dir, "timeseries_runs.csv"), index=False)
        ts_keys = [c for c in df_ts_runs.columns if c not in {"profile", "policy", "seed", "base_load", "burst_load", "time", "stage"}]
        df_ts_mean_std = agg_mean_std(df_ts_runs, keys=ts_keys, group_cols=["profile", "base_load", "burst_load", "policy", "time", "stage"])
        df_ts_mean_std.to_csv(os.path.join(args.out_dir, "timeseries_mean_std.csv"), index=False)
        make_plots(df_ts_mean_std, fig_dir)
    else:
        df_ts_mean_std = pd.DataFrame()

    df_stage_runs = pd.concat(stage_frames, axis=0, ignore_index=True) if len(stage_frames) > 0 else pd.DataFrame()
    if not df_stage_runs.empty:
        df_stage_runs = df_stage_runs.sort_values(["profile", "policy", "seed", "stage"])
        df_stage_runs.to_csv(os.path.join(raw_dir, "stage_runs.csv"), index=False)
        stage_keys = [c for c in df_stage_runs.columns if c not in {"profile", "policy", "seed", "base_load", "burst_load", "stage"}]
        df_stage_mean_std = agg_mean_std(df_stage_runs, keys=stage_keys, group_cols=["profile", "base_load", "burst_load", "policy", "stage"])
        df_stage_mean_std.to_csv(os.path.join(args.out_dir, "stage_mean_std.csv"), index=False)

    rec_cols = ["recovery_time_p95", "recovery_time_queue"]
    df_recovery = df_summary_runs[["profile", "base_load", "burst_load", "policy", "seed"] + rec_cols].copy()
    df_recovery.to_csv(os.path.join(raw_dir, "recovery_runs.csv"), index=False)
    df_recovery_mean_std = agg_mean_std(df_recovery, keys=rec_cols, group_cols=["profile", "base_load", "burst_load", "policy"])
    df_recovery_mean_std.to_csv(os.path.join(args.out_dir, "recovery_mean_std.csv"), index=False)

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "config_path": args.config,
        "base_load": float(args.base_load),
        "burst_loads": burst_loads,
        "phase_ratios": phase_ratios,
        "seeds": seeds,
        "policies": policies,
        "window_s": float(args.window_s),
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
