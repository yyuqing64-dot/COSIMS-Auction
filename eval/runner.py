from __future__ import annotations

import argparse
import json
import os
import sys
import platform
import hashlib
import time

import numpy as np
import pandas as pd
import matplotlib

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(__file__)))
from sim.env import Env
from sim.task import Task
from eval.metrics import compute_metrics, compute_timeseries

from policies.greedy_latency import GreedyLatency
from policies.rr import RoundRobin
from policies.cap_greedy import CapAwareGreedy
from policies.sqlf import ShortestQueueFirst
from policies.cosims_sinkhorn import COSIMSSinkhorn
from policies.cosims_auction_A import COSIMSAuctionA
from policies.cosims_auction_B import COSIMSAuctionB


POLICIES = {
    "greedy_latency": GreedyLatency,
    "rr": RoundRobin,
    "cap_greedy": CapAwareGreedy,
    "sqlf": ShortestQueueFirst,
    "cosims_sinkhorn": COSIMSSinkhorn,
    "cosims_auction_A": COSIMSAuctionA,
    "cosims_auction_B": COSIMSAuctionB,
}


def run_once(config: dict, policy_cls, seed: int, load: float) -> dict:
    rng = np.random.default_rng(seed)
    policy = policy_cls(config, rng)
    env = Env(config, rng=rng, policy=policy)

    if hasattr(policy, "on_env_init"):
        policy.on_env_init(env)

    tasks = env.run(load_scale=load)
    stats = compute_metrics(tasks, warmup=float(config["warmup_time"]), sim_time=float(config["sim_time"]))
    if hasattr(policy, "get_cost_stats"):
        try:
            stats.update(policy.get_cost_stats())
        except Exception:
            pass
    # add scale info for plotting if present
    stats["n_nodes"] = int(config.get("n_nodes", getattr(env, "n", 0)))
    cosims_cfg = config.get("cosims", {})
    if "topk" in cosims_cfg:
        stats["topk"] = int(cosims_cfg.get("topk"))
    return stats


def _run_trace(env: Env, arrival_t: np.ndarray, src: np.ndarray, cpu_demand: np.ndarray, data_mb: np.ndarray):
    tasks = []
    for i in range(arrival_t.shape[0]):
        t = float(arrival_t[i])
        env.now = t
        task = Task(
            tid=env._tid,
            src=int(src[i]),
            arrival_t=t,
            cpu_demand=float(cpu_demand[i]),
            data_mb=float(data_mb[i]),
        )
        env._tid += 1

        W = np.maximum(np.asarray(env.available_time, dtype=float) - t, 0.0)
        task.backlog_time_total = float(W.sum())
        task.backlog_work_total = float(np.dot(W, env.xi))

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
        tasks.append(task)
    return tasks


def _load_trace_arrays(trace_file: str, n_nodes: int):
    df = pd.read_csv(trace_file)
    need = ["arrival_t", "src", "cpu_demand", "data_mb"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Trace missing required column: {c} in {trace_file}")
    t = pd.to_numeric(df["arrival_t"], errors="coerce").to_numpy(dtype=float)
    s_raw = pd.to_numeric(df["src"], errors="coerce").to_numpy(dtype=float)
    cpu = pd.to_numeric(df["cpu_demand"], errors="coerce").to_numpy(dtype=float)
    data = pd.to_numeric(df["data_mb"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(t) & np.isfinite(s_raw) & np.isfinite(cpu) & np.isfinite(data)
    ok = ok & (cpu > 0.0) & (data > 0.0)
    t = t[ok]
    s_raw = s_raw[ok]
    cpu = cpu[ok]
    data = data[ok]
    order = np.argsort(t, kind="mergesort")
    t = t[order]
    s_raw = s_raw[order]
    cpu = cpu[order]
    data = data[order]
    uniq = np.unique(s_raw.astype(np.int64))
    src_map = {int(v): int(i % n_nodes) for i, v in enumerate(uniq.tolist())}
    src = np.array([src_map[int(v)] for v in s_raw.astype(np.int64)], dtype=np.int64)
    sim_time_eval = float(t.max()) if t.size > 0 else 0.0
    return t, src, cpu, data, sim_time_eval

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.json")
    ap.add_argument("--out", type=str, default="results/experiments/manual_run/summary_mean_std.csv")
    ap.add_argument("--loads", type=str, default="0.4,0.6,0.8,1.0,1.2")
    ap.add_argument("--seeds", type=str, default="0-9")
    ap.add_argument("--policies", type=str, default="greedy_latency,rr,cap_greedy,sqlf")
    ap.add_argument("--sweep_nodes", type=str, default="", help="Comma list, e.g., 20,40,80")
    ap.add_argument("--sweep_topk", type=str, default="", help="Comma list for COSIMS topk, e.g., 5,10,20")
    ap.add_argument("--sweep_alpha", type=str, default="", help="Comma list for COSIMS alpha")
    ap.add_argument("--sweep_beta", type=str, default="", help="Comma list for COSIMS beta")
    ap.add_argument("--sweep_lam", type=str, default="", help="Comma list for COSIMS lam")
    ap.add_argument("--sweep_misfit_delta", type=str, default="", help="Comma list for event misfit_delta")
    ap.add_argument("--sweep_wait_tau", type=str, default="", help="Comma list for event wait_tau")
    ap.add_argument("--sweep_slope_th", type=str, default="", help="Comma list for event slope_th")
    ap.add_argument("--sweep_rate_factor", type=str, default="", help="Comma list for arrival trigger rate_factor")
    ap.add_argument("--ts_window", type=float, default=0.0, help="If >0, also compute time-series metrics with this window")
    ap.add_argument("--out_ts", type=str, default="", help="Path to write time-series CSV if ts_window>0")
    ap.add_argument("--trace_load_factors", type=str, default="", help="Optional label sweep for trace-based load factors, e.g., 0.8,1.0,1.2")
    ap.add_argument("--trace_manifest", type=str, default="", help="CSV with trace_load_factor and file columns")
    ap.add_argument("--trace_factor_col", type=str, default="trace_load_factor")
    ap.add_argument("--trace_file_col", type=str, default="file")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    # parse loads
    load_scales = [float(x.strip()) for x in args.loads.split(",") if x.strip()]

    # parse seeds like 0-9 or 0,1,2
    if "-" in args.seeds:
        a, b = args.seeds.split("-")
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    policy_names = [x.strip() for x in args.policies.split(",") if x.strip()]
    for pn in policy_names:
        if pn not in POLICIES:
            raise ValueError(f"Unknown policy: {pn}. Available: {list(POLICIES.keys())}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # parse sweeps
    sweep_nodes = [int(x.strip()) for x in args.sweep_nodes.split(",") if x.strip()] if args.sweep_nodes else [int(config.get("n_nodes", 0))]
    sweep_topk = [int(x.strip()) for x in args.sweep_topk.split(",") if x.strip()] if args.sweep_topk else [int(config.get("cosims", {}).get("topk", 0))] if "cosims" in config else [0]
    sweep_alpha = [float(x.strip()) for x in args.sweep_alpha.split(",") if x.strip()] if args.sweep_alpha else [None]
    sweep_beta = [float(x.strip()) for x in args.sweep_beta.split(",") if x.strip()] if args.sweep_beta else [None]
    sweep_lam = [float(x.strip()) for x in args.sweep_lam.split(",") if x.strip()] if args.sweep_lam else [None]
    sweep_misfit_delta = [float(x.strip()) for x in args.sweep_misfit_delta.split(",") if x.strip()] if args.sweep_misfit_delta else [None]
    sweep_wait_tau = [float(x.strip()) for x in args.sweep_wait_tau.split(",") if x.strip()] if args.sweep_wait_tau else [None]
    sweep_slope_th = [float(x.strip()) for x in args.sweep_slope_th.split(",") if x.strip()] if args.sweep_slope_th else [None]
    sweep_rate_factor = [float(x.strip()) for x in args.sweep_rate_factor.split(",") if x.strip()] if args.sweep_rate_factor else [None]
    sweep_trace_load_factors = [float(x.strip()) for x in args.trace_load_factors.split(",") if x.strip()] if args.trace_load_factors else [None]
    trace_mode = bool(args.trace_manifest)
    trace_lookup = {}
    if trace_mode:
        mf = pd.read_csv(args.trace_manifest)
        if args.trace_factor_col not in mf.columns or args.trace_file_col not in mf.columns:
            raise ValueError(f"Manifest must contain columns: {args.trace_factor_col}, {args.trace_file_col}")
        for _, r in mf.iterrows():
            trace_lookup[float(r[args.trace_factor_col])] = str(r[args.trace_file_col])
        if args.trace_load_factors:
            sweep_trace_load_factors = [float(x.strip()) for x in args.trace_load_factors.split(",") if x.strip()]
        else:
            sweep_trace_load_factors = sorted(trace_lookup.keys())
        load_scales = [1.0]
    trace_cache = {}

    def _scaled_config(base_cfg: dict, n_nodes: int, topk: int | None, alpha: float | None, beta: float | None, lam: float | None, misfit_delta: float | None, wait_tau: float | None, slope_th: float | None, rate_factor: float | None) -> dict:
        cfg = json.loads(json.dumps(base_cfg))
        cfg["n_nodes"] = int(n_nodes)
        # scale xi to length n_nodes by tiling existing pattern
        xi_base = cfg.get("capacity", {}).get("xi", [])
        if not xi_base or len(xi_base) != int(n_nodes):
            if xi_base:
                import math
                times = math.ceil(n_nodes / len(xi_base))
                tiled = (xi_base * times)[:n_nodes]
            else:
                tiled = [1] * n_nodes
            cfg["capacity"]["xi"] = tiled
        # set topk if provided
        if topk and topk > 0:
            cosims = cfg.get("cosims", {})
            cosims["topk"] = int(topk)
            cfg["cosims"] = cosims
        if alpha is not None:
            cosims = cfg.get("cosims", {})
            cosims["alpha"] = float(alpha)
            cfg["cosims"] = cosims
        if beta is not None:
            cosims = cfg.get("cosims", {})
            cosims["beta"] = float(beta)
            cfg["cosims"] = cosims
        if lam is not None:
            cosims = cfg.get("cosims", {})
            cosims["lam"] = float(lam)
            cfg["cosims"] = cosims
        if misfit_delta is not None:
            cosims = cfg.get("cosims", {})
            evt = cosims.get("event", {})
            evt["misfit_delta"] = float(misfit_delta)
            cosims["event"] = evt
            cfg["cosims"] = cosims
        if wait_tau is not None:
            cosims = cfg.get("cosims", {})
            evt = cosims.get("event", {})
            evt["wait_tau"] = float(wait_tau)
            cosims["event"] = evt
            cfg["cosims"] = cosims
        if slope_th is not None:
            cosims = cfg.get("cosims", {})
            evt = cosims.get("event", {})
            evt["slope_th"] = float(slope_th)
            cosims["event"] = evt
            cfg["cosims"] = cosims
        if rate_factor is not None:
            cosims = cfg.get("cosims", {})
            evt = cosims.get("event", {})
            evt["rate_factor"] = float(rate_factor)
            cosims["event"] = evt
            cfg["cosims"] = cosims
        return cfg

    rows = []
    ts_rows = []
    for n_nodes in sweep_nodes:
        for topk in (sweep_topk if any(sweep_topk) else [None]):
            for alpha in (sweep_alpha if any(x is not None for x in sweep_alpha) else [None]):
                for beta in (sweep_beta if any(x is not None for x in sweep_beta) else [None]):
                    for lam in (sweep_lam if any(x is not None for x in sweep_lam) else [None]):
                        for wait_tau in (sweep_wait_tau if any(x is not None for x in sweep_wait_tau) else [None]):
                            for slope_th in (sweep_slope_th if any(x is not None for x in sweep_slope_th) else [None]):
                                for rate_factor in (sweep_rate_factor if any(x is not None for x in sweep_rate_factor) else [None]):
                                    for misfit_delta in (sweep_misfit_delta if any(x is not None for x in sweep_misfit_delta) else [None]):
                                        for trace_load_factor in (sweep_trace_load_factors if any(x is not None for x in sweep_trace_load_factors) else [None]):
                                            cfg_scaled = _scaled_config(config, n_nodes=n_nodes, topk=topk, alpha=alpha, beta=beta, lam=lam, misfit_delta=misfit_delta, wait_tau=wait_tau, slope_th=slope_th, rate_factor=rate_factor)
                                            for load in load_scales:
                                                for pn in policy_names:
                                                    policy_cls = POLICIES[pn]
                                                    stats_list = []
                                                    ts_list = []
                                                    for s in seeds:
                                                        rng = np.random.default_rng(s)
                                                        policy = policy_cls(cfg_scaled, rng)
                                                        env = Env(cfg_scaled, rng=rng, policy=policy)
                                                        if hasattr(policy, "on_env_init"):
                                                            policy.on_env_init(env)
                                                        sim_time_eval = float(cfg_scaled["sim_time"])
                                                        if trace_mode:
                                                            if trace_load_factor is None:
                                                                raise ValueError("trace_load_factor is required in trace mode")
                                                            trace_file = trace_lookup.get(float(trace_load_factor))
                                                            if trace_file is None:
                                                                raise ValueError(f"No trace file for factor={trace_load_factor}")
                                                            ck = (trace_file, int(cfg_scaled["n_nodes"]))
                                                            if ck not in trace_cache:
                                                                trace_cache[ck] = _load_trace_arrays(trace_file, int(cfg_scaled["n_nodes"]))
                                                            arr_t, src_idx, cpu_arr, data_arr, trace_sim_time = trace_cache[ck]
                                                            tasks = _run_trace(env, arr_t, src_idx, cpu_arr, data_arr)
                                                            sim_time_eval = trace_sim_time
                                                        else:
                                                            tasks = env.run(load_scale=load)
                                                        stats = compute_metrics(tasks, warmup=float(cfg_scaled["warmup_time"]), sim_time=sim_time_eval)
                                                        if hasattr(policy, "get_cost_stats"):
                                                            try:
                                                                stats.update(policy.get_cost_stats())
                                                            except Exception:
                                                                pass
                                                        stats["n_nodes"] = int(cfg_scaled.get("n_nodes", getattr(env, "n", 0)))
                                                        cosims_cfg = cfg_scaled.get("cosims", {})
                                                        if "topk" in cosims_cfg:
                                                            stats["topk"] = int(cosims_cfg.get("topk"))
                                                        stats_list.append(stats)
                                                        if args.ts_window and args.ts_window > 0:
                                                            ts = compute_timeseries(tasks, warmup=float(cfg_scaled["warmup_time"]), sim_time=sim_time_eval, window_s=float(args.ts_window))
                                                            ts_list.append(ts)

                                                    keys = stats_list[0].keys()
                                                    mean = {k: float(np.mean([d[k] for d in stats_list])) for k in keys}
                                                    std = {k: float(np.std([d[k] for d in stats_list], ddof=1)) for k in keys}

                                                    row = {"load": load, "policy": pn, "n_nodes": int(n_nodes)}
                                                    if trace_load_factor is not None:
                                                        row["trace_load_factor"] = float(trace_load_factor)
                                                    if topk:
                                                        row["topk"] = int(topk)
                                                    if alpha is not None:
                                                        row["alpha"] = float(alpha)
                                                    if beta is not None:
                                                        row["beta"] = float(beta)
                                                    if lam is not None:
                                                        row["lam"] = float(lam)
                                                    if misfit_delta is not None:
                                                        row["misfit_delta"] = float(misfit_delta)
                                                    if wait_tau is not None:
                                                        row["wait_tau"] = float(wait_tau)
                                                    if slope_th is not None:
                                                        row["slope_th"] = float(slope_th)
                                                    if rate_factor is not None:
                                                        row["rate_factor"] = float(rate_factor)
                                                    for k in keys:
                                                        row[f"{k}_mean"] = mean[k]
                                                        row[f"{k}_std"] = std[k]
                                                    rows.append(row)
                                                    if args.ts_window and args.ts_window > 0 and len(ts_list) > 0:
                                                        times = ts_list[0]["time"]
                                                        def agg(col):
                                                            arrs = [t[col] for t in ts_list]
                                                            stacked = np.vstack(arrs)
                                                            return np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0, ddof=1)
                                                        cols = ["p95_latency_ts", "throughput_ts", "cloud_rate_ts", "backlog_time_total_ts", "backlog_work_total_ts"]
                                                        m_dict = {c: agg(c)[0] for c in cols}
                                                        s_dict = {c: agg(c)[1] for c in cols}
                                                        for i, tm in enumerate(times):
                                                            ts_row = {"time": float(tm), "load": load, "policy": pn, "n_nodes": int(n_nodes)}
                                                            if trace_load_factor is not None:
                                                                ts_row["trace_load_factor"] = float(trace_load_factor)
                                                            if topk:
                                                                ts_row["topk"] = int(topk)
                                                            if alpha is not None:
                                                                ts_row["alpha"] = float(alpha)
                                                            if beta is not None:
                                                                ts_row["beta"] = float(beta)
                                                            if lam is not None:
                                                                ts_row["lam"] = float(lam)
                                                            if misfit_delta is not None:
                                                                ts_row["misfit_delta"] = float(misfit_delta)
                                                            if wait_tau is not None:
                                                                ts_row["wait_tau"] = float(wait_tau)
                                                            if slope_th is not None:
                                                                ts_row["slope_th"] = float(slope_th)
                                                            if rate_factor is not None:
                                                                ts_row["rate_factor"] = float(rate_factor)
                                                            for c in cols:
                                                                ts_row[f"{c}_mean"] = float(m_dict[c][i])
                                                                ts_row[f"{c}_std"] = float(s_dict[c][i])
                                                            ts_rows.append(ts_row)

    df = pd.DataFrame(rows)
    sort_cols = ["load", "policy"]
    if "trace_load_factor" in df.columns:
        sort_cols = ["trace_load_factor", "policy", "load"]
    df = df.sort_values(sort_cols)
    df.to_csv(args.out, index=False)
    print(f"[OK] wrote: {args.out}")
    print(df.head(10).to_string(index=False))
    if args.ts_window and args.ts_window > 0:
        out_ts = args.out_ts if args.out_ts else os.path.join(os.path.dirname(args.out), "timeseries.csv")
        df_ts = pd.DataFrame(ts_rows)
        ts_sort_cols = ["time", "policy"]
        if "trace_load_factor" in df_ts.columns:
            ts_sort_cols = ["trace_load_factor", "time", "policy"]
        df_ts = df_ts.sort_values(ts_sort_cols)
        df_ts.to_csv(out_ts, index=False)
        print(f"[OK] wrote: {out_ts}")

    out_dir = os.path.dirname(args.out)
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "config_path": args.config,
        "loads": load_scales,
        "seeds": seeds,
        "policies": policy_names,
        "sweep_nodes": sweep_nodes,
        "sweep_topk": sweep_topk,
        "trace_load_factors": sweep_trace_load_factors,
        "trace_manifest": args.trace_manifest,
        "trace_factor_col": args.trace_factor_col,
        "trace_file_col": args.trace_file_col,
        "ts_window": float(args.ts_window or 0.0),
        "env": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
    }
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_text = f.read()
    meta["config_sha256"] = hashlib.sha256(cfg_text.encode("utf-8")).hexdigest()
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
        f.write(cfg_text)

if __name__ == "__main__":
    main()
