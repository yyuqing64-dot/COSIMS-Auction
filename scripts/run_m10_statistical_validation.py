from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import compute_metrics
from eval.runner import POLICIES, _load_trace_arrays, _run_trace
from sim.env import Env


T_CRIT_975 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}


def parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_seeds(value: str) -> list[int]:
    if "-" in value:
        a, b = value.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def final_paper_config(config: dict) -> dict:
    cfg = json.loads(json.dumps(config))
    cosims = cfg.setdefault("cosims", {})
    cosims["topk"] = 3
    cosims["lam"] = 0.1
    cosims["max_iter"] = 35
    event = cosims.setdefault("event", {})
    event["misfit_delta"] = 0.5
    event["wait_tau"] = 6e11
    event["slope_th"] = 5e11
    event["rate_factor"] = 10.0
    cosims["event"] = event
    cfg["cosims"] = cosims
    return cfg


def run_one(config: dict, policy_name: str, seed: int, load: float, trace_arrays=None) -> dict:
    rng = np.random.default_rng(seed)
    policy = POLICIES[policy_name](config, rng)
    env = Env(config, rng=rng, policy=policy)
    if hasattr(policy, "on_env_init"):
        policy.on_env_init(env)

    sim_time_eval = float(config["sim_time"])
    if trace_arrays is None:
        tasks = env.run(load_scale=load)
    else:
        arr_t, src_idx, cpu_arr, data_arr, trace_sim_time = trace_arrays
        tasks = _run_trace(env, arr_t, src_idx, cpu_arr, data_arr)
        sim_time_eval = float(trace_sim_time)

    stats = compute_metrics(tasks, warmup=float(config["warmup_time"]), sim_time=sim_time_eval)
    if hasattr(policy, "get_cost_stats"):
        try:
            stats.update(policy.get_cost_stats())
        except Exception:
            pass

    stats["policy"] = policy_name
    stats["seed"] = int(seed)
    stats["load"] = float(load)
    stats["n_nodes"] = int(config.get("n_nodes", getattr(env, "n", 0)))
    stats["topk"] = int(config.get("cosims", {}).get("topk", 0))
    return stats


def summarize(raw: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric_cols = [
        c
        for c in raw.columns
        if c not in set(group_cols + ["seed"])
        and pd.api.types.is_numeric_dtype(raw[c])
    ]
    rows = []
    for keys, g in raw.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        n = int(g["seed"].nunique())
        row["n"] = n
        tcrit = T_CRIT_975.get(n, 1.96)
        for col in numeric_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            if vals.empty:
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            se = std / np.sqrt(len(vals)) if len(vals) > 0 else 0.0
            ci95 = float(tcrit * se) if len(vals) > 1 else 0.0
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_ci95"] = ci95
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(out_dir: Path, name: str, raw: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    target = out_dir / name
    target.mkdir(parents=True, exist_ok=True)
    raw_path = target / "raw_by_seed.csv"
    summary_path = target / "summary_mean_std_ci.csv"
    raw.to_csv(raw_path, index=False)
    summary = summarize(raw, group_cols)
    summary.to_csv(summary_path, index=False)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/auction_B_recommended_sqlf_cloud.json")
    ap.add_argument("--out-dir", default="results/experiments/statistical_validation_m10_rerun")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--synthetic-loads", default="0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6")
    ap.add_argument("--mechanism-loads", default="1.0,1.2,1.4")
    ap.add_argument("--trace-load-factors", default="0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6")
    ap.add_argument(
        "--scenarios",
        default="synthetic_main,trace_main,trace_tradeoff,mechanism_overhead",
        help="Comma-separated subset: synthetic_main, trace_main, trace_tradeoff, mechanism_overhead.",
    )
    ap.add_argument(
        "--trace-policies",
        default="rr,greedy_latency,sqlf,cap_greedy,cosims_sinkhorn,cosims_auction_B",
        help="Comma-separated policies for trace_main.",
    )
    ap.add_argument(
        "--trace-manifest",
        default="data/trace_inputs/arrivals_h77_load_sweep_0.8_1.6_step0.1_manifest.csv",
    )
    args = ap.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        base_config = json.load(f)
    config = final_paper_config(base_config)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    synthetic_loads = parse_csv_floats(args.synthetic_loads)
    mechanism_loads = parse_csv_floats(args.mechanism_loads)
    trace_load_factors = parse_csv_floats(args.trace_load_factors)
    scenarios = set(parse_csv_strings(args.scenarios))
    trace_policies = parse_csv_strings(args.trace_policies)

    policies_main = [
        "rr",
        "greedy_latency",
        "sqlf",
        "cap_greedy",
        "cosims_sinkhorn",
        "cosims_auction_B",
    ]
    policies_mechanism = [
        "cosims_sinkhorn",
        "cosims_auction_A",
        "cosims_auction_B",
    ]

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": seeds,
        "synthetic_loads": synthetic_loads,
        "mechanism_loads": mechanism_loads,
        "trace_load_factors": trace_load_factors,
        "policies_main": policies_main,
        "trace_policies": trace_policies,
        "policies_mechanism": policies_mechanism,
        "paper_thresholds": config.get("cosims", {}).get("event", {}),
        "cosims": config.get("cosims", {}),
        "env": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
    }

    synthetic_raw = pd.DataFrame()
    synthetic_summary = pd.DataFrame()
    if "synthetic_main" in scenarios:
        synthetic_rows = []
        for load in synthetic_loads:
            for policy_name in policies_main:
                for seed in seeds:
                    row = run_one(config, policy_name, seed, load)
                    row["scenario"] = "synthetic_main"
                    synthetic_rows.append(row)
        synthetic_raw = pd.DataFrame(synthetic_rows)
        synthetic_summary = write_outputs(
            out_dir, "synthetic_main", synthetic_raw, ["scenario", "load", "policy"]
        )

    trace_raw = pd.DataFrame()
    trace_summary = pd.DataFrame()
    if "trace_main" in scenarios:
        trace_target = out_dir / "trace_main"
        trace_target.mkdir(parents=True, exist_ok=True)
        trace_checkpoint = trace_target / "raw_by_seed.checkpoint.csv"
        if trace_checkpoint.exists():
            trace_checkpoint.unlink()

        trace_manifest = pd.read_csv(ROOT / args.trace_manifest)
        trace_lookup = {
            float(r["trace_load_factor"]): str(r["file"])
            for _, r in trace_manifest.iterrows()
        }
        trace_cache = {}
        trace_rows = []
        total_trace_runs = len(trace_load_factors) * len(trace_policies) * len(seeds)
        done_trace_runs = 0
        for factor in trace_load_factors:
            trace_file = trace_lookup[float(factor)]
            if trace_file not in trace_cache:
                trace_cache[trace_file] = _load_trace_arrays(trace_file, int(config["n_nodes"]))
            arrays = trace_cache[trace_file]
            for policy_name in trace_policies:
                for seed in seeds:
                    row = run_one(config, policy_name, seed, load=1.0, trace_arrays=arrays)
                    row["scenario"] = "trace_main"
                    row["trace_load_factor"] = float(factor)
                    trace_rows.append(row)
                    pd.DataFrame([row]).to_csv(
                        trace_checkpoint,
                        mode="a",
                        header=not trace_checkpoint.exists(),
                        index=False,
                    )
                    done_trace_runs += 1
                    print(
                        f"[trace] {done_trace_runs}/{total_trace_runs} "
                        f"load={factor} policy={policy_name} seed={seed}",
                        flush=True,
                    )
        trace_raw = pd.DataFrame(trace_rows)
        trace_summary = write_outputs(
            out_dir, "trace_main", trace_raw, ["scenario", "trace_load_factor", "policy"]
        )

    tradeoff_raw = pd.DataFrame()
    tradeoff_summary = pd.DataFrame()
    if "trace_tradeoff" in scenarios:
        if trace_raw.empty:
            trace_path = out_dir / "trace_main" / "raw_by_seed.csv"
            if not trace_path.exists():
                raise FileNotFoundError(
                    f"trace_tradeoff requires trace_main output at {trace_path}"
                )
            trace_raw = pd.read_csv(trace_path)
        tradeoff_raw = trace_raw[trace_raw["policy"].isin(["cap_greedy", "cosims_auction_B"])].copy()
        tradeoff_raw["scenario"] = "trace_tradeoff"
        tradeoff_summary = write_outputs(
            out_dir, "trace_tradeoff", tradeoff_raw, ["scenario", "trace_load_factor", "policy"]
        )

    mechanism_raw = pd.DataFrame()
    mechanism_summary = pd.DataFrame()
    if "mechanism_overhead" in scenarios:
        mech_rows = []
        for load in mechanism_loads:
            for policy_name in policies_mechanism:
                for seed in seeds:
                    row = run_one(config, policy_name, seed, load)
                    row["scenario"] = "mechanism_overhead"
                    mech_rows.append(row)
        mechanism_raw = pd.DataFrame(mech_rows)
        mechanism_summary = write_outputs(
            out_dir, "mechanism_overhead", mechanism_raw, ["scenario", "load", "policy"]
        )

    (out_dir / "config_snapshot.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[OK] wrote M10 statistical-validation outputs")
    print(out_dir)
    print("synthetic rows:", len(synthetic_raw), "summary:", synthetic_summary.shape)
    print("trace rows:", len(trace_raw), "summary:", trace_summary.shape)
    print("tradeoff rows:", len(tradeoff_raw), "summary:", tradeoff_summary.shape)
    print("mechanism rows:", len(mechanism_raw), "summary:", mechanism_summary.shape)


if __name__ == "__main__":
    main()
