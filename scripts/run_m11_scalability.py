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
from eval.runner import POLICIES
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


class TimingPolicy:
    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", inner.__class__.__name__)
        self._time_sum = 0.0
        self._calls = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def on_env_init(self, env):
        if hasattr(self.inner, "on_env_init"):
            return self.inner.on_env_init(env)
        return None

    def select_dest(self, task, env):
        t0 = time.perf_counter()
        dest = self.inner.select_dest(task, env)
        t1 = time.perf_counter()
        self._time_sum += t1 - t0
        self._calls += 1
        return dest

    def get_cost_stats(self) -> dict:
        stats = {}
        if hasattr(self.inner, "get_cost_stats"):
            try:
                stats.update(self.inner.get_cost_stats())
            except Exception:
                pass
        measured = self._time_sum / self._calls if self._calls > 0 else 0.0
        stats["measured_decision_time_ms_mean"] = float(measured * 1000.0)
        stats["measured_decision_calls"] = int(self._calls)
        return stats


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_seeds(value: str) -> list[int]:
    if "-" in value:
        a, b = value.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return parse_csv_ints(value)


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
    return cfg


def scaled_config(base_config: dict, n_nodes: int) -> dict:
    cfg = final_paper_config(base_config)
    cfg["n_nodes"] = int(n_nodes)

    base_xi = list(base_config["capacity"]["xi"])
    repeats = (n_nodes + len(base_xi) - 1) // len(base_xi)
    cfg["capacity"]["xi"] = (base_xi * repeats)[:n_nodes]
    return cfg


def run_one(config: dict, policy_name: str, seed: int, load: float) -> dict:
    rng = np.random.default_rng(seed)
    policy = TimingPolicy(POLICIES[policy_name](config, rng))
    env = Env(config, rng=rng, policy=policy)
    if hasattr(policy, "on_env_init"):
        policy.on_env_init(env)

    t0 = time.perf_counter()
    tasks = env.run(load_scale=load)
    t1 = time.perf_counter()

    stats = compute_metrics(
        tasks,
        warmup=float(config["warmup_time"]),
        sim_time=float(config["sim_time"]),
    )
    if hasattr(policy, "get_cost_stats"):
        try:
            stats.update(policy.get_cost_stats())
        except Exception:
            pass

    stats["policy"] = policy_name
    stats["seed"] = int(seed)
    stats["load"] = float(load)
    stats["n_nodes"] = int(config["n_nodes"])
    stats["topk"] = int(config.get("cosims", {}).get("topk", 0))
    stats["wall_clock_s"] = float(t1 - t0)
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
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            ci95 = float(tcrit * std / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = std
            row[f"{col}_ci95"] = ci95
        rows.append(row)
    return pd.DataFrame(rows)


def compact_scalability_table(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "n_nodes",
        "policy",
        "n",
        "avg_latency_mean",
        "avg_latency_std",
        "p95_latency_mean",
        "p95_latency_std",
        "measured_decision_time_ms_mean_mean",
        "measured_decision_time_ms_mean_std",
        "decision_time_ms_mean_mean",
        "decision_time_ms_mean_std",
        "ot_solve_ms_mean_mean",
        "ot_solve_ms_mean_std",
        "trigger_fired_rate_mean",
        "trigger_fired_rate_std",
        "wall_clock_s_mean",
        "wall_clock_s_std",
    ]
    existing = [c for c in cols if c in summary.columns]
    table = summary[existing].copy()
    for c in ["trigger_fired_rate_mean", "trigger_fired_rate_std"]:
        if c in table.columns:
            table[c] = table[c] * 100.0
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/auction_B_recommended_sqlf_cloud.json")
    ap.add_argument("--out-dir", default="results/experiments/scalability_m11_rerun")
    ap.add_argument("--n-nodes", default="20,50,100")
    ap.add_argument("--optional-n-nodes", default="200")
    ap.add_argument("--include-optional", action="store_true")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--load", type=float, default=1.2)
    ap.add_argument(
        "--policies",
        default="cosims_sinkhorn,cosims_auction_B,cap_greedy",
    )
    args = ap.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        base_config = json.load(f)

    n_nodes = parse_csv_ints(args.n_nodes)
    optional_n_nodes = parse_csv_ints(args.optional_n_nodes)
    if args.include_optional:
        n_nodes = n_nodes + [n for n in optional_n_nodes if n not in n_nodes]
    seeds = parse_seeds(args.seeds)
    policies = parse_csv_strings(args.policies)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "scalability_raw_by_seed.checkpoint.csv"
    if checkpoint.exists():
        checkpoint.unlink()

    rows = []
    total = len(n_nodes) * len(policies) * len(seeds)
    done = 0
    for n in n_nodes:
        config = scaled_config(base_config, n)
        for policy_name in policies:
            for seed in seeds:
                row = run_one(config, policy_name, seed, args.load)
                row["scenario"] = "system_size_scalability"
                row["optional_scale"] = bool(n in optional_n_nodes)
                rows.append(row)
                pd.DataFrame([row]).to_csv(
                    checkpoint,
                    mode="a",
                    header=not checkpoint.exists(),
                    index=False,
                )
                done += 1
                print(
                    f"[m11] {done}/{total} N={n} policy={policy_name} seed={seed}",
                    flush=True,
                )

    raw = pd.DataFrame(rows)
    raw_path = out_dir / "scalability_raw_by_seed.csv"
    summary_path = out_dir / "scalability_summary_mean_std_ci.csv"
    compact_path = out_dir / "scalability_compact_table.csv"
    raw.to_csv(raw_path, index=False)
    summary = summarize(raw, ["scenario", "n_nodes", "optional_scale", "policy"])
    summary.to_csv(summary_path, index=False)
    compact_scalability_table(summary).to_csv(compact_path, index=False)

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Current simulator uses one n_nodes value for both source and edge nodes, so this is N_U=N_E=N system-size scaling.",
        "n_nodes": n_nodes,
        "optional_n_nodes": optional_n_nodes,
        "include_optional": bool(args.include_optional),
        "seeds": seeds,
        "load": float(args.load),
        "policies": policies,
        "metrics": [
            "avg_latency",
            "p95_latency",
            "decision_time_ms_mean",
            "ot_solve_ms_mean",
            "trigger_fired_rate",
        ],
        "paper_thresholds": final_paper_config(base_config).get("cosims", {}).get("event", {}),
        "env": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "config_snapshot.json").write_text(
        json.dumps(final_paper_config(base_config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[OK] wrote M11 scalability outputs")
    print(out_dir)
    print("raw rows:", len(raw), "summary:", summary.shape)


if __name__ == "__main__":
    main()
