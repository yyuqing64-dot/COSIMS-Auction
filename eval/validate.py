from __future__ import annotations
import argparse
import json
import os
import sys
import platform
import hashlib
import time
import math
from typing import List, Dict
import numpy as np
import pandas as pd

from sim.env import Env
from eval.metrics import compute_metrics

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

METRIC_KEYS = ["throughput", "avg_latency", "p95_latency", "cloud_rate", "completion"]

def run_once(config: dict, policy_cls, seed: int, load: float) -> dict:
    rng = np.random.default_rng(seed)
    policy = policy_cls(config, rng)
    env = Env(config, rng=rng, policy=policy)
    if hasattr(policy, "on_env_init"):
        policy.on_env_init(env)
    tasks = env.run(load_scale=load)
    stats = compute_metrics(tasks, warmup=float(config["warmup_time"]), sim_time=float(config["sim_time"]))
    return stats

def welch_t(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = a.size
    nb = b.size
    ma = float(a.mean())
    mb = float(b.mean())
    va = float(a.var(ddof=1))
    vb = float(b.var(ddof=1))
    denom = math.sqrt(va / max(na, 1) + vb / max(nb, 1))
    if denom <= 0:
        t = 0.0
    else:
        t = (ma - mb) / denom
    num = (va / max(na, 1) + vb / max(nb, 1)) ** 2
    dfa = (va / max(na, 1)) ** 2 / max(na - 1, 1)
    dfb = (vb / max(nb, 1)) ** 2 / max(nb - 1, 1)
    df = num / max(dfa + dfb, 1e-12)
    z = abs(t)
    p_norm = 2.0 * (0.5 - 0.5 * math.erf(z / math.sqrt(2.0)))
    return {"t_stat": float(t), "df_approx": float(df), "p_value_approx": float(p_norm)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--loads", type=str, default="0.8,1.0,1.4,1.6")
    ap.add_argument("--seeds", type=str, default="0-9")
    ap.add_argument("--policies", type=str, default="cosims_auction_B,cosims_sinkhorn")
    ap.add_argument("--compare", type=str, default="cosims_auction_B|cosims_sinkhorn;cosims_auction_A|cosims_sinkhorn")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    loads = [float(x.strip()) for x in args.loads.split(",") if x.strip()]
    if "-" in args.seeds:
        a, b = args.seeds.split("-")
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    policy_names = [x.strip() for x in args.policies.split(",") if x.strip()]
    for pn in policy_names:
        if pn not in POLICIES:
            raise ValueError(f"Unknown policy: {pn}")

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    raw_rows = []
    for load in loads:
        for pn in policy_names:
            policy_cls = POLICIES[pn]
            stats_list = []
            for s in seeds:
                stats = run_once(config, policy_cls, seed=s, load=load)
                stats_list.append(stats)
                r = {"load": load, "policy": pn, "seed": s}
                for k in METRIC_KEYS:
                    r[k] = float(stats[k])
                raw_rows.append(r)
            mean = {k: float(np.mean([d[k] for d in stats_list])) for k in METRIC_KEYS}
            std  = {k: float(np.std([d[k] for d in stats_list], ddof=1)) for k in METRIC_KEYS}
            row = {"load": load, "policy": pn}
            for k in METRIC_KEYS:
                row[f"{k}_mean"] = mean[k]
                row[f"{k}_std"] = std[k]
            rows.append(row)

    df_summary = pd.DataFrame(rows).sort_values(["load", "policy"])
    df_raw = pd.DataFrame(raw_rows).sort_values(["load", "policy", "seed"])

    df_summary.to_csv(os.path.join(args.out_dir, "summary_mean_std.csv"), index=False)
    df_raw.to_csv(os.path.join(args.out_dir, "raw_metrics.csv"), index=False)

    pairs = [p for p in args.compare.split(";") if p.strip()]
    for pair in pairs:
        a_name, b_name = [x.strip() for x in pair.split("|")]
        if a_name not in policy_names or b_name not in policy_names:
            continue
        rows_sig = []
        for load in loads:
            dfa = df_raw[(df_raw["load"] == load) & (df_raw["policy"] == a_name)]
            dfb = df_raw[(df_raw["load"] == load) & (df_raw["policy"] == b_name)]
            for k in METRIC_KEYS:
                a = dfa[k].to_numpy()
                b = dfb[k].to_numpy()
                res = welch_t(a, b)
                rows_sig.append({
                    "load": load,
                    "metric": k,
                    "policy_a": a_name,
                    "policy_b": b_name,
                    "mean_a": float(a.mean()) if a.size > 0 else float("nan"),
                    "mean_b": float(b.mean()) if b.size > 0 else float("nan"),
                    "t_stat": res["t_stat"],
                    "df_approx": res["df_approx"],
                    "p_value_approx": res["p_value_approx"],
                    "mean_diff": float((a.mean() - b.mean())) if a.size > 0 and b.size > 0 else float("nan"),
                })
        df_sig = pd.DataFrame(rows_sig).sort_values(["metric", "load"])
        df_sig.to_csv(os.path.join(args.out_dir, f"significance_{a_name}_vs_{b_name}.csv"), index=False)

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "config_path": args.config,
        "loads": loads,
        "seeds": seeds,
        "policies": policy_names,
        "compare_pairs": pairs,
        "env": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
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

if __name__ == "__main__":
    main()
