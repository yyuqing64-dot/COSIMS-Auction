from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import compute_metrics
from policies.cosims_auction_B import COSIMSAuctionB
from sim.env import Env


class TriggerAblationAuction(COSIMSAuctionB):
    def __init__(
        self,
        *args,
        enable_arrival_trigger: bool = True,
        enable_congestion_trigger: bool = True,
        enable_mismatch_trigger: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.enable_arrival_trigger = enable_arrival_trigger
        self.enable_congestion_trigger = enable_congestion_trigger
        self.enable_mismatch_trigger = enable_mismatch_trigger

    def _arrival_trigger(self, now: float) -> bool:
        triggered = super()._arrival_trigger(now)
        return bool(triggered and self.enable_arrival_trigger)

    def _backlog_triggers(self, env, now: float, top1: int) -> bool:
        triggered = super()._backlog_triggers(env, now, top1)
        return bool(triggered and self.enable_congestion_trigger)

    def _misfit_trigger(self, task, env, now: float, p_row: np.ndarray, top1: int) -> bool:
        triggered = super()._misfit_trigger(task, env, now, p_row, top1)
        return bool(triggered and self.enable_mismatch_trigger)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    index_cols = ["variant", "load"]
    metric_cols = [
        "avg_latency",
        "p95_latency",
        "cloud_rate",
        "decision_time_ms_mean",
        "auction_time_ms_mean",
        "trigger_fired_rate",
        "ot_solve_ms_mean",
        "ot_update_count",
    ]
    rows = []
    for keys, group in raw.groupby(index_cols, sort=False):
        row = dict(zip(index_cols, keys))
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["paper", "config"], default="paper", help="paper uses manuscript thresholds; config keeps thresholds from the config file.")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "experiments" / "trigger_ablation_m08"))
    args = ap.parse_args()

    project = REPO_ROOT
    out_dir = Path(args.out_dir)
    result_dir = out_dir / "main_results"
    table_dir = out_dir / "tables"
    result_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    config_path = project / "configs" / "auction_B_recommended_sqlf_cloud.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cosims = config.setdefault("cosims", {})
    event = cosims.setdefault("event", {})

    cosims["topk"] = 3
    cosims["lam"] = 0.1
    cosims["max_iter"] = 35
    cosims["alpha"] = 1.0
    cosims["eta"] = 1.0
    cosims["eps_noise"] = 0.0
    if args.preset == "paper":
        event["misfit_delta"] = 0.5
        event["wait_tau"] = 6e11
        event["slope_th"] = 5e11
        event["rate_factor"] = 10.0

    variants = [
        ("Full", True, True, True),
        ("w/o AS", False, True, True),
        ("w/o LC", True, False, True),
        ("w/o RM", True, True, False),
        ("No Trigger", False, False, False),
    ]
    loads = [1.0, 1.2, 1.4]
    seeds = [0, 1, 2, 3, 4]

    rows = []
    for variant, use_as, use_lc, use_rm in variants:
        for load in loads:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                policy = TriggerAblationAuction(
                    config,
                    rng,
                    enable_arrival_trigger=use_as,
                    enable_congestion_trigger=use_lc,
                    enable_mismatch_trigger=use_rm,
                )
                env = Env(config, rng=rng, policy=policy)
                tasks = env.run(load_scale=load)
                row = {
                    "variant": variant,
                    "enable_arrival_trigger": use_as,
                    "enable_congestion_trigger": use_lc,
                    "enable_mismatch_trigger": use_rm,
                    "load": load,
                    "seed": seed,
                    "topk": int(cosims["topk"]),
                    "misfit_delta": float(event["misfit_delta"]),
                    "wait_tau": float(event["wait_tau"]),
                    "slope_th": float(event["slope_th"]),
                    "rate_factor": float(event["rate_factor"]),
                }
                row.update(compute_metrics(tasks, float(config["warmup_time"]), float(config["sim_time"])))
                row.update(policy.get_cost_stats())
                rows.append(row)

    raw = pd.DataFrame(rows)
    summary = summarize(raw)

    raw_path = result_dir / "trigger_ablation_raw_by_seed.csv"
    summary_path = result_dir / "trigger_ablation_summary_mean_std.csv"
    compact_path = table_dir / "trigger_ablation_compact_table.csv"
    config_snapshot = result_dir / "config_snapshot.json"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    config_snapshot.write_text(json.dumps(config, indent=2), encoding="utf-8")

    compact = summary.copy()
    compact["avg_latency_1e11"] = compact["avg_latency_mean"] / 1e11
    compact["p95_latency_1e11"] = compact["p95_latency_mean"] / 1e11
    compact["trigger_rate_percent"] = compact["trigger_fired_rate_mean"] * 100
    compact["decision_time_ms"] = compact["decision_time_ms_mean_mean"]
    compact = compact[
        [
            "variant",
            "load",
            "avg_latency_1e11",
            "p95_latency_1e11",
            "trigger_rate_percent",
            "decision_time_ms",
        ]
    ]
    compact.to_csv(compact_path, index=False)

    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {compact_path}")
    print(f"Wrote {config_snapshot}")
    print()
    print(compact.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
