from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_METRICS = [
    ("throughput_mean", "throughput_std", "Throughput (tasks/s)"),
    ("avg_latency_mean", "avg_latency_std", "Avg Latency (s)"),
    ("p95_latency_mean", "p95_latency_std", "P95 Latency (s)"),
    ("cloud_rate_mean", "cloud_rate_std", "Cloud Offload Rate"),
    ("completion_mean", "completion_std", "Completion Ratio"),
]


def plot_metric(df: pd.DataFrame, metric: str, std_col: str, title: str, out_path: str, group_by: str = "policy"):
    plt.figure(figsize=(6, 4))
    for grp, g in df.groupby(group_by):
        g = g.sort_values("load")
        plt.errorbar(
            g["load"],
            g[metric],
            yerr=g.get(std_col, None),
            marker="o",
            capsize=3,
            label=str(grp),
        )
    plt.title(title)
    plt.xlabel("Load scale")
    plt.ylabel(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to summary CSV (from eval.runner)")
    ap.add_argument("--out_dir", required=True, help="Directory to write plots")
    ap.add_argument("--group_by", type=str, default="policy", help="Column to group lines by")
    ap.add_argument("--filter_policy", type=str, default="", help="Filter to a single policy name for plotting")
    ap.add_argument("--ts_csv", type=str, default="", help="Optional time-series CSV")
    ap.add_argument("--filter_load", type=float, default=None, help="Filter specific load for time-series plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)
    if args.filter_policy:
        df = df[df["policy"] == args.filter_policy]

    for metric, std_col, title in DEFAULT_METRICS:
        out_path = os.path.join(args.out_dir, f"{metric}.png")
        if metric not in df.columns:
            continue
        plot_metric(df, metric, std_col if std_col in df.columns else None, title, out_path, group_by=args.group_by)
        print(f"[OK] saved {out_path}")

    # Additional cost plots if available
    if "decision_time_ms_mean_mean" in df.columns and "n_nodes" in df.columns:
        plt.figure(figsize=(6, 4))
        for policy, g in df.groupby("policy"):
            g = g.groupby("n_nodes", as_index=False)["decision_time_ms_mean_mean"].mean().sort_values("n_nodes")
            plt.plot(g["n_nodes"], g["decision_time_ms_mean_mean"], marker="o", label=policy)
        plt.title("Decision Time (ms) vs Nodes")
        plt.xlabel("Number of nodes")
        plt.ylabel("Decision time (ms)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        out_path = os.path.join(args.out_dir, "decision_time_vs_nodes.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"[OK] saved {out_path}")

    if "decision_time_ms_mean_mean" in df.columns and "topk" in df.columns:
        plt.figure(figsize=(6, 4))
        for policy, g in df[df["topk"].notna()].groupby("policy"):
            g = g.groupby("topk", as_index=False)["decision_time_ms_mean_mean"].mean().sort_values("topk")
            plt.plot(g["topk"], g["decision_time_ms_mean_mean"], marker="o", label=policy)
        plt.title("Decision Time (ms) vs Top-K")
        plt.xlabel("Top-K")
        plt.ylabel("Decision time (ms)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        out_path = os.path.join(args.out_dir, "decision_time_vs_topk.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"[OK] saved {out_path}")

    # Time-series plots (p95 vs time, backlog vs time, cloud rate vs time)
    if args.ts_csv and os.path.exists(args.ts_csv):
        df_ts = pd.read_csv(args.ts_csv)
        if args.filter_policy:
            df_ts = df_ts[df_ts["policy"] == args.filter_policy]
        if args.filter_load is not None:
            df_ts = df_ts[df_ts["load"] == args.filter_load]
        # p95 vs time
        plt.figure(figsize=(7, 4))
        for pol, g in df_ts.groupby("policy"):
            g = g.sort_values("time")
            plt.plot(g["time"], g["p95_latency_ts_mean"], label=pol)
        plt.title("P95 Latency vs Time (Stress Test)")
        plt.xlabel("Time (s)")
        plt.ylabel("P95 Latency (s)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        out_path = os.path.join(args.out_dir, "p95_latency_vs_time.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"[OK] saved {out_path}")
        # backlog work vs time
        plt.figure(figsize=(7, 4))
        for pol, g in df_ts.groupby("policy"):
            g = g.sort_values("time")
            plt.plot(g["time"], g["backlog_work_total_ts_mean"], label=pol)
        plt.title("Queue Backlog (work units) vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Backlog Work (units)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        out_path = os.path.join(args.out_dir, "backlog_work_vs_time.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"[OK] saved {out_path}")
        # cloud rate vs time
        plt.figure(figsize=(7, 4))
        for pol, g in df_ts.groupby("policy"):
            g = g.sort_values("time")
            plt.plot(g["time"], g["cloud_rate_ts_mean"], label=pol)
        plt.title("Cloud Offload Rate vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Cloud Offload Rate")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        out_path = os.path.join(args.out_dir, "cloud_rate_vs_time.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"[OK] saved {out_path}")


if __name__ == "__main__":
    main()
