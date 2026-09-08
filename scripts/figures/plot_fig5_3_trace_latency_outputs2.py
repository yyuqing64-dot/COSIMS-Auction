#!/usr/bin/env python3
"""
Figure 5.3: trace latency performance.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    input_csv = REPO_ROOT / "results" / "experiments" / "trace_main_fig5_3" / "summary_mean_std.csv"
    out_dir = REPO_ROOT / "results" / "figures"
    out_png = out_dir / "fig5_3_trace_latency.png"
    out_pdf = out_dir / "fig5_3_trace_latency.pdf"
    table_dir = REPO_ROOT / "results" / "tables"
    out_table = table_dir / "fig5_3_trace_latency.csv"

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)

    required_cols = {"trace_load_factor", "policy", "avg_latency_mean", "p95_latency_mean"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    policy_keys = [
        "rr",
        "greedy_latency",
        "sqlf",
        "cap_greedy",
        "cosims_sinkhorn",
        "cosims_auction_B",
    ]
    policy_show = {
        "rr": "RR",
        "greedy_latency": "Greedy-Latency",
        "sqlf": "SQLF",
        "cap_greedy": "Cap-Greedy",
        "cosims_sinkhorn": "COSIMS-Sinkhorn",
        "cosims_auction_B": "COSIMS-Auction",
    }

    missing_policies = sorted(set(policy_keys) - set(df["policy"].unique()))
    if missing_policies:
        raise ValueError(f"Missing policies in CSV: {missing_policies}")

    colors = {
        "rr": (0.1216, 0.4667, 0.7059),
        "greedy_latency": (1.0000, 0.4980, 0.0549),
        "sqlf": (0.1725, 0.6275, 0.1725),
        "cap_greedy": (0.8392, 0.1529, 0.1569),
        "cosims_sinkhorn": (0.5804, 0.4039, 0.7412),
        "cosims_auction_B": (0.0, 0.0, 0.0),
    }
    linestyles = {
        "rr": "-",
        "greedy_latency": "--",
        "sqlf": "-.",
        "cap_greedy": "-",
        "cosims_sinkhorn": "--",
        "cosims_auction_B": "-",
    }
    markers = {
        "rr": "o",
        "greedy_latency": "s",
        "sqlf": "^",
        "cap_greedy": "D",
        "cosims_sinkhorn": "v",
        "cosims_auction_B": "*",
    }
    linewidths = {
        "rr": 1.8,
        "greedy_latency": 1.8,
        "sqlf": 1.8,
        "cap_greedy": 2.2,
        "cosims_sinkhorn": 2.0,
        "cosims_auction_B": 2.8,
    }
    zorders = {
        "rr": 2,
        "greedy_latency": 2,
        "sqlf": 2,
        "cap_greedy": 3,
        "cosims_sinkhorn": 3,
        "cosims_auction_B": 5,
    }

    plt.style.use("default")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.55), sharex=True)
    panels = [
        (axes[0], "avg_latency_mean", "(a) Average latency", "Average Completion Delay (STU)", (5000, 11200)),
        (axes[1], "p95_latency_mean", "(b) P95 latency", "P95 Completion Delay (STU)", (7500, 23000)),
    ]

    handles = []
    labels = []
    for ax, metric, title, ylabel, ylim in panels:
        ax.set_facecolor("white")
        for key in policy_keys:
            sub = df[df["policy"] == key].sort_values("trace_load_factor")
            x = sub["trace_load_factor"].to_numpy()
            y = sub[metric].to_numpy()
            line = ax.plot(
                x,
                y,
                label=policy_show[key],
                color=colors[key],
                linestyle=linestyles[key],
                marker=markers[key],
                linewidth=linewidths[key],
                markersize=8.0 if key == "cosims_auction_B" else 6.2,
                markerfacecolor=colors[key],
                markeredgecolor="white" if key == "cosims_auction_B" else colors[key],
                markeredgewidth=0.9 if key == "cosims_auction_B" else 0.6,
                zorder=zorders[key],
            )[0]
            if ax is axes[0]:
                handles.append(line)
                labels.append(policy_show[key])

        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.28)
        ax.set_xlim(0.75, 1.65)
        ax.set_xticks([0.8, 1.0, 1.2, 1.4, 1.6])
        ax.set_ylim(*ylim)
        ax.set_xlabel("Trace Load Factor", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.055),
        frameon=False,
        fontsize=9,
        handlelength=2.4,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.3)

    plot_table = df[["trace_load_factor", "policy", "avg_latency_mean", "avg_latency_std", "p95_latency_mean", "p95_latency_std"]].copy()
    plot_table.to_csv(out_table, index=False)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")
    print(f"Saved CSV: {out_table}")


if __name__ == "__main__":
    main()
