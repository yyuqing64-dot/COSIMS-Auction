#!/usr/bin/env python3
"""
Figure 5.6: triggering behavior and online decision overhead.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    input_csv = REPO_ROOT / "results" / "experiments" / "mechanism_variants_fig5_5_5_6" / "summary_mean_std.csv"
    out_dir = REPO_ROOT / "results" / "figures"
    out_png = out_dir / "fig5_6_trigger_overhead.png"
    out_pdf = out_dir / "fig5_6_trigger_overhead.pdf"

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)
    loads = [1.0, 1.2, 1.4]

    required = {
        "load",
        "policy",
        "trigger_fired_rate_mean",
        "decision_time_ms_mean_mean",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    full = df[(df["policy"] == "cosims_auction_A") & (df["load"].isin(loads))].sort_values("load")
    auction = df[(df["policy"] == "cosims_auction_B") & (df["load"].isin(loads))].sort_values("load")
    if full.shape[0] != len(loads):
        raise ValueError("Missing COSIMS-Auction-Full rows")
    if auction.shape[0] != len(loads):
        raise ValueError("Missing COSIMS-Auction rows")

    trigger_pct = auction["trigger_fired_rate_mean"].to_numpy(dtype=float) * 100.0
    default_pct = 100.0 - trigger_pct
    full_decision = full["decision_time_ms_mean_mean"].to_numpy(dtype=float)
    auction_decision = auction["decision_time_ms_mean_mean"].to_numpy(dtype=float)

    plt.style.use("default")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.85), sharex=False)
    x = np.arange(len(loads), dtype=float)

    # (a) Triggering behavior.
    ax = axes[0]
    default_color = "#D9E2EC"
    trigger_color = "#4A4A4A"
    ax.bar(
        x,
        default_pct,
        width=0.52,
        color=default_color,
        edgecolor="#6B8FB3",
        linewidth=1.0,
        label="Default Path",
        zorder=3,
    )
    ax.bar(
        x,
        trigger_pct,
        bottom=default_pct,
        width=0.52,
        color=trigger_color,
        edgecolor=trigger_color,
        linewidth=1.0,
        label="Triggered Correction",
        zorder=3,
    )
    for i, (d, t) in enumerate(zip(default_pct, trigger_pct)):
        ax.text(i, d / 2.0, f"{d:.1f}%", ha="center", va="center", fontsize=8.4, color="#2F3A45")
        ax.text(i, d + t / 2.0, f"{t:.1f}%", ha="center", va="center", fontsize=8.4, color="white")

    ax.set_title("(a) Trigger ratios", fontsize=10)
    ax.set_xlabel("Load Factor", fontsize=10)
    ax.set_ylabel("Task Ratio (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.1f}" for v in loads])
    ax.set_ylim(0, 108)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.25, zorder=0)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # legend handled at figure level

    # (b) Online decision overhead.
    ax = axes[1]
    width = 0.28
    full_color = "#FFFFFF"
    full_edge = "#B84A3A"
    auction_color = "#1A1A1A"
    ax.bar(
        x - width / 2,
        full_decision,
        width=width,
        color=full_color,
        edgecolor=full_edge,
        hatch="//",
        linewidth=1.15,
        label="Auction-Full",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        auction_decision,
        width=width,
        color=auction_color,
        edgecolor=auction_color,
        linewidth=1.15,
        label="COSIMS-Auction",
        zorder=3,
    )
    for i, (a, b) in enumerate(zip(full_decision, auction_decision)):
        ax.text(i - width / 2, a + 0.004, f"{a:.3f}", ha="center", va="bottom", fontsize=8.0, color="#4A4A4A")
        ax.text(i + width / 2, b + 0.004, f"{b:.3f}", ha="center", va="bottom", fontsize=8.0, color="#4A4A4A")

    ax.set_title("(b) Per-task decision time", fontsize=10)
    ax.set_xlabel("Load Factor", fontsize=10)
    ax.set_ylabel("Decision Time (ms)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.1f}" for v in loads])
    ax.set_ylim(0, 0.15)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.25, zorder=0)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # legend handled at figure level

    
    # Separate compact legends keep the top of each subplot readable.
    handles_left, labels_left = axes[0].get_legend_handles_labels()
    handles_right, labels_right = axes[1].get_legend_handles_labels()
    fig.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.03),
        frameon=False,
        fontsize=8.6,
        handlelength=1.5,
        columnspacing=1.1,
    )
    fig.tight_layout(w_pad=2.4, rect=(0, 0, 1, 0.91))
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")


if __name__ == "__main__":
    main()
