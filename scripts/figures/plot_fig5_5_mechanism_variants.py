#!/usr/bin/env python3
"""
Figure 5.5: mechanism variant comparison under synthetic workloads.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_normalized_table(df: pd.DataFrame, loads: list[float]) -> pd.DataFrame:
    policy_map = {
        "cosims_sinkhorn": "COSIMS-Sinkhorn",
        "cosims_auction_A": "COSIMS-Auction-Full",
        "cosims_auction_B": "COSIMS-Auction",
    }
    rows = []
    for load in loads:
        sub = df[df["load"] == load].copy()
        if sub.empty:
            raise ValueError(f"Missing load={load}")
        full = sub[sub["policy"] == "cosims_auction_A"]
        if full.empty:
            raise ValueError(f"Missing COSIMS-Auction-Full baseline at load={load}")
        avg_base = float(full.iloc[0]["avg_latency_mean"])
        p95_base = float(full.iloc[0]["p95_latency_mean"])
        for raw_policy, show_policy in policy_map.items():
            row = sub[sub["policy"] == raw_policy]
            if row.empty:
                raise ValueError(f"Missing policy={raw_policy} at load={load}")
            r = row.iloc[0]
            rows.append(
                {
                    "load": load,
                    "method": show_policy,
                    "avg_latency_norm": float(r["avg_latency_mean"]) / avg_base,
                    "p95_latency_norm": float(r["p95_latency_mean"]) / p95_base,
                    "avg_latency_raw": float(r["avg_latency_mean"]),
                    "p95_latency_raw": float(r["p95_latency_mean"]),
                }
            )
    return pd.DataFrame(rows)


def build_trigger_gap_table(df: pd.DataFrame, loads: list[float]) -> pd.DataFrame:
    rows = []
    for load in loads:
        sub = df[df["load"] == load]
        full = sub[sub["policy"] == "cosims_auction_A"].iloc[0]
        trigger = sub[sub["policy"] == "cosims_auction_B"].iloc[0]
        rows.append(
            {
                "load": load,
                "delta_avg_pct": (float(trigger["avg_latency_mean"]) / float(full["avg_latency_mean"]) - 1.0) * 100.0,
                "delta_p95_pct": (float(trigger["p95_latency_mean"]) / float(full["p95_latency_mean"]) - 1.0) * 100.0,
                "trigger_rate": float(trigger["trigger_fired_rate_mean"]),
            }
        )
    return pd.DataFrame(rows)


def write_trigger_gap_latex(table: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Performance gap between continuous and trigger-based correction. Here, $\Delta$ is computed relative to COSIMS-Auction-Full.}",
        r"\label{tab:trigger_gap}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Load & $\Delta$ Avg (\%) & $\Delta$ P95 (\%) & Trigger Rate \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{row['load']:.1f} & {row['delta_avg_pct']:+.3f} & {row['delta_p95_pct']:+.3f} & {row['trigger_rate']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    input_csv = REPO_ROOT / "results" / "experiments" / "mechanism_variants_fig5_5_5_6" / "summary_mean_std.csv"
    fig_dir = REPO_ROOT / "results" / "figures"
    table_dir = REPO_ROOT / "results" / "tables"
    out_png = fig_dir / "fig5_5_mechanism_variant_normalized.png"
    out_pdf = fig_dir / "fig5_5_mechanism_variant_normalized.pdf"
    out_csv = table_dir / "fig5_5_mechanism_variant_normalized.csv"
    gap_csv = table_dir / "table_trigger_gap.csv"
    gap_tex = table_dir / "table_trigger_gap.tex"

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)
    required = {"load", "policy", "avg_latency_mean", "p95_latency_mean", "trigger_fired_rate_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    loads = [1.0, 1.2, 1.4]
    methods = ["COSIMS-Sinkhorn", "COSIMS-Auction-Full", "COSIMS-Auction"]
    norm = build_normalized_table(df, loads)
    norm.to_csv(out_csv, index=False)
    trigger_gap = build_trigger_gap_table(df, loads)
    trigger_gap.to_csv(gap_csv, index=False)
    write_trigger_gap_latex(trigger_gap, gap_tex)

    facecolors = {
        "COSIMS-Sinkhorn": "#D9CCE8",
        "COSIMS-Auction-Full": "#FFFFFF",
        "COSIMS-Auction": "#1A1A1A",
    }
    edgecolors = {
        "COSIMS-Sinkhorn": "#7C5AA6",
        "COSIMS-Auction-Full": "#B84A3A",
        "COSIMS-Auction": "#1A1A1A",
    }
    hatches = {
        "COSIMS-Sinkhorn": "",
        "COSIMS-Auction-Full": "//",
        "COSIMS-Auction": "",
    }

    plt.style.use("default")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.35), sharex=True)
    panels = [
        (axes[0], "avg_latency_norm", "(a) Normalized average latency", (0.96, 1.42)),
        (axes[1], "p95_latency_norm", "(b) Normalized P95 latency", (0.96, 1.74)),
    ]

    x = np.arange(len(loads), dtype=float)
    width = 0.18
    offsets = [-width, 0.0, width]
    legend_show = {
        "COSIMS-Sinkhorn": "Sinkhorn",
        "COSIMS-Auction-Full": "Auction-Full",
        "COSIMS-Auction": "COSIMS-Auction",
    }

    handles = []
    labels = []

    for ax, metric, title, ylim in panels:
        for idx, method in enumerate(methods):
            vals = []
            for load in loads:
                val = norm[(norm["load"] == load) & (norm["method"] == method)][metric].iloc[0]
                vals.append(float(val))
            bars = ax.bar(
                x + offsets[idx],
                vals,
                width=width,
                color=facecolors[method],
                edgecolor=edgecolors[method],
                linewidth=1.15 if method != "COSIMS-Auction" else 1.35,
                hatch=hatches[method],
                label=method,
                zorder=4 if method == "COSIMS-Auction" else 3,
            )
            if ax is axes[0]:
                handles.append(bars[0])
                labels.append(legend_show[method])

        ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.85, alpha=0.70, zorder=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Load Factor", fontsize=10)
        ax.set_ylabel("Normalized Latency", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.1f}" for v in loads])
        ax.set_ylim(*ylim)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.25, zorder=0)
        ax.tick_params(axis="both", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.08),
        frameon=False,
        fontsize=9,
        handlelength=1.6,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.0)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")
    print(f"Saved normalized data: {out_csv}")
    print(f"Saved trigger-gap table: {gap_tex}")


if __name__ == "__main__":
    main()
