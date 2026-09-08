from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    base = REPO_ROOT / "results" / "experiments" / "k_sensitivity"
    raw = base / "main_results" / "summary_mean_std.csv"
    fig_dir = base / "fig"
    tab_dir = base / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw).sort_values(["load", "topk"]).copy()
    rows = []
    for load, group in df.groupby("load"):
        ref = group[group["topk"] == 3].iloc[0]
        for _, r in group.iterrows():
            avg_norm = float(r["avg_latency_mean"] / ref["avg_latency_mean"])
            p95_norm = float(r["p95_latency_mean"] / ref["p95_latency_mean"])
            rows.append(
                {
                    "load": float(load),
                    "K": int(r["topk"]),
                    "avg_latency": float(r["avg_latency_mean"]),
                    "p95_latency": float(r["p95_latency_mean"]),
                    "cloud_rate_percent": float(r["cloud_rate_mean"] * 100),
                    "trigger_rate_percent": float(r.get("trigger_fired_rate_mean", np.nan) * 100),
                    "decision_time_ms": float(r.get("decision_time_ms_mean_mean", np.nan)),
                    "auction_time_ms": float(r.get("auction_time_ms_mean_mean", np.nan)),
                    "avg_latency_norm_K3": avg_norm,
                    "p95_latency_norm_K3": p95_norm,
                    "avg_latency_delta_vs_K3_percent": float((avg_norm - 1) * 100),
                    "p95_latency_delta_vs_K3_percent": float((p95_norm - 1) * 100),
                }
            )

    compact = pd.DataFrame(rows)
    compact.to_csv(tab_dir / "k_sensitivity_summary_compact.csv", index=False)
    key_points = compact[compact["K"].isin([1, 3, 20])].copy()
    key_points.to_csv(tab_dir / "k_sensitivity_key_points.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#222222",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    ks = [1, 2, 3, 5, 8, 10, 20]
    xpos = np.arange(len(ks))
    load_styles = {
        1.0: {"color": "#4A5568", "marker": "o", "label": "Load 1.0"},
        1.2: {"color": "#C0392B", "marker": "s", "label": "Load 1.2"},
        1.4: {"color": "#111111", "marker": "^", "label": "Load 1.4"},
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=False)
    axes = axes.ravel()
    metrics = [
        ("avg_latency_norm_K3", "(a) Average latency", "Normalized to K=3"),
        ("p95_latency_norm_K3", "(b) P95 latency", "Normalized to K=3"),
        ("cloud_rate_percent", "(c) Cloud offloading rate", "Cloud offloading rate (%)"),
        ("decision_time_ms", "(d) Per-task decision time", "Decision time (ms)"),
    ]

    for ax, (col, title, ylabel) in zip(axes, metrics):
        for load in [1.0, 1.2, 1.4]:
            g = compact[compact["load"] == load].set_index("K").loc[ks].reset_index()
            st = load_styles[load]
            ax.plot(
                xpos,
                g[col],
                color=st["color"],
                marker=st["marker"],
                markersize=4.2,
                linewidth=1.55 if load == 1.4 else 1.4,
                label=st["label"],
            )
        if "norm" in col:
            ax.axhline(1.0, color="#888888", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_title(title, fontsize=9.2, pad=5)
        ax.set_ylabel(ylabel)
        ax.set_xticks(xpos)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("Candidate size K")
        ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[:2]:
        ys = []
        for line in ax.lines:
            ys.extend(line.get_ydata())
        ys = np.array([y for y in ys if np.isfinite(y)])
        ax.set_ylim(max(0.96, ys.min() - 0.015), ys.max() + 0.02)
    axes[2].set_ylim(0, compact["cloud_rate_percent"].max() * 1.18)
    axes[3].set_ylim(0, compact["decision_time_ms"].max() * 1.18)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.88, left=0.085, right=0.985, bottom=0.105, wspace=0.28, hspace=0.42)
    fig.savefig(fig_dir / "figM07_k_sensitivity_latency_overhead.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "figM07_k_sensitivity_latency_overhead.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:")
    print(raw)
    print(tab_dir / "k_sensitivity_summary_compact.csv")
    print(tab_dir / "k_sensitivity_key_points.csv")
    print(fig_dir / "figM07_k_sensitivity_latency_overhead.pdf")
    print(fig_dir / "figM07_k_sensitivity_latency_overhead.png")
    print()
    cols = [
        "load",
        "K",
        "avg_latency_delta_vs_K3_percent",
        "p95_latency_delta_vs_K3_percent",
        "cloud_rate_percent",
        "trigger_rate_percent",
        "decision_time_ms",
    ]
    print(key_points[cols].to_string(index=False))


if __name__ == "__main__":
    main()
