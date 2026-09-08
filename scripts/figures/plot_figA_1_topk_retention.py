#!/usr/bin/env python3
"""
Appendix Fig. A.1: assignment consistency with OT-guided candidates.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT))

from policies.cosims_auction_B import COSIMSAuctionB  # noqa: E402
from sim.env import Env  # noqa: E402


class RecordingCOSIMSAuction(COSIMSAuctionB):
    def __init__(self, *args, warmup_time: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._record_warmup_time = float(warmup_time)
        self.retention_records: list[dict] = []

    def select_dest(self, task, env):
        dest = super().select_dest(task, env)
        if float(task.arrival_t) >= self._record_warmup_time:
            src = int(task.src)
            p_row = np.asarray(self.P[src], dtype=float)
            p_row = p_row / max(float(p_row.sum()), 1e-12)
            topk = [int(v) for v in self._topk_indices(p_row, int(self.topk))]
            top1 = int(topk[0])
            if dest is None:
                category = "Cloud"
            else:
                dest_i = int(dest)
                if dest_i == top1:
                    category = "OT Top-1"
                elif dest_i in topk:
                    category = "OT Top-2/3"
                else:
                    category = "Outside Top-3"
            self.retention_records.append(
                {
                    "src": src,
                    "dest": None if dest is None else int(dest),
                    "ot_top1": top1,
                    "category": category,
                }
            )
        return dest


def run_load(config: dict, load: float, seeds: list[int]) -> dict[str, float]:
    counts = Counter()
    total = 0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        policy = RecordingCOSIMSAuction(config, rng, warmup_time=float(config["warmup_time"]))
        env = Env(config, rng=rng, policy=policy)
        env.run(load_scale=load)
        for rec in policy.retention_records:
            counts[rec["category"]] += 1
            total += 1
    categories = ["OT Top-1", "OT Top-2/3", "Outside Top-3", "Cloud"]
    return {cat: 100.0 * counts[cat] / max(total, 1) for cat in categories} | {"total_tasks": float(total)}


def main() -> None:
    config_path = ROOT / "configs" / "auction_B_recommended_sqlf_cloud.json"
    fig_dir = ROOT / "results" / "figures"
    table_dir = ROOT / "results" / "tables"
    out_png = fig_dir / "figA_1_topk_retention.png"
    out_pdf = fig_dir / "figA_1_topk_retention.pdf"
    out_csv = table_dir / "figA_1_topk_retention.csv"

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    loads = [1.0, 1.2, 1.4]
    seeds = list(range(10))
    categories = ["OT Top-1", "OT Top-2/3", "Outside Top-3", "Cloud"]

    rows = []
    for load in loads:
        ratios = run_load(config, load, seeds)
        row = {"load": load, "seeds": f"{seeds[0]}-{seeds[-1]}", **ratios}
        rows.append(row)

    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out_csv, index=False)

    plt.style.use("default")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    colors = {
        "OT Top-1": "#D9E2EC",
        "OT Top-2/3": "#7EA6C8",
        "Outside Top-3": "#E3B2A8",
        "Cloud": "#4A4A4A",
    }
    edges = {
        "OT Top-1": "#6B8FB3",
        "OT Top-2/3": "#4D789D",
        "Outside Top-3": "#B84A3A",
        "Cloud": "#4A4A4A",
    }

    fig, ax = plt.subplots(figsize=(6.4, 3.35))
    x = np.arange(len(loads), dtype=float)
    bottom = np.zeros(len(loads), dtype=float)
    for cat in categories:
        vals = table[cat].to_numpy(dtype=float)
        bars = ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.52,
            color=colors[cat],
            edgecolor=edges[cat],
            linewidth=1.0,
            label={"OT Top-1": "Top-1", "OT Top-2/3": "Top-2/3"}.get(cat, cat),
            zorder=3,
        )
        if cat in {"OT Top-1", "OT Top-2/3", "Cloud"}:
            for i, (b, v) in enumerate(zip(bottom, vals)):
                if v >= 5.0:
                    txt_color = "white" if cat == "Cloud" else "#2F3A45"
                    ax.text(i, b + v / 2.0, f"{v:.1f}%", ha="center", va="center", fontsize=8.2, color=txt_color)
        bottom += vals

    ax.set_title("Assignment consistency with OT-guided candidates", fontsize=10, pad=8)
    ax.set_xlabel("Load Factor", fontsize=10)
    ax.set_ylabel("Decision Ratio (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.1f}" for v in loads])
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.25, zorder=0)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.23),
        ncol=4,
        frameon=False,
        fontsize=8.6,
        handlelength=1.4,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    # Remove the earlier heatmap outputs so the fig directory only keeps the useful appendix analysis.
    for old in [
        fig_dir / "figA_1_assignment_structure.png",
        fig_dir / "figA_1_assignment_structure.pdf",
        table_dir / "figA_1_assignment_structure_metrics.csv",
    ]:
        if old.exists():
            old.unlink()

    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")
    print(f"Saved data: {out_csv}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
