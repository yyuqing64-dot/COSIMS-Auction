#!/usr/bin/env python3
"""
Appendix Fig. A.1: assignment-structure visualization.

The figure compares the row-normalized OT reference matrix with the actual
edge-side assignment matrix of COSIMS-Auction at load factor 1.2.
"""

from __future__ import annotations

import json
import sys
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
        self.p_history: list[np.ndarray] = []

    def _maybe_update_epoch(self, now: float):
        before = int(self._ot_updates)
        super()._maybe_update_epoch(now)
        if int(self._ot_updates) > before and float(now) >= self._record_warmup_time:
            self.p_history.append(np.asarray(self.P, dtype=float).copy())


def row_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    return mat / np.maximum(mat.sum(axis=1, keepdims=True), 1e-12)


def average_row_cosine(a: np.ndarray, b: np.ndarray) -> float:
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = den > 1e-12
    if not np.any(valid):
        return float("nan")
    return float(np.mean(num[valid] / den[valid]))


def main() -> None:
    config_path = ROOT / "configs" / "auction_B_recommended_sqlf_cloud.json"
    fig_dir = ROOT / "results" / "figures"
    table_dir = ROOT / "results" / "tables"
    out_png = fig_dir / "figA_1_assignment_structure.png"
    out_pdf = fig_dir / "figA_1_assignment_structure.pdf"
    out_metrics = table_dir / "figA_1_assignment_structure_metrics.csv"

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    load_factor = 1.2
    seed = 0
    rng = np.random.default_rng(seed)
    warmup_time = float(config["warmup_time"])
    policy = RecordingCOSIMSAuction(config, rng, warmup_time=warmup_time)
    env = Env(config, rng=rng, policy=policy)
    tasks = env.run(load_scale=load_factor)

    n = int(config["n_nodes"])
    if not policy.p_history:
        raise RuntimeError("No OT matrices were recorded after warmup.")

    p_avg = row_normalize(np.mean(np.stack(policy.p_history, axis=0), axis=0))

    actual = np.zeros((n, n), dtype=float)
    edge_assignments = 0
    cloud_after_warmup = 0
    total_after_warmup = 0
    for task in tasks:
        if float(task.arrival_t) < warmup_time:
            continue
        total_after_warmup += 1
        if bool(task.is_cloud):
            cloud_after_warmup += 1
            continue
        if task.dest is None:
            continue
        src = int(task.src)
        dest = int(task.dest)
        if 0 <= src < n and 0 <= dest < n:
            actual[src, dest] += 1.0
            edge_assignments += 1

    a_norm = row_normalize(actual)

    source_order = np.argsort(np.argmax(p_avg, axis=1), kind="stable")
    p_show = p_avg[source_order, :]
    a_show = a_norm[source_order, :]
    vmax = float(max(p_show.max(), a_show.max()))

    cosine = average_row_cosine(p_avg, a_norm)

    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "load_factor": load_factor,
                "seed": seed,
                "ot_updates_recorded": len(policy.p_history),
                "tasks_after_warmup": total_after_warmup,
                "edge_assignments_after_warmup": edge_assignments,
                "cloud_assignments_after_warmup": cloud_after_warmup,
                "cloud_fraction_after_warmup": cloud_after_warmup / max(total_after_warmup, 1),
                "avg_row_cosine_similarity": cosine,
            }
        ]
    ).to_csv(out_metrics, index=False)

    plt.style.use("default")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.45), sharey=True, constrained_layout=True)
    cmap = "Blues"
    im0 = axes[0].imshow(p_show, aspect="equal", cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
    im1 = axes[1].imshow(a_show, aspect="equal", cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")

    titles = ["(a) OT reference matrix", "(b) Actual assignment matrix"]
    tick_pos = [0, 4, 9, 14, 19]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Edge Node", fontsize=10)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([str(i) for i in tick_pos])
        ax.tick_params(axis="both", labelsize=8.5, length=2.5, width=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#444444")

    axes[0].set_ylabel("Source Node", fontsize=10)
    axes[0].set_yticks(tick_pos)
    axes[0].set_yticklabels([str(int(source_order[i])) for i in tick_pos])
    axes[1].tick_params(axis="y", left=False, labelleft=False)

    cbar = fig.colorbar(im1, ax=axes, fraction=0.035, pad=0.025)
    cbar.set_label("Assignment Ratio", fontsize=10)
    cbar.ax.tick_params(labelsize=8.5, length=2.5, width=0.7)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")

    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")
    print(f"Saved metrics: {out_metrics}")
    print(f"Average row-wise cosine similarity: {cosine:.4f}")
    print(f"Cloud fraction after warmup: {cloud_after_warmup / max(total_after_warmup, 1):.4f}")


if __name__ == "__main__":
    main()
