#!/usr/bin/env python3
"""Fig. 5.4 with trace load factors 0.8--1.6 step 0.1, outputs2."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

import matplotlib.pyplot as plt
import pandas as pd

INPUT = REPO_ROOT / 'results' / 'experiments' / 'trace_tradeoff_fig5_4' / 'summary_mean_std.csv'
FIG_DIR = REPO_ROOT / 'results' / 'figures'
TABLE_DIR = REPO_ROOT / 'results' / 'tables'
OUT_PNG = FIG_DIR / 'fig5_4_trace_tradeoff_relative.png'
OUT_PDF = FIG_DIR / 'fig5_4_trace_tradeoff_relative.pdf'
OUT_CSV = TABLE_DIR / 'fig5_4_trace_tradeoff_relative.csv'
OUT_PNG_ALIAS = FIG_DIR / 'fig5_4_trace_tradeoff.png'
OUT_PDF_ALIAS = FIG_DIR / 'fig5_4_trace_tradeoff.pdf'


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)
    required = {'trace_load_factor', 'policy', 'avg_latency_mean', 'cloud_rate_mean'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')

    cap = df[df['policy'] == 'cap_greedy'].sort_values('trace_load_factor')
    auc = df[df['policy'] == 'cosims_auction_B'].sort_values('trace_load_factor')
    merged = cap[['trace_load_factor', 'avg_latency_mean', 'cloud_rate_mean']].merge(
        auc[['trace_load_factor', 'avg_latency_mean', 'cloud_rate_mean']],
        on='trace_load_factor', suffixes=('_cap', '_auction')
    )
    merged['avg_latency_reduction_pct'] = (
        (merged['avg_latency_mean_cap'] - merged['avg_latency_mean_auction'])
        / merged['avg_latency_mean_cap'] * 100.0
    )
    merged['cloud_util_reduction_pct'] = (
        (merged['cloud_rate_mean_cap'] - merged['cloud_rate_mean_auction'])
        / merged['cloud_rate_mean_cap'] * 100.0
    )
    merged.to_csv(OUT_CSV, index=False)

    x = merged['trace_load_factor']
    avg = merged['avg_latency_reduction_pct']
    cloud = merged['cloud_util_reduction_pct']

    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.45), sharex=True)
    panels = [
        (
            axes[0],
            avg,
            '(a) Average-latency reduction',
            'Average latency reduction vs Cap-Greedy (%)',
            (-0.5, 6.4),
        ),
        (
            axes[1],
            cloud,
            '(b) Cloud offloading-rate reduction',
            'Cloud offloading-rate reduction vs Cap-Greedy (%)',
            (-3.3, 5.5),
        ),
    ]
    for ax, values, title, ylabel, ylim in panels:
        ax.axhline(0.0, color='#555555', linestyle='--', linewidth=0.85, zorder=1)
        ax.plot(
            x, values,
            color='black', marker='o', markersize=4.8, linewidth=1.9,
            markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.0,
            zorder=4,
        )
        ax.fill_between(x.to_numpy(), 0.0, values.to_numpy(), color='#D9D9D9', alpha=0.15, zorder=2)
        for xi, yi in zip(x, values):
            # Label representative points only; nudge near-zero labels away from the reference line.
            if float(xi) in {0.8, 1.0, 1.2, 1.4, 1.6}:
                va = 'bottom' if yi >= 0 else 'top'
                off = 0.18 if yi >= 0 else -0.18
                x_shift = 0.0
                if abs(float(yi)) < 0.35:
                    off = -0.48 if yi < 0 else 0.40
                    x_shift = 0.025
                ax.text(xi + x_shift, yi + off, f'{yi:+.1f}', ha='center', va=va, fontsize=7.1, color='#333333')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Trace Load Factor', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9.3)
        ax.set_xlim(0.77, 1.63)
        ax.set_xticks([0.8, 1.0, 1.2, 1.4, 1.6])
        ax.set_ylim(*ylim)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.30, zorder=0)
        ax.tick_params(axis='both', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('COSIMS-Auction vs. Cap-Greedy under trace replay', fontsize=10.2, y=1.02)
    fig.tight_layout(w_pad=2.1)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    fig.savefig(OUT_PDF, bbox_inches='tight')
    fig.savefig(OUT_PNG_ALIAS, dpi=300, bbox_inches='tight')
    fig.savefig(OUT_PDF_ALIAS, bbox_inches='tight')
    print(f'Saved PNG: {OUT_PNG}')
    print(f'Saved PDF: {OUT_PDF}')
    print(f'Saved alias PNG: {OUT_PNG_ALIAS}')
    print(f'Saved alias PDF: {OUT_PDF_ALIAS}')
    print(f'Saved CSV: {OUT_CSV}')
    print(merged[['trace_load_factor','avg_latency_reduction_pct','cloud_util_reduction_pct']].to_string(index=False))

if __name__ == '__main__':
    main()
