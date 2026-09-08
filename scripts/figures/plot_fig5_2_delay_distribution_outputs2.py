#!/usr/bin/env python3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sample_csv = REPO_ROOT / 'results' / 'tables' / 'fig5_2_task_delay_distribution_with_greedy_latency_v2_sample.csv'
out_dir = REPO_ROOT / 'results' / 'figures'
out_png = out_dir / 'fig5_2_task_delay_distribution.png'
out_pdf = out_dir / 'fig5_2_task_delay_distribution.pdf'

df = pd.read_csv(sample_csv)
labels = ['RR', 'Greedy-Latency', 'SQLF', 'COSIMS-Sinkhorn', 'Cap-Greedy', 'COSIMS-Auction']
key_by_label = {'RR':'rr','Greedy-Latency':'greedy_latency','SQLF':'sqlf','COSIMS-Sinkhorn':'cosims_sinkhorn','Cap-Greedy':'cap_greedy','COSIMS-Auction':'cosims_auction_B'}
face = {'rr':'#D9E2EC','greedy_latency':'#FDE3C2','sqlf':'#DDEEDB','cosims_sinkhorn':'#E7DDF1','cap_greedy':'#F6D8C8','cosims_auction_B':'#F2F2F2'}
edge = {'rr':'#6B8FB3','greedy_latency':'#D47A1F','sqlf':'#5F9E6E','cosims_sinkhorn':'#8B6BB1','cap_greedy':'#C85C3D','cosims_auction_B':'#111111'}
data = [np.sort(df[df.method == label].completion_delay.to_numpy()) for label in labels]
positions = np.arange(len(labels)) * 0.56 + 1

plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
fig, ax = plt.subplots(figsize=(5.95, 3.45))
bp = ax.boxplot(
    data, positions=positions, labels=labels, patch_artist=True, whis=(5, 95), showfliers=False, widths=0.30,
    medianprops={'color':'black','linewidth':1.45}, boxprops={'linewidth':1.05},
    whiskerprops={'color':'#555555','linewidth':0.95}, capprops={'color':'#555555','linewidth':0.95},
)
for box, label in zip(bp['boxes'], labels):
    key = key_by_label[label]
    box.set_facecolor(face[key]); box.set_edgecolor(edge[key])
    box.set_alpha(0.72 if key != 'cosims_auction_B' else 0.84)
    box.set_linewidth(1.05 if key != 'cosims_auction_B' else 1.45)
for median, label in zip(bp['medians'], labels):
    key = key_by_label[label]
    median.set_color(edge[key] if key == 'cosims_auction_B' else 'black')
    median.set_linewidth(1.75 if key == 'cosims_auction_B' else 1.45)
ax.set_ylabel('Task completion delay ($10^9$ STU)', fontsize=10)
ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.24)
ax.tick_params(axis='y', labelsize=9)
ax.tick_params(axis='x', labelsize=8.0, rotation=10, pad=2)
for tick in ax.get_xticklabels():
    tick.set_ha('center')
    tick.set_rotation_mode('anchor')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(positions[0] - 0.34, positions[-1] + 0.34)
ax.set_ylim(0.0, float(np.percentile(np.concatenate(data), 99)) * 1.05)
fig.tight_layout()
fig.savefig(out_png, dpi=300, bbox_inches='tight')
fig.savefig(out_pdf, bbox_inches='tight')
print(f'Saved PNG: {out_png}')
print(f'Saved PDF: {out_pdf}')
