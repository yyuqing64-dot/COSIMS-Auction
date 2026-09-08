from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import compute_metrics
from policies.cosims_auction_B import COSIMSAuctionB
from policies.cosims_sinkhorn import COSIMSSinkhorn
from sim.env import Env


T_CRIT_975 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}


def parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(',') if x.strip()]


def parse_seeds(value: str) -> list[int]:
    if '-' in value:
        a, b = value.split('-', 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in value.split(',') if x.strip()]


def final_paper_config(config: dict) -> dict:
    cfg = json.loads(json.dumps(config))
    cosims = cfg.setdefault('cosims', {})
    cosims['topk'] = 3
    cosims['lam'] = 0.1
    cosims['max_iter'] = 35
    cosims['ot_demand_mode'] = 'count'
    event = cosims.setdefault('event', {})
    event['misfit_delta'] = 0.5
    event['wait_tau'] = 6e11
    event['slope_th'] = 5e11
    event['rate_factor'] = 10.0
    cosims['event'] = event
    cfg['cosims'] = cosims
    return cfg


def variant_spec(name: str):
    specs = {
        'sinkhorn_prop': (COSIMSSinkhorn, 'prop', 'COSIMS-Sinkhorn-prop'),
        'sinkhorn_comp': (COSIMSSinkhorn, 'completion', 'COSIMS-Sinkhorn-comp'),
        'auction_prop': (COSIMSAuctionB, 'prop', 'COSIMS-Auction-prop'),
        'auction_comp': (COSIMSAuctionB, 'completion', 'COSIMS-Auction-comp'),
    }
    if name not in specs:
        raise KeyError(f'Unknown variant {name}; choices={sorted(specs)}')
    return specs[name]


def run_one(base_config: dict, variant: str, seed: int, load: float) -> dict:
    policy_cls, cost_mode, label = variant_spec(variant)
    config = json.loads(json.dumps(base_config))
    cosims = config.setdefault('cosims', {})
    cosims['ot_cost_mode'] = cost_mode
    cosims['ot_cost_service_weight'] = 1.0
    cosims['ot_cost_queue_weight'] = 1.0
    rng = np.random.default_rng(seed)
    policy = policy_cls(config, rng)
    env = Env(config, rng=rng, policy=policy)
    if hasattr(policy, 'on_env_init'):
        policy.on_env_init(env)

    tasks = env.run(load_scale=load)
    stats = compute_metrics(tasks, warmup=float(config['warmup_time']), sim_time=float(config['sim_time']))
    if hasattr(policy, 'get_cost_stats'):
        stats.update(policy.get_cost_stats())

    stats['scenario'] = 'm13_ot_cost_construction'
    stats['variant'] = label
    stats['variant_key'] = variant
    stats['base_policy'] = 'COSIMS-Auction' if 'auction' in variant else 'COSIMS-Sinkhorn'
    stats['ot_cost_mode'] = 'prop' if cost_mode == 'prop' else 'completion'
    stats['seed'] = int(seed)
    stats['load'] = float(load)
    stats['n_nodes'] = int(config.get('n_nodes', getattr(env, 'n', 0)))
    stats['topk'] = int(config.get('cosims', {}).get('topk', 0))
    return stats


def summarize(raw: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric_cols = [
        c for c in raw.columns
        if c not in set(group_cols + ['seed'])
        and pd.api.types.is_numeric_dtype(raw[c])
    ]
    rows = []
    for keys, g in raw.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        n = int(g['seed'].nunique())
        row['n'] = n
        tcrit = T_CRIT_975.get(n, 1.96)
        for col in numeric_cols:
            vals = pd.to_numeric(g[col], errors='coerce').dropna()
            if vals.empty:
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            ci95 = float(tcrit * std / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            row[f'{col}_mean'] = mean
            row[f'{col}_std'] = std
            row[f'{col}_ci95'] = ci95
        rows.append(row)
    return pd.DataFrame(rows)


def write_compact(summary: pd.DataFrame, out_dir: Path) -> None:
    compact = summary.copy()
    for col in ['avg_latency_mean', 'avg_latency_std', 'avg_latency_ci95', 'p95_latency_mean', 'p95_latency_std', 'p95_latency_ci95']:
        if col in compact.columns:
            compact[col.replace('latency', 'latency_1e11')] = compact[col] / 1e11
    for col in ['trigger_fired_rate_mean', 'trigger_fired_rate_std', 'trigger_fired_rate_ci95', 'auction_call_rate_mean', 'auction_call_rate_std', 'auction_call_rate_ci95']:
        if col in compact.columns:
            compact[col] = compact[col] * 100.0
    cols = [
        'load', 'variant', 'base_policy', 'ot_cost_mode', 'n',
        'avg_latency_1e11_mean', 'avg_latency_1e11_std', 'avg_latency_1e11_ci95',
        'p95_latency_1e11_mean', 'p95_latency_1e11_std', 'p95_latency_1e11_ci95',
        'cloud_rate_mean', 'cloud_rate_std', 'decision_time_ms_mean_mean', 'decision_time_ms_mean_std',
        'ot_solve_ms_mean_mean', 'ot_solve_ms_mean_std', 'trigger_fired_rate_mean', 'trigger_fired_rate_std',
    ]
    compact[[c for c in cols if c in compact.columns]].to_csv(out_dir / 'compact_table.csv', index=False)

    rows = []
    for (load, base_policy), g in compact.groupby(['load', 'base_policy'], dropna=False):
        modes = {str(r['ot_cost_mode']): r for _, r in g.iterrows()}
        if 'prop' not in modes or 'completion' not in modes:
            continue
        prop, comp = modes['prop'], modes['completion']
        row = {'load': load, 'base_policy': base_policy}
        for metric in [
            'avg_latency_1e11_mean', 'p95_latency_1e11_mean', 'cloud_rate_mean',
            'decision_time_ms_mean_mean', 'ot_solve_ms_mean_mean', 'trigger_fired_rate_mean'
        ]:
            if metric in prop.index and metric in comp.index and pd.notna(prop[metric]) and pd.notna(comp[metric]):
                row[f'{metric}_prop'] = float(prop[metric])
                row[f'{metric}_completion'] = float(comp[metric])
                row[f'{metric}_delta_pct'] = float((comp[metric] - prop[metric]) / max(abs(prop[metric]), 1e-12) * 100.0)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / 'prop_vs_completion_deltas.csv', index=False)


def write_paper_table(out_dir: Path) -> None:
    d = pd.read_csv(out_dir / 'prop_vs_completion_deltas.csv')
    rows = []
    for _, r in d.sort_values(['base_policy', 'load']).iterrows():
        trigger = '--'
        if 'trigger_fired_rate_mean_prop' in r.index and pd.notna(r.get('trigger_fired_rate_mean_prop')):
            trigger = f"{r['trigger_fired_rate_mean_prop']:.2f} / {r['trigger_fired_rate_mean_completion']:.2f}"
        rows.append({
            'policy': r['base_policy'],
            'load': f"{r['load']:.1f}",
            'avg_change_pct': f"{r['avg_latency_1e11_mean_delta_pct']:+.3f}",
            'p95_change_pct': f"{r['p95_latency_1e11_mean_delta_pct']:+.3f}",
            'ot_update_prop_comp_ms': f"{r['ot_solve_ms_mean_mean_prop']:.3f} / {r['ot_solve_ms_mean_mean_completion']:.3f}",
            'trigger_prop_comp': trigger,
        })
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / 'paper_table.csv', index=False)

    tex = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Validation of OT cost construction. Latency changes are relative to propagation-only OT; positive values indicate higher latency under completion-aware OT.}',
        r'\label{tab:ot_cost_construction}',
        r'\setlength{\tabcolsep}{3pt}',
        r'\renewcommand{\arraystretch}{1.05}',
        r'\begin{tabular}{ccccc}',
        r'\toprule',
        r'\textbf{Policy} & \textbf{Load} & \makecell{\textbf{Avg.}\\\textbf{Change (\%)}} & \makecell{\textbf{P95}\\\textbf{Change (\%)}} & \makecell{\textbf{OT Update (ms)}\\\textbf{Prop / Comp}} \\',
        r'\midrule',
    ]
    last_policy = None
    for r in rows:
        policy = r['policy']
        if last_policy is not None and policy != last_policy:
            tex.append(r'\midrule')
        tex.append(f"{policy} & {r['load']} & {r['avg_change_pct']} & {r['p95_change_pct']} & {r['ot_update_prop_comp_ms']} \\")
        last_policy = policy
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (out_dir / 'paper_table.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/auction_B_recommended_sqlf_cloud.json')
    ap.add_argument('--out-dir', default='results/experiments/ot_cost_construction_m13_rerun')
    ap.add_argument('--seeds', default='0-9')
    ap.add_argument('--loads', default='1.0,1.2,1.4')
    ap.add_argument('--variants', default='sinkhorn_prop,sinkhorn_comp,auction_prop,auction_comp')
    args = ap.parse_args()

    with open(ROOT / args.config, 'r', encoding='utf-8') as f:
        config = final_paper_config(json.load(f))
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / 'raw_by_seed.checkpoint.csv'
    if checkpoint.exists():
        checkpoint.unlink()

    seeds = parse_seeds(args.seeds)
    loads = parse_csv_floats(args.loads)
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]

    rows = []
    total = len(loads) * len(variants) * len(seeds)
    done = 0
    for load in loads:
        for variant in variants:
            for seed in seeds:
                done += 1
                print(f'[m13] {done}/{total} load={load} variant={variant} seed={seed}', flush=True)
                row = run_one(config, variant, seed, load)
                rows.append(row)
                pd.DataFrame([row]).to_csv(checkpoint, mode='a', header=not checkpoint.exists(), index=False)

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / 'raw_by_seed.csv', index=False)
    summary = summarize(raw, ['scenario', 'load', 'variant', 'base_policy', 'ot_cost_mode'])
    summary.to_csv(out_dir / 'summary_mean_std_ci.csv', index=False)
    write_compact(summary, out_dir)
    write_paper_table(out_dir)

    (out_dir / 'config_snapshot.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    (out_dir / 'run_meta.json').write_text(json.dumps({
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'seeds': seeds,
        'loads': loads,
        'variants': variants,
        'note': 'M13 compares propagation-only and completion-aware OT cost construction for Sinkhorn and Auction.',
        'env': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'platform': platform.platform()},
    }, indent=2, ensure_ascii=False), encoding='utf-8')

    print('[OK] wrote M13 OT cost construction outputs')
    print(out_dir)
    print('raw rows:', len(raw), 'summary:', summary.shape)


if __name__ == '__main__':
    main()
