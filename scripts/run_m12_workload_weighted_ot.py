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
from eval.runner import _load_trace_arrays, _run_trace
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
        'sinkhorn_count': (COSIMSSinkhorn, 'count', 'COSIMS-Sinkhorn-count'),
        'sinkhorn_workload': (COSIMSSinkhorn, 'workload', 'COSIMS-Sinkhorn-workload'),
        'auction_count': (COSIMSAuctionB, 'count', 'COSIMS-Auction-count'),
        'auction_workload': (COSIMSAuctionB, 'workload', 'COSIMS-Auction-workload'),
    }
    if name not in specs:
        raise KeyError(f'Unknown variant {name}; choices={sorted(specs)}')
    return specs[name]


def run_one(base_config: dict, variant: str, seed: int, trace_factor: float, trace_arrays) -> dict:
    policy_cls, demand_mode, label = variant_spec(variant)
    config = json.loads(json.dumps(base_config))
    config.setdefault('cosims', {})['ot_demand_mode'] = demand_mode
    rng = np.random.default_rng(seed)
    policy = policy_cls(config, rng)
    env = Env(config, rng=rng, policy=policy)
    if hasattr(policy, 'on_env_init'):
        policy.on_env_init(env)

    arr_t, src_idx, cpu_arr, data_arr, trace_sim_time = trace_arrays
    tasks = _run_trace(env, arr_t, src_idx, cpu_arr, data_arr)
    stats = compute_metrics(tasks, warmup=float(config['warmup_time']), sim_time=float(trace_sim_time))
    if hasattr(policy, 'get_cost_stats'):
        stats.update(policy.get_cost_stats())

    stats['scenario'] = 'm12_workload_weighted_ot'
    stats['variant'] = label
    stats['variant_key'] = variant
    stats['base_policy'] = 'COSIMS-Auction' if 'auction' in variant else 'COSIMS-Sinkhorn'
    stats['ot_demand_mode'] = demand_mode
    stats['seed'] = int(seed)
    stats['trace_load_factor'] = float(trace_factor)
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
    for col in ['trigger_fired_rate_mean', 'trigger_fired_rate_std', 'trigger_fired_rate_ci95', 'auction_call_rate_mean', 'auction_call_rate_std', 'auction_call_rate_ci95']:
        if col in compact.columns:
            compact[col] = compact[col] * 100.0
    cols = [
        'trace_load_factor', 'variant', 'base_policy', 'ot_demand_mode', 'n',
        'avg_latency_mean', 'avg_latency_std', 'p95_latency_mean', 'p95_latency_std',
        'cloud_rate_mean', 'cloud_rate_std', 'decision_time_ms_mean_mean', 'decision_time_ms_mean_std',
        'ot_solve_ms_mean_mean', 'ot_solve_ms_mean_std', 'trigger_fired_rate_mean', 'trigger_fired_rate_std',
    ]
    compact[[c for c in cols if c in compact.columns]].to_csv(out_dir / 'compact_table.csv', index=False)

    # Pairwise count-vs-workload deltas by base policy and trace load.
    rows = []
    for (factor, base_policy), g in compact.groupby(['trace_load_factor', 'base_policy'], dropna=False):
        modes = {str(r['ot_demand_mode']): r for _, r in g.iterrows()}
        if 'count' not in modes or 'workload' not in modes:
            continue
        cnt, work = modes['count'], modes['workload']
        row = {'trace_load_factor': factor, 'base_policy': base_policy}
        for metric in ['avg_latency_mean', 'p95_latency_mean', 'cloud_rate_mean', 'decision_time_ms_mean_mean', 'ot_solve_ms_mean_mean', 'trigger_fired_rate_mean']:
            if metric in cnt.index and metric in work.index and pd.notna(cnt[metric]) and pd.notna(work[metric]):
                row[f'{metric}_count'] = float(cnt[metric])
                row[f'{metric}_workload'] = float(work[metric])
                row[f'{metric}_delta_pct'] = float((work[metric] - cnt[metric]) / max(abs(cnt[metric]), 1e-12) * 100.0)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / 'count_vs_workload_deltas.csv', index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/auction_B_recommended_sqlf_cloud.json')
    ap.add_argument('--out-dir', default='results/experiments/ot_demand_trace_m12_rerun')
    ap.add_argument('--seeds', default='0-4')
    ap.add_argument('--trace-load-factors', default='1.0,1.2,1.4')
    ap.add_argument('--variants', default='sinkhorn_count,sinkhorn_workload,auction_count,auction_workload')
    ap.add_argument('--trace-manifest', default='data/trace_inputs/arrivals_h77_load_sweep_0.8_1.6_step0.1_manifest.csv')
    args = ap.parse_args()

    with open(ROOT / args.config, 'r', encoding='utf-8') as f:
        config = final_paper_config(json.load(f))
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / 'raw_by_seed.checkpoint.csv'
    if checkpoint.exists():
        checkpoint.unlink()

    seeds = parse_seeds(args.seeds)
    trace_factors = parse_csv_floats(args.trace_load_factors)
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]

    manifest = pd.read_csv(ROOT / args.trace_manifest)
    trace_lookup = {float(r['trace_load_factor']): str(r['file']) for _, r in manifest.iterrows()}
    trace_cache = {}
    rows = []
    total = len(trace_factors) * len(variants) * len(seeds)
    done = 0
    for factor in trace_factors:
        trace_file = trace_lookup[float(factor)]
        if trace_file not in trace_cache:
            trace_cache[trace_file] = _load_trace_arrays(trace_file, int(config['n_nodes']))
        arrays = trace_cache[trace_file]
        for variant in variants:
            for seed in seeds:
                done += 1
                print(f'[m12] {done}/{total} load={factor} variant={variant} seed={seed}', flush=True)
                row = run_one(config, variant, seed, factor, arrays)
                rows.append(row)
                pd.DataFrame([row]).to_csv(checkpoint, mode='a', header=not checkpoint.exists(), index=False)

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / 'raw_by_seed.csv', index=False)
    summary = summarize(raw, ['scenario', 'trace_load_factor', 'variant', 'base_policy', 'ot_demand_mode'])
    summary.to_csv(out_dir / 'summary_mean_std_ci.csv', index=False)
    write_compact(summary, out_dir)

    (out_dir / 'config_snapshot.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    (out_dir / 'run_meta.json').write_text(json.dumps({
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'seeds': seeds,
        'trace_load_factors': trace_factors,
        'variants': variants,
        'note': 'M12 compares count-based and workload-weighted source demand vectors for the epoch-level OT reference.',
        'env': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'platform': platform.platform()},
    }, indent=2, ensure_ascii=False), encoding='utf-8')

    print('[OK] wrote M12 workload-weighted OT outputs')
    print(out_dir)
    print('raw rows:', len(raw), 'summary:', summary.shape)


if __name__ == '__main__':
    main()
