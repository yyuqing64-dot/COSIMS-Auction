"""Load-specific Q90/Q95/Q99 calibration and paired evaluation.

Run explicitly from the repository root. --help performs no experiment.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_seeds(value):
    result = []
    for part in value.split(','):
        part = part.strip()
        if '-' in part:
            first, last = map(int, part.split('-', 1))
            if first > last:
                raise argparse.ArgumentTypeError('Seed range must be ascending.')
            result.extend(range(first, last + 1))
        else:
            result.append(int(part))
    if not result or len(set(result)) != len(result) or any(x < 0 for x in result):
        raise argparse.ArgumentTypeError('Seeds must be distinct non-negative integers.')
    return result


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/manuscript_reproduction.json')
    parser.add_argument('--loads', default='1.0,1.2,1.4')
    parser.add_argument('--quantiles', default='0.90,0.95,0.99')
    parser.add_argument('--calibration-seed', type=int, default=1000)
    parser.add_argument('--seeds', type=parse_seeds, default=parse_seeds('0-9'))
    parser.add_argument('--out', default='results/experiments/threshold_calibration_reproduction')
    return parser.parse_args()


def main():
    args = arguments()
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import pandas as pd
    from scripts import threshold_calibration_support as core

    loads = [float(x) for x in args.loads.split(',')]
    quantiles = [float(x) for x in args.quantiles.split(',')]
    if not loads or len(set(loads)) != len(loads) or any(x <= 0 for x in loads):
        raise ValueError('Loads must be distinct positive numbers.')
    if not quantiles or len(set(quantiles)) != len(quantiles) or any(not 0 < x < 1 for x in quantiles):
        raise ValueError('Quantiles must be distinct values strictly between 0 and 1.')
    if args.calibration_seed < 0 or args.calibration_seed in args.seeds:
        raise ValueError('Calibration and evaluation seeds must be independent.')
    labels = {q: 'q' + format(q * 100, '.10g') for q in quantiles}
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    if config['n_nodes'] != 20 or config['cosims']['topk'] != 3:
        raise ValueError('This reproduction entry point expects N=20 and K=3.')
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    core.dump(out / 'config_snapshot.json', config)
    sources = ['policies/cosims_auction_B.py', 'sim/env.py', 'sim/network.py',
               'sim/task.py', 'eval/metrics.py', 'scripts/threshold_calibration_support.py',
               'scripts/run_threshold_calibration.py']
    manifest = {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'loads': loads, 'quantiles': quantiles, 'evaluation_seeds': args.seeds,
        'calibration_seed': args.calibration_seed, 'n_nodes': config['n_nodes'],
        'scheduling_k': config['cosims']['topk'], 'sim_time': config['sim_time'],
        'warmup': config['warmup_time'],
        'protocol': 'Independent load-specific calibration; thresholds frozen before paired evaluation.',
        'waiting_signal': 'Native backlog at the OT-preferred edge.',
        'growth_signal': 'Maximum positive native backlog growth across edges, using the policy observation interval.',
        'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'environment': {'python': sys.version, 'numpy': np.__version__,
                        'pandas': pd.__version__, 'platform': platform.platform()},
    }
    core.dump(out / 'manifest.json', manifest)
    frozen = {}
    threshold_rows = []
    # Every threshold is frozen before evaluation outcomes are produced.
    for load in loads:
        core.LOAD = load
        directory = out / 'calibration' / ('load_' + format(load, '.10g'))
        directory.mkdir(parents=True)
        tasks, env, policy = core.execute(config, args.calibration_seed)
        expected = core.state(tasks, env)
        del tasks, env, policy
        gc.collect()
        tasks, env, policy = core.execute(config, args.calibration_seed, 'calibration', directory)
        try:
            if core.state(tasks, env) != expected:
                raise RuntimeError('Calibration observation changed the simulation trajectory.')
            frozen[load] = {}
            for quantile in quantiles:
                core.Q = quantile
                values = core.thresholds(policy.observations)
                values.update(load_factor=load, quantile=quantile,
                              calibration_seed=args.calibration_seed)
                frozen[load][labels[quantile]] = values
                threshold_rows.append({'variant': labels[quantile], **values})
            core.dump(directory / 'thresholds.json', frozen[load])
        finally:
            policy.f.close()
        del tasks, env, policy
        gc.collect()
    pd.DataFrame(threshold_rows).to_csv(out / 'calibrated_thresholds.csv', index=False)

    rows = []
    for load in loads:
        core.LOAD = load
        variants = {'original_fixed': config}
        for label, values in frozen[load].items():
            calibrated = copy.deepcopy(config)
            calibrated['cosims']['event']['wait_tau'] = values['tau_W']
            calibrated['cosims']['event']['slope_th'] = values['tau_S']
            variants[label] = calibrated
        for label, cfg in variants.items():
            path = out / 'configs' / ('load_' + format(load, '.10g'))
            path.mkdir(parents=True, exist_ok=True)
            core.dump(path / (label + '.json'), cfg)
        for seed in args.seeds:
            control = None
            for label, cfg in variants.items():
                directory = out / 'runs' / ('load_' + format(load, '.10g')) / ('seed_' + str(seed)) / label
                row, state = core.run_evaluation(cfg, seed, label, directory)
                if control is None:
                    control = state
                elif state['input_sha256'] != control['input_sha256']:
                    raise RuntimeError('The paired variants used different task inputs.')
                row['quantile'] = None if label == 'original_fixed' else frozen[load][label]['quantile']
                rows.append(row)
                print(f'Completed load={load} seed={seed} variant={label}', flush=True)
                pd.DataFrame(rows).to_csv(out / 'raw_by_seed.csv', index=False)

    raw = pd.DataFrame(rows)
    changes = []
    for (load, seed), group in raw.groupby(['load_factor', 'seed'], sort=True):
        control = group[group.variant == 'original_fixed'].iloc[0]
        for _, row in group[group.variant != 'original_fixed'].iterrows():
            changes.append({
                'load_factor': float(load), 'seed': int(seed), 'variant': row.variant,
                'avg_latency_change_pct': float(100 * (row.avg_latency / control.avg_latency - 1)),
                'p95_latency_change_pct': float(100 * (row.p95_latency / control.p95_latency - 1)),
                'trigger_rate_change_pp': float(row.trigger_rate_pct - control.trigger_rate_pct),
            })
    paired = pd.DataFrame(changes)
    paired.to_csv(out / 'paired_changes.csv', index=False)
    table = []
    for (load, variant), group in paired.groupby(['load_factor', 'variant'], sort=True):
        matching = raw[(raw.load_factor == load) & (raw.variant == variant)]
        table.append({
            'load_factor': float(load), 'variant': variant, 'n_seeds': len(group),
            'avg_latency_change_pct_mean': float(group.avg_latency_change_pct.mean()),
            'p95_latency_change_pct_mean': float(group.p95_latency_change_pct.mean()),
            'trigger_rate_pct_mean': float(matching.trigger_rate_pct.mean()),
        })
    pd.DataFrame(table).to_csv(out / 'calibration_summary.csv', index=False)
    core.dump(out / 'completion.json', {
        'completed_evaluation_runs': len(rows), 'independent_calibration_runs': len(loads),
        'evaluation_seeds': args.seeds, 'calibration_seed': args.calibration_seed,
        'completed_utc': datetime.now(timezone.utc).isoformat(),
    })


if __name__ == '__main__':
    main()
