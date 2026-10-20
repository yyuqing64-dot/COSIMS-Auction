# Observation-based threshold calibration

This document supplies the reproducibility protocol and executable command for the observation-based calibration experiment described in Section 5.6. It is separate from M09 fixed-threshold sensitivity. Results are generated locally when the command is run; no output files are bundled with this documentation update.

## Configuration

The executable entry point is [scripts/run_threshold_calibration.py](../scripts/run_threshold_calibration.py), using [configs/manuscript_reproduction.json](../configs/manuscript_reproduction.json) and [its observation utilities](../scripts/threshold_calibration_support.py).

| Parameter | Reproduction setting |
|---|---|
| Sources and edge nodes | N_U = N_E = 20 |
| Actual scheduling candidate size | K = 3 |
| Synthetic load factors | 1.0, 1.2, 1.4 |
| Arrival horizon | 600 simulator clock units |
| Warm-up | First 60 units |
| Calibration quantiles | Q90, Q95, Q99 |
| Independent calibration seed | 1000 |
| Evaluation seed option in the command below | 0--9, ten paired repetitions |
| Fixed waiting threshold | 6 x 10^11 STU |
| Fixed growth threshold | 5 x 10^11 STU per arrival-clock unit |
| Arrival-surge threshold | 10 |
| Reference-mismatch threshold | 0.5 |
| OT regularization and iteration limit | 0.1 and 35 |

The manuscript reproduction configuration also sets the arrival-statistics window to 1.0, the OT epoch to 1.0, and ranking weights alpha = eta = 1. The remaining configuration fields are explicit in the JSON file. The seed arguments specify the reproduction command; each generated manifest records the actual values used.

## Procedure

1. At each load, use an independent calibration realization and exclude its warm-up observations.
2. Observe the waiting and queue-growth quantities used by the policy, outside its native decision timer and before queue commitment. The utilities sample the OT-preferred edge backlog and the maximum positive backlog-growth signal used by the checked-in policy, retaining the policy's observation-interval clipping.
3. Compute empirical quantiles using linear interpolation. Freeze all load-specific Q90/Q95/Q99 threshold pairs before producing any evaluation outcomes.
4. For each evaluation seed, compare original_fixed, q90, q95, and q99 on the same workload realization. Only wait_tau and slope_th change between the control and calibrated variants.
5. Compute average- and P95-latency percentage changes within each pair, then average those changes over the selected seeds. Report the mean fraction of tasks that activate any correction trigger; do not equate an individual signal's quantile exceedance with the combined trigger rate.

Calibration is load-specific and evaluation is at the same load. The command does not establish cross-load transfer or identify an optimal quantile. The utility validates that observation preserves a plain run's task/queue/RNG trajectory and that paired variants receive the same task inputs. These validations are performed when the experiment is explicitly run, not by reading this documentation.

## Command

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
python scripts/run_threshold_calibration.py \\
  --config configs/manuscript_reproduction.json \\
  --loads 1.0,1.2,1.4 \\
  --quantiles 0.90,0.95,0.99 \\
  --calibration-seed 1000 \\
  --seeds 0-9 \\
  --out results/experiments/threshold_calibration_reproduction
```

The calibration seed must not occur in the evaluation seed list. Use a new output directory; the script refuses to overwrite an existing one. Select a different seed list or output path explicitly when needed. To inspect the interface without running simulations:

```bash
python scripts/run_threshold_calibration.py --help
```

## Locally generated files

| File | Content |
|---|---|
| manifest.json | Parameters, calibration/evaluation seeds, software versions, and source/configuration hashes |
| config_snapshot.json | Fixed control configuration |
| calibration/load_*/thresholds.json | Generated Q90/Q95/Q99 thresholds at each load |
| calibration/load_*/calibration_signals.csv.gz | Post-selection, pre-commitment signal observations with the warm-up inclusion flag |
| calibrated_thresholds.csv | Quantiles, generated thresholds, and calibration observation counts |
| configs/load_*/original_fixed.json and q*.json | Complete control and calibrated configurations |
| raw_by_seed.csv | Run-level metrics, trigger fractions, and decision-time measures |
| paired_changes.csv | Within-seed relative latency changes and trigger-rate differences |
| calibration_summary.csv | Mean paired latency changes, mean trigger rates, and actual repetition counts |
| completion.json | Logical evaluation-run counts and completion time |

The default command has three calibration realizations and 120 logical evaluation runs (3 loads x 4 variants x 10 seeds). Plain and observed executions are compared to keep observation overhead separate from native decision timing. Generated results are not required to be committed to the repository.
