# Random Seeds Used in the Experiments

This document records the random seeds used by the packaged experiment outputs and by the reproduction commands in `REPRODUCE.md`.

All seeds are integer seeds passed to the simulator. The exact seed list for each packaged run is also stored in the corresponding `run_meta.json` file under `results/experiments/`.

| Experiment | Packaged output directory | Seeds |
|---|---|---|
| Synthetic main evaluation, Fig. 3 | `results/experiments/synthetic_main_fig5_1/` | `0,1,2,3,4` |
| Trace replay main evaluation, Fig. 5 | `results/experiments/trace_main_fig5_3/` | `0,1,2,3,4` |
| Trace trade-off, Fig. 6 | `results/experiments/trace_tradeoff_fig5_4/` | `0,1,2,3,4` |
| Mechanism variants and trigger overhead, Fig. 7 and Fig. 8 | `results/experiments/mechanism_variants_fig5_5_5_6/` | `0,1,2,3,4` |
| Candidate-size sensitivity | `results/experiments/k_sensitivity/` | `0,1,2,3,4` |
| M08 trigger ablation | `results/experiments/trigger_ablation_m08/` | `0,1,2,3,4` |
| M09 trigger-threshold sensitivity | `results/experiments/threshold_sensitivity_m09/` | `0,1,2,3,4` |
| M10 statistical validation | `results/experiments/statistical_validation_m10/` | `0,1,2,3,4,5,6,7,8,9` |
| M10 trace statistical validation | `results/experiments/statistical_validation_trace_m10/` | `0,1,2,3,4,5,6,7,8,9` |
| M11 scalability/runtime analysis | `results/experiments/scalability_m11/` | `0,1,2,3,4` |
| M12 OT demand construction validation | `results/experiments/ot_demand_construction_m12/` | `0,1,2,3,4` |
| M13 OT cost construction validation | `results/experiments/ot_cost_construction_m13/` | `0,1,2,3,4,5,6,7,8,9` |

The manuscript reports statistical validation with 10 seeds where significance testing is needed. Several design-validation and sensitivity experiments are packaged with 5 seeds because their purpose is to check mechanism robustness rather than to establish the primary performance comparison.
