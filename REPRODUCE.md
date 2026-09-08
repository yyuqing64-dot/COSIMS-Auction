# Reproducing the COSIMS-Auction Experiments

This document lists the commands needed to reproduce the main figures and the supplementary experiments reported in the manuscript. Commands are intended to be run from the repository root.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Conda alternative:

```bash
conda env create -f environment.yml
conda activate cosims-auction
```

Docker alternative:

```bash
docker build -t cosims-auction .
docker run --rm cosims-auction
```

Run a quick import check:

```bash
bash scripts/smoke_test.sh
```

## 2. Regenerate Paper Figures from Packaged Results

This is the fastest way to verify that the repository matches the manuscript figures.

```bash
bash scripts/reproduce_figures.sh
```

The regenerated figures are written to:

```text
results/figures/
```

Figure mapping:

| Manuscript item | Repository artifact |
|---|---|
| Fig. 3 | `results/figures/fig5_1_synthetic_latency.pdf` |
| Fig. 4 | `results/figures/fig5_2_task_delay_distribution.pdf` |
| Fig. 5 | `results/figures/fig5_3_trace_latency.pdf` |
| Fig. 6 | `results/figures/fig5_4_trace_tradeoff.pdf` |
| Fig. 7 | `results/figures/fig5_5_mechanism_variant_normalized.pdf` |
| Fig. 8 | `results/figures/fig5_6_trigger_overhead.pdf` |

`fig5_4_trace_tradeoff_relative.pdf` is kept as a descriptive alias for the same Fig. 6 relative trade-off plot.

## 3. Main Experiment Outputs Included in the Repository

The packaged outputs used by the manuscript are stored under:

```text
results/experiments/
```

Important directories:

| Experiment | Output directory |
|---|---|
| Synthetic main evaluation | `results/experiments/synthetic_main_fig5_1/` |
| Trace replay evaluation | `results/experiments/trace_main_fig5_3/` |
| Trace trade-off vs. Cap-Greedy | `results/experiments/trace_tradeoff_fig5_4/` |
| Mechanism variants and trigger overhead | `results/experiments/mechanism_variants_fig5_5_5_6/` |
| K sensitivity | `results/experiments/k_sensitivity/` |
| M08 trigger ablation | `results/experiments/trigger_ablation_m08/` |
| M09 threshold sensitivity | `results/experiments/threshold_sensitivity_m09/` |
| M10 statistical validation | `results/experiments/statistical_validation_m10/` |
| M11 scalability/runtime | `results/experiments/scalability_m11/` |
| M12 OT demand construction | `results/experiments/ot_demand_construction_m12/` |
| M13 OT cost construction | `results/experiments/ot_cost_construction_m13/` |

## 4. Re-run Main Synthetic and Trace Experiments

The general runner can be used for the main synthetic and trace replay evaluations.

Synthetic main experiment:

```bash
python eval/runner.py \
  --config configs/auction_B_recommended_sqlf_cloud.json \
  --out results/experiments/synthetic_main_rerun/summary_mean_std.csv \
  --loads 0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6 \
  --seeds 0-9 \
  --policies rr,greedy_latency,sqlf,cap_greedy,cosims_sinkhorn,cosims_auction_B
```

Trace replay main experiment:

```bash
python eval/runner.py \
  --config configs/auction_B_recommended_sqlf_cloud.json \
  --out results/experiments/trace_main_rerun/summary_mean_std.csv \
  --loads 1.0 \
  --seeds 0-9 \
  --policies rr,greedy_latency,sqlf,cap_greedy,cosims_sinkhorn,cosims_auction_B \
  --trace_manifest data/trace_inputs/arrivals_h77_load_sweep_0.8_1.6_step0.1_manifest.csv \
  --trace_load_factors 0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6
```

The full trace replay can take noticeably longer than regenerating figures because the trace CSV files are large.

## 5. Re-run Supplementary Experiments

M08 trigger ablation:

```bash
python scripts/run_trigger_ablation.py \
  --out-dir results/experiments/trigger_ablation_m08_rerun
```

M10 statistical validation:

```bash
python scripts/run_m10_statistical_validation.py \
  --out-dir results/experiments/statistical_validation_m10_rerun
```

For a faster M10 check, run only the synthetic part:

```bash
python scripts/run_m10_statistical_validation.py \
  --scenarios synthetic_main,mechanism_overhead \
  --out-dir results/experiments/statistical_validation_m10_synthetic_rerun
```

M11 scalability/runtime analysis:

```bash
python scripts/run_m11_scalability.py \
  --n-nodes 20,50,100,150,200 \
  --load 1.2 \
  --seeds 0-4 \
  --out-dir results/experiments/scalability_m11_rerun
```

M12 OT demand construction validation:

```bash
python scripts/run_m12_synthetic_workload_weighted_ot.py \
  --seeds 0-4 \
  --loads 1.0,1.2,1.4 \
  --out-dir results/experiments/ot_demand_construction_m12_rerun
```

M13 OT cost construction validation:

```bash
python scripts/run_m13_ot_cost_construction.py \
  --seeds 0-9 \
  --loads 1.0,1.2,1.4 \
  --out-dir results/experiments/ot_cost_construction_m13_rerun
```

## 6. Re-run K Sensitivity Figure/Table

The packaged K-sensitivity results can be reprocessed with:

```bash
python scripts/plot_k_sensitivity.py
```

This writes compact CSVs and the K-sensitivity figure under:

```text
results/experiments/k_sensitivity/
```

## 7. Notes on Randomness and Exact Reproduction

The reported results are based on fixed random seeds listed in [`docs/random_seeds.md`](docs/random_seeds.md) and in each experiment's `run_meta.json`. Minor differences in measured wall-clock or per-task decision time may occur across machines because those metrics depend on CPU, Python runtime, and system load. Latency and trigger-rate metrics should match up to ordinary floating-point variation when the same seeds and configurations are used.

## 8. Trace Data and Packaged Numerical Results

Trace input files are included in:

```text
data/trace_inputs/
```

These files are the processed Alibaba trace arrival streams used by the trace replay experiments. The manifest `data/trace_inputs/arrivals_h77_load_sweep_0.8_1.6_step0.1_manifest.csv` uses repository-relative paths so the trace experiments can run after cloning the repository.

Packaged numerical results are stored under `results/experiments/`. Main figure directories include the summary statistics used to regenerate the exact paper figures. Statistical-validation and supplementary experiment directories additionally include raw per-seed CSV files where those values are needed for confidence intervals, significance tests, or mechanism comparisons.

## 9. Reproducibility Checklist

The file [`docs/reproducibility_checklist.md`](docs/reproducibility_checklist.md) maps reviewer reproducibility concerns to the corresponding repository files, including the license, Docker/environment files, trace inputs, experiment configurations, figure commands, and seed lists.
