# COSIMS-Auction Reproducibility Package

This repository contains the simulation code, experiment configurations, final plotting scripts, and result artifacts for the COSIMS-Auction edge-cloud scheduling experiments.

## Repository Layout

```text
configs/                 Experiment configuration files
sim/                     Simulation environment and network/task models
policies/                Scheduling policies and COSIMS variants
eval/                    Metrics and experiment runner utilities
scripts/                 Experiment scripts for M08-M13 and sensitivity analyses
scripts/figures/         Scripts used to regenerate the final paper figures
data/trace_inputs/       Trace replay input CSV files
results/figures/         Final paper figures used in the manuscript
results/tables/          Figure/table CSVs and LaTeX table snippets
results/experiments/     Main and supplementary experiment outputs
```

## Included Experiments

The `results/experiments/` directory contains the final outputs for:

- Synthetic main evaluation (`synthetic_main_fig5_1`)
- Trace replay main evaluation (`trace_main_fig5_3`)
- Trace trade-off with Cap-Greedy (`trace_tradeoff_fig5_4`)
- Mechanism variants and trigger overhead (`mechanism_variants_fig5_5_5_6`)
- Candidate-size sensitivity (`k_sensitivity`)
- Trigger ablation, M08 (`trigger_ablation_m08`)
- Trigger-threshold sensitivity, M09 (`threshold_sensitivity_m09`)
- Statistical validation, M10 (`statistical_validation_m10`, `statistical_validation_trace_m10`)
- Scalability/runtime analysis, M11 (`scalability_m11`)
- OT demand construction validation, M12 (`ot_demand_construction_m12`)
- OT cost construction validation, M13 (`ot_cost_construction_m13`)


## Reproduction Guide

See [`REPRODUCE.md`](REPRODUCE.md) for commands to regenerate figures, re-run the main experiments, and re-run the supplementary M08-M13 experiments.

For a reviewer-facing checklist of reproducibility items, see [`docs/reproducibility_checklist.md`](docs/reproducibility_checklist.md). The random seeds used in the packaged outputs are listed in [`docs/random_seeds.md`](docs/random_seeds.md).

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The review repository URL should be provided through the submission system or an anonymous archival link during double-blind review. 

## License

This artifact is released under the MIT License. See [`LICENSE`](LICENSE).

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternatively, the Conda environment can be created with:

```bash
conda env create -f environment.yml
conda activate cosims-auction
```

Docker users can run the quick check in a container:

```bash
docker build -t cosims-auction .
docker run --rm cosims-auction
```

## Quick Check

```bash
bash scripts/smoke_test.sh
```

## Regenerate Figures

The final paper figures can be regenerated with:

```bash
bash scripts/reproduce_figures.sh
```

The scripts use repository-relative paths and write regenerated figures to `results/figures/` and regenerated plotting tables to `results/tables/`.

## Notes on Units

Synthetic latency plots report completion delay in scaled simulator time units, e.g. `10^9 STU`. Trace replay latency plots report completion delay in native simulator time units (`STU`). Decision overhead is reported in milliseconds per task.

## Data and Results Scope

The processed trace data mentioned in the manuscript refers to the preprocessed Alibaba trace arrival CSV files in `data/trace_inputs/` and their manifest CSV. These files contain the replay-ready arrival streams used by the trace experiments, so reviewers do not need to regenerate them from the external Alibaba Cluster Trace. They account for most of the repository size. Each file is below GitHub's single-file size limit, but the directory is intentionally kept because it is needed for complete trace replay reproducibility.

The numerical results under `results/experiments/` include `run_meta.json` files, configuration snapshots, summary tables for the main manuscript figures, and raw-by-seed CSV files for statistical validation and supplementary analyses where retained. Main figure directories provide the packaged summary statistics used to regenerate the paper figures; experiments that support per-seed statistical analysis include `raw_by_seed.csv` or similarly named raw files.

## Manuscript File Names

The manuscript may refer to Fig. 6 as `fig5_4_trace_tradeoff.pdf`; the repository also keeps `fig5_4_trace_tradeoff_relative.pdf` as the descriptive filename for the same relative trade-off plot.
