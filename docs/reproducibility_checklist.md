# Reproducibility Checklist

This checklist maps the repository contents to the reproducibility issues raised during review.

| Reviewer concern | Repository coverage | Status |
|---|---|---|
| The code repository should be verifiable during review. | `REPRODUCE.md` provides commands for regenerating figures and rerunning experiments. `scripts/smoke_test.sh` provides a quick import/run check. `docs/file_inventory.tsv` lists packaged files. | Addressed; provide the repository through the submission system or an anonymous archival link during double-blind review. |
| A license should be specified. | `LICENSE` provides the MIT License. `README.md` links to it. | Addressed. |
| Hardware and software environment details should be specified. | [environment_and_timing.md](environment_and_timing.md) gives the authors' experimental-server specification, separately inspected local workstation, software setup, and implemented timer boundary. | Documented. |
| Observation-based threshold calibration should be reproducible. | [threshold_calibration.md](threshold_calibration.md), [the runner](../scripts/run_threshold_calibration.py), and [the manuscript configuration](../configs/manuscript_reproduction.json) provide the protocol and executable command. Generated numerical results remain local. | Documentation and execution entry point provided; simulations are run only when requested by the user. |
| Complete experiment configurations should be included. | `configs/` contains the JSON configurations used by the experiments; each packaged result directory also includes `config_snapshot.json` when applicable. | Addressed. |
| Preprocessed trace data should be included. | `data/trace_inputs/` contains the processed Alibaba trace arrival CSV files and `arrivals_h77_load_sweep_0.8_1.6_step0.1_manifest.csv` with repository-relative paths. | Addressed. |
| A Docker or equivalent reproducible environment should be provided. | `Dockerfile`, `environment.yml`, and `requirements.txt` provide container, Conda, and pip setup paths. | Addressed. |
| Instructions should be provided for reproducing paper figures. | `scripts/reproduce_figures.sh` regenerates the final paper figures from packaged results; `REPRODUCE.md` maps manuscript figure numbers to output files and states which result directories contain summaries versus raw-by-seed outputs. | Addressed. |
| Synthetic workload random seeds should be specified. | `docs/random_seeds.md` lists the seed sets, and each packaged experiment stores seeds in `run_meta.json`. | Addressed. |
| The repository should clarify what processed data and numerical results include. | `README.md`, `REPRODUCE.md`, `docs/file_inventory.tsv`, and `results/experiments/` distinguish processed trace inputs, main-figure summary statistics, and raw-by-seed files retained for statistical and supplementary analyses. | Addressed. |

During double-blind review, use only the anonymous repository link provided by the submission system. Add the public GitHub URL after acceptance or when double-blind constraints no longer apply.
