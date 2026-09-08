#!/usr/bin/env bash
set -euo pipefail

# Run from repository root.
python scripts/figures/plot_fig5_1_synthetic_latency_outputs2.py
python scripts/figures/plot_fig5_2_delay_distribution_outputs2.py
python scripts/figures/plot_fig5_3_trace_latency_outputs2.py
python scripts/figures/plot_fig5_4_outputs2.py
python scripts/figures/plot_fig5_5_mechanism_variants.py
python scripts/figures/plot_fig5_6_trigger_overhead.py
