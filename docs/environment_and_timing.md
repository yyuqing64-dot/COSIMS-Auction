# Experiment environment and decision-time measurement

## Experimental server

The authors supplied the following configuration for the experimental server:

| Item | Configuration |
|---|---|
| CPU allocation | 15 vCPUs, Intel(R) Xeon(R) Platinum 8474C |
| Memory | 80 GB |
| GPU | One NVIDIA RTX 4090D, 24 GB |
| System image | Ubuntu 22.04 |
| Image Python | Python 3.12 |
| Image PyTorch | 2.8.0 |
| Image CUDA | 12.8 |
| System disk | 30 GB |
| Data disk | 50 GB SSD |

The vCPU allocation is not a count of physical cores. GPU and framework versions describe the supplied image; the scheduling simulation and the threshold-calibration entry point use CPU-based NumPy/Pandas operations. PyTorch and CUDA are not required to execute these scheduling experiments. Package versions inside a selected Python environment are recorded by each runner and may differ from the image defaults.

## Local workstation

The local workstation is used for manuscript preparation, file organization, and figure/result inspection. Its system information was read on 2026-10-20:

| Item | Configuration |
|---|---|
| CPU | Intel Core i5-1035G1, 1.00 GHz |
| Logical processors | 8 |
| Physical memory reported by Windows | 16,944,979,968 bytes (approximately 15.78 GiB; 16 GB class) |
| Operating system | Windows 10, build 19045, 64-bit |

This workstation configuration is reported separately from the experimental server and is not used to identify the server timing platform.

## Software installation

[Dockerfile](../Dockerfile), [requirements.txt](../requirements.txt), and [environment.yml](../environment.yml) provide installation paths. See [REPRODUCE.md](../REPRODUCE.md) for commands. The experiment-level run_meta.json files and the calibration runner's manifest.json record the environment actually selected for each run.

## Per-task decision-time measurement

The timing boundary follows select_dest() and get_cost_stats() in [policies/cosims_auction_B.py](../policies/cosims_auction_B.py):

1. time.perf_counter() starts after policy initialization and the initial task demand counters, immediately before _maybe_update_epoch().
2. The interval includes an epoch update when invoked by that task, reference-row processing, trigger checking, candidate selection, and the final edge-cloud comparison.
3. Timing stops before the destination is returned. Queue commitment subsequently performed by the environment is outside this policy timer.
4. decision_time_ms_mean equals 1000 times accumulated decision seconds divided by the number of recorded task decisions. The runner retrieves this value after env.run(); latency warm-up filtering does not automatically filter the native policy timer.

OT update time is measured separately and averaged over OT updates. Per-task decision time is averaged over task decisions and includes an update when it occurs. These metrics have different denominators and should not be added as disjoint component costs.

The manuscript reports aggregate per-task decision time. A comprehensive component-level breakdown is deferred to future prototype-based evaluation. Internal implementation counters do not by themselves constitute that evaluation.

Decision time is wall-clock milliseconds. Latency is reported in simulator time units (STU), with scaling given in each figure or table. The approximately 0.1 ms description summarizes the measured magnitude; it is not a platform-independent constant. Use the applicable configuration, seeds, and run metadata when comparing timings across experiments.
