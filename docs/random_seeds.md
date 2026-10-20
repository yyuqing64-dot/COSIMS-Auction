# Random seeds for reproduction commands

The manuscript reproduction commands use ten paired evaluation seeds, 0--9, unless a command explicitly selects another list. Methods compared within a seed must use the same workload stream, configuration, and initial state.

| Reproduction command | Seed option |
|---|---|
| Main synthetic and trace comparisons | --seeds 0-9 |
| Candidate-size sweep | --seeds 0-9 |
| Observation-based threshold calibration | --calibration-seed 1000 --seeds 0-9 |
| Existing M08--M13 entry points | Use their --seeds option when supported; consult --help for the interface |

The threshold-calibration seed is independent of every evaluation seed. Calibration thresholds are frozen before evaluation, and the generated manifest records both seed lists. A command's selected seeds and the metadata written by its runner determine the actual replication count. When inspecting an existing result file, use its associated run_meta.json rather than inferring its seeds from a command example.
