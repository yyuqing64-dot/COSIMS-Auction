from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Network:
    """Simple latency matrix wrapper."""

    lat_matrix_ms: np.ndarray

    @property
    def lat_ms(self) -> np.ndarray:
        """Alias used by policies; returns latency matrix in ms."""
        return self.lat_matrix_ms

    @property
    def lat_s(self) -> np.ndarray:
        """Latency matrix in seconds."""
        return self.lat_matrix_ms / 1000.0

    @classmethod
    def random_latency_matrix(
        cls,
        n: int,
        rng: np.random.Generator,
        ms_min: float,
        ms_max: float,
        diagonal_ms: float = 1.0,
        symmetric: bool = True,
    ) -> "Network":
        mat = rng.uniform(low=ms_min, high=ms_max, size=(n, n))
        if symmetric:
            # make symmetric while keeping diagonal intact
            mat = (mat + mat.T) / 2.0
        np.fill_diagonal(mat, diagonal_ms)
        return cls(lat_matrix_ms=mat)

    def latency_ms(self, i: int, j: int) -> float:
        return float(self.lat_matrix_ms[i, j])

    def latency_s(self, i: int, j: int) -> float:
        return self.latency_ms(i, j) / 1000.0

