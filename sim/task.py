from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Task:
    """
    Minimal task record used by Env and policies.
    """

    tid: int
    src: int
    arrival_t: float
    cpu_demand: float
    data_mb: float

    dest: Optional[int] = None  # None means cloud
    is_cloud: bool = False
    start_t: Optional[float] = None
    finish_t: Optional[float] = None
    backlog_time_total: float = 0.0
    backlog_work_total: float = 0.0
