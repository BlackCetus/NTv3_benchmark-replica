from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from src.utils.ddp_runtime_utils import _is_main_process
from src.utils.logging_utils import get_benchmark_logger


_LOGGER = get_benchmark_logger()


class TimingRecorder:
    """Collects wall-clock durations for named benchmark steps.

    Steps are timed on every rank (so the `with` block itself doesn't skew
    DDP synchronization), but only rank 0 records/logs them, since all ranks
    run in lockstep via collective ops inside training/eval anyway.
    """

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def record(self, *, model: str, task: str, step: str, seconds: float) -> None:
        self._rows.append({"model": model, "task": task, "step": step, "seconds": seconds})

    @contextmanager
    def timed(self, *, model: str, task: str, step: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if _is_main_process():
                self.record(model=model, task=task, step=step, seconds=elapsed)
                _LOGGER.info(f"[Timing] {model} | {task} | {step}: {elapsed:.2f}s")

    def save_csv(self, output_path: Path) -> None:
        if not _is_main_process() or not self._rows:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._rows).to_csv(output_path, index=False)

    def print_summary(self, top_n: int = 20) -> None:
        if not _is_main_process() or not self._rows:
            return
        df = pd.DataFrame(self._rows)
        totals = (
            df.groupby(["model", "task", "step"], as_index=False)["seconds"]
            .sum()
            .sort_values("seconds", ascending=False)
        )

        header = "[Timing Summary - Slowest Steps]"
        sep = "-" * 41
        _LOGGER.info(header)
        _LOGGER.info(sep)
        for row in totals.head(top_n).itertuples(index=False):
            _LOGGER.info(f"  {row.seconds:>10.2f}s  {row.model} | {row.task} | {row.step}")
        _LOGGER.info(sep)
        _LOGGER.info(f"  Total tracked time: {df['seconds'].sum():.2f}s")
        _LOGGER.info(sep)
