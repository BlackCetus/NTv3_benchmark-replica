from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator

import torch

from src.utils.logging_utils import get_benchmark_logger


_LOGGER = get_benchmark_logger()

# Top-level phases summed per batch; forward/backward are additionally split
# into backbone/head via module hooks (see _register_hooks).
_TOP_LEVEL_PHASES = ("data_fetch", "to_device", "forward", "loss", "backward", "optimizer", "metrics")


class StepTimer:
    """Fine-grained per-phase timer for a single training epoch.

    Wraps the train batch loop and attributes wall-clock time to phases
    (data_fetch, to_device, forward, loss, backward, optimizer, metrics),
    additionally splitting forward/backward into backbone vs head via module
    hooks. GPU work is attributed correctly by calling torch.cuda.synchronize()
    at each phase boundary — this adds a small, roughly uniform overhead per
    phase, so the numbers are best read as a *relative* breakdown.

    A disabled timer is a cheap no-op: phase() yields immediately and no hooks
    are registered, so normal runs are unaffected.
    """

    def __init__(self, model: torch.nn.Module, enabled: bool, device: torch.device | str) -> None:
        self.device = device
        dev_type = getattr(device, "type", str(device))
        self.enabled = bool(enabled) and torch.cuda.is_available() and dev_type == "cuda"
        self.totals: dict[str, float] = defaultdict(float)
        self.count = 0
        self._handles: list[Any] = []
        self._starts: dict[str, float] = {}
        self._iter_end: float | None = None
        if self.enabled:
            self._register_hooks(model)

    # -- internals -------------------------------------------------------
    def _sync(self) -> None:
        torch.cuda.synchronize(self.device)

    @staticmethod
    def _t() -> float:
        return time.perf_counter()

    def _register_hooks(self, model: torch.nn.Module) -> None:
        inner = model.module if hasattr(model, "module") else model
        backbone = getattr(inner, "backbone", None)
        head = getattr(inner, "head", None)
        if backbone is not None:
            self._add_fwd_hooks(backbone, "fwd_backbone")
            self._add_bwd_hooks(backbone, "bwd_backbone")
        if head is not None:
            self._add_fwd_hooks(head, "fwd_head")
            self._add_bwd_hooks(head, "bwd_head")

    def _add_fwd_hooks(self, module: torch.nn.Module, name: str) -> None:
        def pre(_m: Any, _inp: Any) -> None:
            self._sync()
            self._starts[name] = self._t()

        def post(_m: Any, _inp: Any, _out: Any) -> None:
            self._sync()
            if name in self._starts:
                self.totals[name] += self._t() - self._starts[name]

        self._handles.append(module.register_forward_pre_hook(pre))
        self._handles.append(module.register_forward_hook(post))

    def _add_bwd_hooks(self, module: torch.nn.Module, name: str) -> None:
        def pre(_m: Any, _gout: Any) -> None:
            self._sync()
            self._starts[name] = self._t()

        def post(_m: Any, _gin: Any, _gout: Any) -> None:
            self._sync()
            if name in self._starts:
                self.totals[name] += self._t() - self._starts[name]

        # register_full_backward_pre_hook requires torch >= 2.0; guard both so a
        # missing/failed hook never breaks training. When the backbone is frozen
        # these simply never fire, leaving bwd_backbone == 0 (expected).
        try:
            self._handles.append(module.register_full_backward_pre_hook(pre))
        except Exception:
            pass
        try:
            self._handles.append(module.register_full_backward_hook(post))
        except Exception:
            pass

    # -- loop API --------------------------------------------------------
    def begin_epoch(self) -> None:
        if not self.enabled:
            return
        self._sync()
        self._iter_end = self._t()

    def data_fetch_boundary(self) -> None:
        """Record time spent waiting for the dataloader to yield this batch."""
        if not self.enabled or self._iter_end is None:
            return
        self.totals["data_fetch"] += self._t() - self._iter_end

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync()
        start = self._t()
        try:
            yield
        finally:
            self._sync()
            self.totals[name] += self._t() - start

    def end_iter(self) -> None:
        if not self.enabled:
            return
        self.count += 1
        self._sync()
        self._iter_end = self._t()

    def remove(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles = []

    # -- reporting -------------------------------------------------------
    def log_epoch(self, epoch: int) -> None:
        if not self.enabled or self.count == 0:
            return

        n = self.count
        get = self.totals.get
        top_sum = sum(get(p, 0.0) for p in _TOP_LEVEL_PHASES)
        if top_sum <= 0.0:
            return

        def ms(seconds: float) -> float:
            return 1000.0 * seconds / n

        def pct(seconds: float) -> float:
            return 100.0 * seconds / top_sum

        fwd = get("forward", 0.0)
        bwd = get("backward", 0.0)
        fwd_bb, fwd_h = get("fwd_backbone", 0.0), get("fwd_head", 0.0)
        bwd_bb, bwd_h = get("bwd_backbone", 0.0), get("bwd_head", 0.0)
        fwd_other = max(0.0, fwd - fwd_bb - fwd_h)
        bwd_other = max(0.0, bwd - bwd_bb - bwd_h)

        sep = "-" * 41
        _LOGGER.info(f"[Step Timing][Epoch {epoch}] (avg per batch over {n:,} batches)")
        _LOGGER.info(sep)
        _LOGGER.info(f"  data_fetch:   {ms(get('data_fetch', 0.0)):8.2f} ms  ({pct(get('data_fetch', 0.0)):5.1f}%)")
        _LOGGER.info(f"  to_device:    {ms(get('to_device', 0.0)):8.2f} ms  ({pct(get('to_device', 0.0)):5.1f}%)")
        _LOGGER.info(f"  forward:      {ms(fwd):8.2f} ms  ({pct(fwd):5.1f}%)")
        _LOGGER.info(f"    backbone:   {ms(fwd_bb):8.2f} ms")
        _LOGGER.info(f"    head:       {ms(fwd_h):8.2f} ms")
        _LOGGER.info(f"    other:      {ms(fwd_other):8.2f} ms")
        _LOGGER.info(f"  loss:         {ms(get('loss', 0.0)):8.2f} ms  ({pct(get('loss', 0.0)):5.1f}%)")
        _LOGGER.info(f"  backward:     {ms(bwd):8.2f} ms  ({pct(bwd):5.1f}%)")
        _LOGGER.info(f"    backbone:   {ms(bwd_bb):8.2f} ms")
        _LOGGER.info(f"    head:       {ms(bwd_h):8.2f} ms")
        _LOGGER.info(f"    other:      {ms(bwd_other):8.2f} ms")
        _LOGGER.info(f"  optimizer:    {ms(get('optimizer', 0.0)):8.2f} ms  ({pct(get('optimizer', 0.0)):5.1f}%)")
        _LOGGER.info(f"  metrics:      {ms(get('metrics', 0.0)):8.2f} ms  ({pct(get('metrics', 0.0)):5.1f}%)")
        _LOGGER.info(sep)
        _LOGGER.info(f"  measured/batch: {ms(top_sum):8.2f} ms")
        _LOGGER.info(sep)
