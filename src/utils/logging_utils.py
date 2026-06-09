from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache

import torch.distributed as dist


def _is_main_process() -> bool:
    rank_env = os.environ.get("RANK")
    if rank_env is not None:
        try:
            return int(rank_env) == 0
        except Exception:
            pass

    if dist.is_available() and dist.is_initialized():
        try:
            return dist.get_rank() == 0
        except Exception:
            return False

    return True


class _MainProcessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _is_main_process()


@lru_cache(maxsize=None)
def get_benchmark_logger(name: str = "ntv3_benchmark") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler_exists = any(
        isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is sys.stdout
        for handler in logger.handlers
    )
    if not handler_exists:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.addFilter(_MainProcessFilter())
        logger.addHandler(handler)

    return logger