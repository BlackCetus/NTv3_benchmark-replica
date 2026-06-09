from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.utils.logging_utils import get_benchmark_logger


_LOGGER = get_benchmark_logger()


def _env_int(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return int(value)


def _init_ddp() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    cuda_available = torch.cuda.is_available()
    device: torch.device

    if world_size > 1:
        if cuda_available:
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                raise RuntimeError("WORLD_SIZE>1 but no CUDA devices found for NCCL backend.")
            device_index = int(local_rank) % num_gpus
            torch.cuda.set_device(device_index)
            device = torch.device(f"cuda:{device_index}")
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"

        dist.init_process_group(backend=backend)
    else:
        if cuda_available:
            num_gpus = torch.cuda.device_count()
            device_index = int(local_rank) % max(1, num_gpus)
            if num_gpus > 0:
                torch.cuda.set_device(device_index)
                device = torch.device(f"cuda:{device_index}")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")

    return local_rank, rank, world_size, device


def _is_main_process() -> bool:
    # Prefer explicit RANK env var when present (works reliably with torchrun).
    rank_env = os.environ.get("RANK")
    if rank_env is not None:
        try:
            return int(rank_env) == 0
        except Exception:
            pass

    # Fallback to distributed API if available.
    if dist.is_available() and dist.is_initialized():
        try:
            return dist.get_rank() == 0
        except Exception:
            return False

    # Default to True for single-process runs.
    return True


def _build_loader_with_distributed_sampler(
    dataset,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=True,
        seed=seed,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,        
    )


def _log_gpu_memory(label: str, device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}

    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    total = torch.cuda.get_device_properties(device).total_memory / 1024**2
    free = total - allocated

    _LOGGER.info(f"{label}: Allocated={allocated:.0f}MiB / Reserved={reserved:.0f}MiB / Free={free:.0f}MiB")
    return {"allocated_mb": allocated, "reserved_mb": reserved, "free_mb": free}