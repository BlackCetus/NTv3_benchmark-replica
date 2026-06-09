from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.logging_utils import get_benchmark_logger


_LOGGER = get_benchmark_logger()


def load_cache(cache_file: Path) -> dict[str, Any]:
    """Load the computation cache from file."""
    if cache_file.exists():
        try:
            with cache_file.open("r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_cache(cache_file: Path, cache_data: dict[str, Any]) -> None:
    """Save the computation cache to file."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w") as f:
        json.dump(cache_data, f, indent=2)


def has_combination_been_computed(
    cache_file: Path,
    model_name: str,
    task_name: str,
    config_hash: str,
) -> bool:
    """Check if a specific model-task-config combination has already been computed."""
    cache = load_cache(cache_file)
    combination_key = f"{model_name}:{task_name}:{config_hash}"

    if combination_key not in cache:
        return False

    cached_entry = cache[combination_key]

    # Verify the cached entry has required fields
    if "timestamp" not in cached_entry or "result_dir" not in cached_entry:
        return False

    # Verify result directory still exists
    result_dir = Path(cached_entry["result_dir"])
    if not result_dir.exists():
        return False

    return True


def mark_combination_computed(
    cache_file: Path,
    model_name: str,
    task_name: str,
    config_hash: str,
    result_dir: Path,
) -> None:
    """Mark a specific model-task-config combination as computed in the cache."""
    cache = load_cache(cache_file)
    combination_key = f"{model_name}:{task_name}:{config_hash}"

    cache[combination_key] = {
        "timestamp": datetime.now().isoformat(),
        "config_hash": config_hash,
        "model_name": model_name,
        "task_name": task_name,
        "result_dir": str(result_dir),
    }

    save_cache(cache_file, cache)


def reconcile_cache_with_results(cache_file: Path, results_root: Path) -> None:
    """Scan the results directory and add any completed runs to the cache.

    This helps recover cache entries when they were not written (e.g., after
    a crash) or were removed by force-recompute logic.
    """
    cache = load_cache(cache_file)

    def iter_config_dirs(root: Path):
        """Yield tuples (model_name, task_name, config_dir) for either layout.

        Supports:
          - task-first: root/<task>/<model>/config_*
          - model-first: root/<model>/<task>/config_*
        """
        for first in root.iterdir():
            if not first.is_dir():
                continue
            children = list(first.iterdir())
            if not children:
                continue
            if any(p.name.startswith("config_") for p in children):
                continue
            for second in children:
                if not second.is_dir():
                    continue
                grandchildren = list(second.iterdir())
                if any(p.name.startswith("config_") for p in grandchildren):
                    for config_dir in second.glob("config_*"):
                        yield first.name, second.name, config_dir

    for model_name, task_name, config_dir in iter_config_dirs(results_root):
        final_metrics = config_dir / "final_metrics.csv"
        task_level_final = config_dir.parent / "final_metrics.csv"
        if not final_metrics.exists() and not task_level_final.exists():
            continue
        parts = config_dir.name.split("config_")
        if len(parts) != 2:
            continue
        config_hash = parts[1]
        key = f"{model_name}:{task_name}:{config_hash}"
        if key not in cache:
            result_dir = config_dir if config_dir.exists() else config_dir.parent
            cache[key] = {
                "timestamp": datetime.now().isoformat(),
                "config_hash": config_hash,
                "model_name": model_name,
                "task_name": task_name,
                "result_dir": str(result_dir),
            }
    save_cache(cache_file, cache)


def cleanup_incomplete_results(
    cache_file: Path,
    results_root: Path,
    apply: bool = False,
) -> None:
    """Remove incomplete config directories and stale cache entries.

    A run directory is considered complete if either:
    - config_dir/final_metrics.csv exists, or
    - config_dir.parent/final_metrics.csv exists (backward compatibility)

    By default this function runs in dry-run mode and only reports what would
    be removed. Set apply=True to perform deletions and cache updates.
    """

    def is_complete(config_dir: Path) -> bool:
        return (config_dir / "final_metrics.csv").exists() or (config_dir.parent / "final_metrics.csv").exists()

    if not results_root.exists():
        _LOGGER.info(f"[Cleanup] Results directory does not exist: {results_root}")
        return

    all_config_dirs = sorted(
        p for p in results_root.rglob("config_*")
        if p.is_dir()
    )
    incomplete_dirs = [p for p in all_config_dirs if not is_complete(p)]

    mode = "APPLY" if apply else "DRY-RUN"
    _LOGGER.info(f"[Cleanup] Mode: {mode}")
    _LOGGER.info(f"[Cleanup] Results root: {results_root}")
    _LOGGER.info(f"[Cleanup] Found config dirs: {len(all_config_dirs)}")
    _LOGGER.info(f"[Cleanup] Incomplete config dirs: {len(incomplete_dirs)}")

    for config_dir in incomplete_dirs:
        _LOGGER.info(f"  - {config_dir}")

    if apply:
        for config_dir in incomplete_dirs:
            shutil.rmtree(config_dir, ignore_errors=True)

    cache = load_cache(cache_file)
    stale_keys: list[str] = []
    for key, entry in list(cache.items()):
        result_dir_raw = entry.get("result_dir")
        if not result_dir_raw:
            stale_keys.append(key)
            continue

        result_dir = Path(result_dir_raw)
        if not result_dir.exists() or not is_complete(result_dir):
            stale_keys.append(key)

    _LOGGER.info(f"[Cleanup] Stale cache entries: {len(stale_keys)}")
    for key in stale_keys:
        _LOGGER.info(f"  - {key}")

    if apply and stale_keys:
        for key in stale_keys:
            cache.pop(key, None)
        save_cache(cache_file, cache)

    if not apply:
        _LOGGER.info("[Cleanup] Dry-run complete. Re-run with --cleanup-apply to delete and update cache.")
    else:
        _LOGGER.info("[Cleanup] Apply complete.")
