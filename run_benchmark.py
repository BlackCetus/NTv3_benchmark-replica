from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import math
from typing import Any

import pandas as pd
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data import (
    create_dataset_and_loader,
    create_targets_scaling_fn,
    prepare_genomics_inputs,
)
from src.utils.cache_utils import (
    cleanup_incomplete_results as _cleanup_incomplete_results,
    has_combination_been_computed as _has_combination_been_computed,
    mark_combination_computed as _mark_combination_computed,
    reconcile_cache_with_results as _reconcile_cache_with_results,
)
from src.engine import evaluate, train_one_epoch
from src.metrics import get_loss_and_metrics
from src.models import build_benchmark_model
from src.utils.ddp_runtime_utils import _init_ddp, _is_main_process, _log_gpu_memory, _build_loader_with_distributed_sampler
from src.utils.logging_utils import get_benchmark_logger
from src.utils.plot_utils import generate_comparative_plots as _generate_comparative_plots
from src.utils.plot_utils import generate_run_plots as _generate_run_plots
from src.utils.report_formatting import (
    print_budget_summary,
    print_final_eval,
    print_run_banner,
    print_status_block,
)


_LOGGER = get_benchmark_logger()


def _sanitize_dir_name(name: str) -> str:
    return name.replace(os.sep, "_").replace(" ", "_")


def _species_key(species: str) -> str:
    return str(species).strip().lower().replace("-", "_").replace(" ", "_")


def _parse_species_token_budgets(settings_cfg: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Parse optional per-species benchmark token budgets from external file or config settings.
    
    Loads from configs/species_token_budgets.yaml if available, otherwise falls back to
    species_token_budgets in settings_cfg.
    """
    # Try to load from external species_token_budgets.yaml file
    budgets_file_path = settings_cfg.get("species_token_budgets_file")
    if budgets_file_path and os.path.isfile(budgets_file_path):
        budgets_file = budgets_file_path
    else:
        budgets_file = os.path.join(os.path.dirname(__file__), "configs", "species_token_budgets.yaml")
    raw: dict = {}
    
    if os.path.isfile(budgets_file):
        try:
            with open(budgets_file) as f:
                budgets_data = yaml.safe_load(f)
                if budgets_data and isinstance(budgets_data, dict):
                    raw = budgets_data
        except Exception as e:
            if _is_main_process():
                _LOGGER.warning("[Budget File] Failed to load species_token_budgets.yaml")
                _LOGGER.warning(f"  Error: {e}")
    
    if not isinstance(raw, dict):
        return {}

    parsed: dict[str, dict[str, int]] = {}
    for species_name, values in raw.items():
        if not isinstance(values, dict):
            continue

        train_tokens = values.get("train")
        val_tokens = values.get("val")
        try:
            if train_tokens is None or val_tokens is None:
                continue
            # Preserve original YAML keys: 'train' and 'val'
            parsed[_species_key(str(species_name))] = {
                "train": int(train_tokens),
                "val": int(val_tokens),
            }
        except Exception:
            continue

    return parsed


def _find_eval_split(splits_df) -> str:
    available = set(splits_df["split"].astype(str).unique())
    if "val" in available:
        return "val"
    if "valid" in available:
        return "valid"
    if "test" in available:
        return "test"
    for split_name in sorted(available):
        if split_name != "train":
            return split_name
    return "train"


def _build_model(
    model_cfg: dict[str, Any],
    task_type: str,
    num_tracks: int,
    keep_target_center_fraction: float,
    device: torch.device,
) -> torch.nn.Module:
    model_cfg["keep_target_center_fraction"] = keep_target_center_fraction
    model_type = str(model_cfg.get("type", "")).strip().lower()
    model = build_benchmark_model(model_cfg=model_cfg, task_type=task_type, num_tracks=num_tracks)
    # Some HF remote-code backbones (including NTv3) are not safe to promote to bf16
    # because their internal activations remain fp32.
    if model_type not in {"hf", "huggingface", "transformers"}:
        model = model.bfloat16()
    return model.to(device)



def _pick_score(metrics: dict[str, float], task_type: str) -> float:
    if task_type == "regression":
        if "mean/pearson" in metrics:
            return float(metrics["mean/pearson"])
        return float(metrics.get("loss", 0.0))

    for key in ("mean/auprc", "mean/accuracy", "mean/f1", "accuracy", "f1", "loss"):
        if key in metrics:
            return float(metrics[key])
    return 0.0


def _save_metrics_csv(metrics: dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output_path, index=False)


def _snapshot_first_batch(dataloader: Any) -> Any:
    for batch in dataloader:
        return batch
    raise RuntimeError("Unable to snapshot a batch for overfit debug mode.")


def _get_config_hash(config_dict: dict[str, Any]) -> str:
    """Generate a hash of the configuration for cache validation."""
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:8]


def main(
    config_path: str | Path | None = None,
    cleanup_incomplete: bool = False,
    cleanup_apply: bool = False,
    overfit_debug: bool = False,
) -> None:
    if cleanup_incomplete:
        results_root = Path(__file__).resolve().parent / "results"
        cache_file = results_root / ".benchmark_cache.json"
        _cleanup_incomplete_results(
            cache_file=cache_file,
            results_root=results_root,
            apply=cleanup_apply,
        )
        return

    local_rank, rank, world_size, device = _init_ddp()

    try:
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "configs" / "core_10_tasks.yaml"
        else:
            config_path = Path(config_path)

        with config_path.open("r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f)

        settings_cfg = full_cfg.get("settings", {})
        global_cache_dir = settings_cfg.get("data_cache_dir", "data")
        models_cfg = full_cfg.get("models_to_test", [])
        default_window_size = int(settings_cfg.get("window_size", 1024))
        tasks_cfg = full_cfg.get("tasks", {})
        species_token_budgets = _parse_species_token_budgets(settings_cfg)

        # Optional reproducibility seed from config: sets Python, numpy, and torch RNGs
        seed = settings_cfg.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
                import random

                import numpy as np

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                os.environ['PYTHONHASHSEED'] = str(seed)
                try:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                except Exception:
                    pass
                if _is_main_process():
                    _LOGGER.info(f"[Seed] Using seed: {seed}")
            except Exception:
                if _is_main_process():
                    _LOGGER.warning("[Seed] Invalid seed in settings, ignoring.")

        results_root = Path(__file__).resolve().parent / "results"
        results_root.mkdir(parents=True, exist_ok=True)
        
        # Cache file to track computed model-task combinations
        cache_file = results_root / ".benchmark_cache.json"
        # Reconcile cache with any existing results on disk (recover missing entries)
        try:
            _reconcile_cache_with_results(cache_file, results_root)
        except Exception:
            pass

        epochs = settings_cfg.get("epochs", 10)
        default_training_fraction = float(settings_cfg.get("training_fraction", 1.0))
        default_validation_fraction = float(settings_cfg.get("validation_fraction", default_training_fraction))

        for model_cfg in models_cfg:
            model_name = _sanitize_dir_name(str(model_cfg["name"]))
            model_window_size = int(model_cfg.get("window_size", default_window_size))
            model_batch_size = int(model_cfg.get("batch_size", settings_cfg.get("batch_size", 2)))

            for task_name, task_cfg in tasks_cfg.items():
                task_type = str(task_cfg["type"]).lower()
                species = str(task_cfg["species"])
                num_classes = int(task_cfg.get("num_classes", 2))
                target_column = task_cfg.get("target_column")
                bigwig_ids = task_cfg.get("bigwig_ids")
                bed_ids = task_cfg.get("bed_ids")
                keep_target_center_fraction = float(task_cfg.get("keep_target_center_fraction", 0.375))
                window_size = int(task_cfg.get("window_size", model_window_size))

                batch_size = model_batch_size
                num_workers = int(task_cfg.get("num_workers", 0))
                train_num_samples = int(task_cfg.get("train_num_samples", 256))
                eval_num_samples = int(task_cfg.get("eval_num_samples", 128))
                learning_rate = float(task_cfg.get("learning_rate", 1e-05))
                force_recompute = bool(task_cfg.get("force_recompute", False))
                training_fraction = float(task_cfg.get("training_fraction", default_training_fraction))
                validation_fraction = float(task_cfg.get("validation_fraction", default_validation_fraction))

                if training_fraction <= 0.0 or training_fraction > 1.0:
                    raise ValueError(f"training_fraction must be in (0, 1], got {training_fraction}.")
                if validation_fraction <= 0.0 or validation_fraction > 1.0:
                    raise ValueError(f"validation_fraction must be in (0, 1], got {validation_fraction}.")

                if _is_main_process():
                    print_run_banner(model_cfg["name"], task_name, species)

                # If species token budgets are available, compute dataset sizes from budgets
                if species_token_budgets:
                    budget = species_token_budgets.get(_species_key(species))
                    if budget is not None:
                        # Budget YAML uses 'train' and 'val' keys; these are token counts PER EPOCH.
                        train_tokens = budget.get("train")
                        val_tokens = budget.get("val")

                        # Compute samples from tokens using window_size tokens per sample
                        computed_train_samples = None
                        computed_eval_samples = None
                        if train_tokens is not None:
                            computed_train_samples = max(1, int(float(train_tokens) // window_size))
                            if "train_num_samples" not in task_cfg:
                                train_num_samples = computed_train_samples

                        if val_tokens is not None:
                            computed_eval_samples = max(1, int(float(val_tokens) // window_size))
                            if "eval_num_samples" not in task_cfg:
                                eval_num_samples = computed_eval_samples

                        def ceil_div(a, b):
                            return (a + b - 1) // b

                        exp_train_batches = ceil_div(train_num_samples, batch_size)
                        exp_eval_batches = ceil_div(eval_num_samples, batch_size)
                        max_train_batches = int(math.ceil(exp_train_batches * float(training_fraction))) if training_fraction < 1.0 else exp_train_batches
                        max_eval_batches = int(math.ceil(exp_eval_batches * float(validation_fraction))) if validation_fraction < 1.0 else exp_eval_batches
                        
                        # KEEP the print limited to Rank 0 to avoid log spam
                        if _is_main_process():
                            print_budget_summary(
                                species=species,
                                train_tokens=train_tokens,
                                val_tokens=val_tokens,
                                window_size=window_size,
                                batch_size=batch_size,
                                train_num_samples=train_num_samples,
                                eval_num_samples=eval_num_samples,
                                exp_train_batches=exp_train_batches,
                                exp_eval_batches=exp_eval_batches,
                                max_train_batches=max_train_batches,
                                max_eval_batches=max_eval_batches,
                                training_fraction=training_fraction,
                                validation_fraction=validation_fraction,
                            )

                # Calculate config hash excluding force_recompute so it doesn't affect caching
                task_cfg_for_hash = {k: v for k, v in task_cfg.items() if k != "force_recompute"}
                task_cfg_for_hash["window_size"] = window_size
                task_cfg_for_hash["training_fraction"] = training_fraction
                task_cfg_for_hash["validation_fraction"] = validation_fraction
                task_cfg_for_hash["overfit_debug"] = overfit_debug
                combined_cfg = {"task": task_cfg_for_hash, "model": model_cfg}
                config_hash = _get_config_hash(combined_cfg)

                run_dir = results_root / model_name / task_name / f"config_{config_hash}"
                final_metrics_path = run_dir / "final_metrics.csv"
                
                # Check if this specific config combination has already been computed (skip if force_recompute=true)
                if not force_recompute and not overfit_debug and _has_combination_been_computed(cache_file, model_name, task_name, config_hash):
                    if _is_main_process():
                        print_status_block("[Skipped - Already Computed]", model_cfg["name"], task_name, config_hash)
                        if final_metrics_path.exists():
                            metrics_df = pd.read_csv(final_metrics_path)
                            if not metrics_df.empty:
                                score = _pick_score(
                                    {str(key): value for key, value in metrics_df.iloc[-1].to_dict().items()},
                                    task_type,
                                )
                                _LOGGER.info(f"  Previous Score: {score:.6f}")
                            # Also generate plots for this existing run (if possible)
                            try:
                                # Prepare minimal genomics inputs to obtain track names and metadata
                                genomics_inputs = prepare_genomics_inputs(
                                    species=species,
                                    bigwig_file_ids=bigwig_ids,
                                    bed_file_ids=bed_ids,
                                    data_cache_dir=global_cache_dir,
                                )
                                num_tracks = genomics_inputs.num_tracks_for_task(task_type)
                                if task_type == "regression":
                                    track_names = genomics_inputs.bigwig_ids
                                else:
                                    track_names = genomics_inputs.bed_ids
                                if not track_names:
                                    track_names = [f"track_{i}" for i in range(num_tracks)]
                                _generate_run_plots(run_dir, task_type, track_names, genomics_inputs.metadata_df)
                            except Exception:
                                pass
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                    continue
                
                if force_recompute and _is_main_process():
                    print_status_block("[Force Recompute]", model_cfg["name"], task_name, config_hash)
                    # Keep the existing run directory in place and overwrite files in place.
                    # This avoids losing a completed result if the run is cancelled mid-way.
                    pass

                elif _is_main_process():
                    print_status_block("[Running]", model_cfg["name"], task_name, config_hash)
                
                run_dir.mkdir(parents=True, exist_ok=True)

                genomics_inputs = prepare_genomics_inputs(
                    species=species,
                    bigwig_file_ids=bigwig_ids,
                    bed_file_ids=bed_ids,
                    data_cache_dir=global_cache_dir,
                )

                # Compute task-appropriate number of tracks from genomics inputs
                num_tracks = genomics_inputs.num_tracks_for_task(task_type)

                model = _build_model(model_cfg, task_type, num_tracks, keep_target_center_fraction, device=device)
                model = model.to(device)
                if dist.is_initialized():
                    # For CUDA devices, pass the local device index; for CPU leave device_ids unset
                    if device.type == "cuda":
                        dev_idx = device.index if device.index is not None else 0
                        model = DDP(model, device_ids=[dev_idx], output_device=dev_idx, find_unused_parameters=True)
                    else:
                        model = DDP(model, find_unused_parameters=True)

                tokenizer = getattr(model, "tokenizer", None)
                if tokenizer is None and hasattr(model, "module"):
                    tokenizer = getattr(model.module, "tokenizer", None)
                if tokenizer is None:
                    raise RuntimeError(
                        "GenomeModelWithHead must expose a tokenizer attribute for dataset creation."
                    )

                if task_type == "regression":
                    transform_fn = create_targets_scaling_fn(genomics_inputs.metadata_df)
                else:
                    transform_fn = lambda x: x

                train_dataset, _ = create_dataset_and_loader(
                    fasta_path=genomics_inputs.fasta_path,
                    bigwig_path_list=genomics_inputs.bigwig_paths,
                    chrom_regions=genomics_inputs.splits_df,
                    split="train",
                    tokenizer=tokenizer,
                    transform_fn=transform_fn,
                    num_samples=train_num_samples,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    target_center_col=target_column,
                    shuffle=False,
                    task_type=task_type,
                    bed_path_list=genomics_inputs.bed_paths,
                    bigwig_ids=genomics_inputs.bigwig_ids,
                    bed_ids=genomics_inputs.bed_ids,
                    metadata_df=genomics_inputs.metadata_df,
                    keep_target_center_fraction=keep_target_center_fraction,
                    window_size=window_size,
                    seed=seed,
                )
                eval_split = _find_eval_split(genomics_inputs.splits_df)
                eval_dataset, _ = create_dataset_and_loader(
                    fasta_path=genomics_inputs.fasta_path,
                    bigwig_path_list=genomics_inputs.bigwig_paths,
                    chrom_regions=genomics_inputs.splits_df,
                    split=eval_split,
                    tokenizer=tokenizer,
                    transform_fn=transform_fn,
                    num_samples=eval_num_samples,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    target_center_col=target_column,
                    shuffle=False,
                    task_type=task_type,
                    bed_path_list=genomics_inputs.bed_paths,
                    bigwig_ids=genomics_inputs.bigwig_ids,
                    bed_ids=genomics_inputs.bed_ids,
                    metadata_df=genomics_inputs.metadata_df,
                    keep_target_center_fraction=keep_target_center_fraction,
                    window_size=window_size,
                    seed=seed,
                )

                train_loader = _build_loader_with_distributed_sampler(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    rank=rank,
                    world_size=world_size,
                    shuffle=False,
                    seed=seed,
                )
                eval_loader = _build_loader_with_distributed_sampler(
                    dataset=eval_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    rank=rank,
                    world_size=world_size,
                    shuffle=False,
                    seed=seed,
                )

                if overfit_debug:
                    debug_batch = _snapshot_first_batch(train_loader)
                    train_loader = [debug_batch]
                    eval_loader = [debug_batch]
                    if _is_main_process():
                        _LOGGER.info("[Overfit Debug] Using one cached batch for both train and eval.")

                if _is_main_process() and (training_fraction < 1.0 or validation_fraction < 1.0):
                    train_total_batches = len(train_loader)
                    eval_total_batches = len(eval_loader)
                    train_max_batches = max(1, int(math.ceil(train_total_batches * training_fraction))) if training_fraction < 1.0 else train_total_batches
                    eval_max_batches = max(1, int(math.ceil(eval_total_batches * validation_fraction))) if validation_fraction < 1.0 else eval_total_batches
                    _LOGGER.info("[Epoch Fraction Summary]")
                    _LOGGER.info(f"  Train: total={train_total_batches:,} fraction={training_fraction:.3f} max={train_max_batches:,}")
                    _LOGGER.info(f"  Eval:  total={eval_total_batches:,} fraction={validation_fraction:.3f} max={eval_max_batches:,}")

                if task_type == "regression":
                    track_names = genomics_inputs.bigwig_ids
                else:
                    track_names = genomics_inputs.bed_ids
                if not track_names:
                    track_names = [f"track_{i}" for i in range(num_tracks)]
                
                assert len(track_names) == num_tracks, f"Number of track names ({len(track_names)}) must match number of tracks ({num_tracks})"

                loss_fn, metrics_tracker = get_loss_and_metrics(
                    task_type=task_type,
                    num_tracks=num_tracks,
                    num_classes=num_classes,
                    track_names=track_names,
                    device=str(device),
                )
                loss_fn = loss_fn.to(device)

                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                # Baseline memory measurement after model + optimizer are ready
                try:
                    _log_gpu_memory("post_model_optimizer", device)
                except Exception:
                    pass

                epoch_history: list[dict[str, float]] = []
                eval_metrics: dict[str, float] = {}
                for epoch in range(epochs):
                    train_one_epoch(
                        model=model,
                        dataloader=train_loader,
                        loss_fn=loss_fn,
                        metrics_tracker=metrics_tracker,
                        optimizer=optimizer,
                        device=device,
                        task_type=task_type,
                        epoch=epoch,
                        epoch_fraction=training_fraction,
                    )
                    eval_metrics = evaluate(
                        model=model,
                        dataloader=eval_loader,
                        loss_fn=loss_fn,
                        metrics_tracker=metrics_tracker,
                        optimizer=optimizer,
                        device=device,
                        task_type=task_type,
                        epoch_fraction=validation_fraction,
                    )

                    epoch_history.append({"epoch": float(epoch), **eval_metrics})

                    if hasattr(metrics_tracker, "update_mean_metrics"):
                        metrics_tracker.update_mean_metrics(
                            epoch,
                            save_csv=True,
                            output_dir=run_dir,
                            filename="metrics_history.csv",
                        ) # pyright: ignore[reportCallIssue]
                    else:
                        pd.DataFrame(epoch_history).to_csv(run_dir / "metrics_history.csv", index=False)

                if _is_main_process():
                    _save_metrics_csv(eval_metrics, run_dir / "final_metrics.csv")
                    score = _pick_score(eval_metrics, task_type)
                    # Generate run-specific plots (training curve + per-track comparison)
                    try:
                        _generate_run_plots(run_dir, task_type, track_names, genomics_inputs.metadata_df)
                    except Exception:
                        pass
                    print_final_eval(model_cfg["name"], task_name, config_hash, score)
                    # Mark this specific config combination as computed in the cache
                    if not overfit_debug:
                        _mark_combination_computed(cache_file, model_name, task_name, config_hash, run_dir)

                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if _is_main_process():
            _generate_comparative_plots(results_root, tasks_cfg, models_cfg)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ntv3 benchmark.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the benchmark YAML config file.",
    )
    parser.add_argument(
        "--cleanup-incomplete",
        action="store_true",
        help="Dry-run: list incomplete config_* result folders and stale cache entries.",
    )
    parser.add_argument(
        "--cleanup-apply",
        action="store_true",
        help="Apply cleanup: delete incomplete result folders and remove stale cache entries.",
    )
    parser.add_argument(
        "--overfit-debug",
        action="store_true",
        help="Run a single cached batch repeatedly to sanity-check whether the model can overfit.",
    )
    args = parser.parse_args()
    main(
        config_path=args.config,
        cleanup_incomplete=args.cleanup_incomplete,
        cleanup_apply=args.cleanup_apply,
        overfit_debug=args.overfit_debug,
    )
