from __future__ import annotations

from src.utils.logging_utils import get_benchmark_logger


_LOGGER = get_benchmark_logger()


def print_run_banner(model_name: str, task_name: str, species: str) -> None:
    run_label = f"Model={model_name} | Task={task_name} | Species={species}"
    run_banner = f"# {run_label}"
    _LOGGER.info("")
    _LOGGER.info(run_banner)
    _LOGGER.info("=" * len(run_banner))


def print_budget_summary(
    species: str,
    train_tokens: int | None,
    val_tokens: int | None,
    window_size: int,
    batch_size: int,
    train_num_samples: int,
    eval_num_samples: int,
    exp_train_batches: int,
    exp_eval_batches: int,
    max_train_batches: int,
    max_eval_batches: int,
    training_fraction: float,
    validation_fraction: float,
) -> None:
    train_tokens_display = f"{int(train_tokens):,}" if train_tokens is not None else "NA"
    val_tokens_display = f"{int(val_tokens):,}" if val_tokens is not None else "NA"
    header = f"[Budget Summary - {str(species).title()}]"
    sep = "-" * 41
    pct_train = int(float(training_fraction) * 100)
    pct_val = int(float(validation_fraction) * 100)

    _LOGGER.info(header)
    _LOGGER.info(sep)
    _LOGGER.info("Data Scope:")
    _LOGGER.info(f"  Train Tokens:    {train_tokens_display:>15}")
    _LOGGER.info(f"  Val Tokens:      {val_tokens_display:>15}")
    _LOGGER.info(f"  Window Size:     {window_size:>13,}")
    _LOGGER.info("")
    _LOGGER.info(f"Samples & Batches (Batch Size: {batch_size}):")
    _LOGGER.info(f"  Train Samples:   {train_num_samples:>13,}")
    _LOGGER.info(f"  Eval Samples:    {eval_num_samples:>13,}")
    _LOGGER.info(f"  Max Train Batches: {max_train_batches:,} ({pct_train}% of {exp_train_batches:,} total)")
    _LOGGER.info(f"  Max Eval Batches:  {max_eval_batches:,} ({pct_val}% of {exp_eval_batches:,} total)")
    _LOGGER.info(sep)


def print_status_block(title: str, model_name: str, task_name: str, config_hash: str) -> None:
    _LOGGER.info(title)
    _LOGGER.info(f"  Model:  {model_name}")
    _LOGGER.info(f"  Task:   {task_name}")
    _LOGGER.info(f"  Config: {config_hash}")


def print_final_eval(model_name: str, task_name: str, config_hash: str, score: float) -> None:
    _LOGGER.info("[Final Eval]")
    _LOGGER.info(f"  Model:  {model_name}")
    _LOGGER.info(f"  Task:   {task_name}")
    _LOGGER.info(f"  Config: {config_hash}")
    _LOGGER.info(f"  Score:  {score:.6f}")