from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist

from src.utils.logging_utils import get_benchmark_logger


_LOGGER = get_benchmark_logger()


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_main_process() -> bool:
    if not _is_distributed():
        return True
    return dist.get_rank() == 0


def _maybe_set_epoch(dataloader: Any, epoch: int) -> None:
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _extract_tokens(batch: Dict[str, Any]) -> torch.Tensor:
    for key in ("tokens", "input_ids"):
        if key in batch:
            return batch[key]
    raise KeyError("Batch does not contain token tensor under 'tokens' or 'input_ids'.")


def _extract_targets(batch: Dict[str, Any]) -> torch.Tensor:
    possible_keys = ['bed_targets', 'bigwig_targets', 'targets', 'target', 'labels', 'y']
    for key in possible_keys:
        if key in batch:
            return batch[key]      
         
    available_keys = list(batch.keys())
    raise KeyError(
        f"Batch does not contain expected target keys. "
        f"Available keys in this dataset are: {available_keys}"
    )


def _extract_model_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("logits", "bigwig_tracks_logits", "predictions", "output"):
            if key in output:
                return output[key]
        raise KeyError(
            "Model output dict does not contain one of: "
            "'logits', 'bigwig_tracks_logits', 'predictions', 'output'."
        )
    if hasattr(output, "logits"):
        return output.logits
    raise TypeError(f"Unsupported model output type: {type(output)}")


def _forward_model(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        
        # 1. Check and cache the signature safely to avoid DDP corruption
        if not hasattr(model, "_token_kwarg"):
            # Unwrap DDP to inspect the actual inner model
            unwrapped_model = model.module if hasattr(model, "module") else model
            
            import inspect
            sig = inspect.signature(unwrapped_model.forward)
            
            if "input_ids" in sig.parameters:
                model._token_kwarg = "input_ids"
            elif "tokens" in sig.parameters:
                model._token_kwarg = "tokens"
            else:
                model._token_kwarg = "args"
                
        # 2. Execute the forward pass EXACTLY once
        kwarg = model._token_kwarg
        if kwarg == "input_ids":
            output = model(input_ids=tokens)
        elif kwarg == "tokens":
            output = model(tokens=tokens)
        else:
            output = model(tokens)
            
        return _extract_model_output(output)


def _center_crop_sequence(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    if tensor.dim() < 2:
        return tensor

    current_length = tensor.shape[1]
    if current_length == target_length:
        return tensor
    if target_length <= 0 or target_length > current_length:
        raise ValueError(
            f"Cannot crop sequence of length {current_length} to {target_length}."
        )

    start = max(0, (current_length - target_length) // 2)
    end = start + target_length
    return tensor[:, start:end, ...]


def _flatten_for_task(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    task_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten and reshape outputs/targets for multi-track support.
    
    Regression: outputs (B,S,T)->flattened, targets (B,S,T)->flattened
    Classification: outputs (B,S,T,C) or (B,S,C), targets (B,S,T) or (B,S) 
                   -> reshape to (B*S,T,C) and (B*S,T) for metrics
    """
    task = task_type.strip().lower()

    if task == "regression":
        # Handle (B,S,T) or (B,S,1) or (B,S) shapes
        if outputs.dim() == 2:
            outputs = outputs.unsqueeze(-1)
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)

        if outputs.dim() >= 3 and targets.dim() >= 3 and outputs.shape[1] != targets.shape[1]:
            if outputs.shape[1] > targets.shape[1]:
                outputs = _center_crop_sequence(outputs, targets.shape[1])
            else:
                targets = _center_crop_sequence(targets, outputs.shape[1])
        
        # Flatten: (B, S, T) -> (B*S*T, 1)
        outputs = outputs.reshape(-1, outputs.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])
        return outputs, targets

    if task == "classification":
        # outputs: (B, S, T, C) or (B, S, C)
        # targets: (B, S, T) or (B, S)
        
        if outputs.dim() == 3:
            # Single-track case: (B, S, C) -> add T=1
            outputs = outputs.unsqueeze(2)  # (B, S, 1, C)
        elif outputs.dim() != 4:
            raise ValueError(f"Classification outputs must be 3D or 4D, got {outputs.dim()}D")
        
        if targets.dim() == 2:
            targets = targets.unsqueeze(2)  # (B, S, 1)
        elif targets.dim() != 3:
            raise ValueError(f"Classification targets must be 2D or 3D, got {targets.dim()}D")
        
        batch_size, seq_len, num_tracks, num_classes = outputs.shape
        seq_len_targets = targets.shape[1]
        
        # Align sequence lengths if needed
        if seq_len != seq_len_targets:
            diff = seq_len - seq_len_targets
            start = diff // 2
            end = start + seq_len_targets
            outputs = outputs[:, start:end, :, :]
        
        # Reshape to (B*S, T, C) and (B*S, T) for metrics
        # If targets have singleton track dimension (e.g., shape (B, S, 1)),
        # broadcast to match model's number of tracks.
        if targets.shape[2] == 1 and num_tracks > 1:
            targets = targets.expand(-1, -1, num_tracks)

        if targets.shape[2] != num_tracks:
            raise RuntimeError(
                f"Num tracks mismatch between model outputs ({num_tracks}) "
                f"and targets ({targets.shape[2]})."
            )

        outputs = outputs.reshape(-1, num_tracks, num_classes)
        targets = targets.reshape(-1, num_tracks).long()
        
        return outputs, targets

    raise ValueError(f"Unsupported task_type '{task_type}'.")


def _align_regression_tensors(
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if outputs.dim() == 2:
        outputs = outputs.unsqueeze(-1)
    if targets.dim() == 2:
        targets = targets.unsqueeze(-1)

    if outputs.dim() >= 3 and targets.dim() >= 3 and outputs.shape[1] != targets.shape[1]:
        if outputs.shape[1] > targets.shape[1]:
            outputs = _center_crop_sequence(outputs, targets.shape[1])
        else:
            targets = _center_crop_sequence(targets, outputs.shape[1])

    return outputs, targets


def _to_float_dict(metrics: Dict[str, Any]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            result[key] = float(value.detach().cpu().item())
        else:
            result[key] = float(value)
    return result


def _max_batches_from_fraction(dataloader: Any, fraction: float) -> int | None:
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}.")
    if fraction >= 1.0:
        return None

    try:
        total_batches = int(len(dataloader))
    except Exception:
        return None

    if total_batches <= 0:
        return None
    return max(1, int(math.ceil(total_batches * fraction)))


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: Any,
    loss_fn: torch.nn.Module,
    metrics_tracker: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    task_type: str,
    epoch: int,
    epoch_fraction: float = 1.0,
) -> Dict[str, float]:
    """Train one epoch with DDP-aware sampler epoch setting and metric tracking."""
    _maybe_set_epoch(dataloader, epoch)

    model.train()
    metrics_tracker.reset()

    # Prepare CUDA peak memory tracking for an accurate first-batch peak measurement
    first_batch_logged = bool(getattr(model, "_first_batch_memory_logged", False))
    if torch.cuda.is_available() and _is_main_process():
        try:
            if not first_batch_logged:
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    running_loss = 0.0
    num_batches = 0
    max_batches = _max_batches_from_fraction(dataloader, epoch_fraction)

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        tokens = _extract_tokens(batch).to(device, non_blocking=True)
        targets = _extract_targets(batch).to(device, non_blocking=True)

        outputs = _forward_model(model, tokens)
        outputs = outputs.float()
        if task_type.strip().lower() == "regression":
            outputs, targets = _align_regression_tensors(outputs, targets)
        outputs_flat, targets_flat = _flatten_for_task(outputs, targets, task_type)

        if task_type.strip().lower() == "regression":
            loss = loss_fn(outputs, targets)
        else:
            loss = loss_fn(outputs_flat, targets_flat)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        outputs_tracked = outputs_flat.detach()
        targets_tracked = targets_flat.detach()

        metrics_tracker.update(
            outputs_tracked,
            targets_tracked,
            loss=loss.item(), 
        )

        running_loss += loss.item()
        num_batches += 1

        # Log peak memory after the first batch to capture the initial allocation pattern
        if (
            torch.cuda.is_available()
            and not first_batch_logged
            and _is_main_process()
        ):
            try:
                torch.cuda.synchronize()
                peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
                alloc = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                _LOGGER.info("[Mem][FirstBatch]")
                _LOGGER.info(f"  Allocated:     {alloc:.0f} MiB")
                _LOGGER.info(f"  Reserved:      {reserved:.0f} MiB")
                _LOGGER.info(f"  Peak Reserved: {peak_reserved:.0f} MiB")
                first_batch_logged = True
                setattr(model, "_first_batch_memory_logged", True)
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    # Aggregate metrics across all ranks for DDP
    if _is_distributed():
        loss_tensor = torch.tensor(running_loss, device="cuda" if torch.cuda.is_available() else "cpu")
        num_batches_tensor = torch.tensor(num_batches, device="cuda" if torch.cuda.is_available() else "cpu")
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
        running_loss = float(loss_tensor.item())
        num_batches = int(num_batches_tensor.item())
    
    metrics = _to_float_dict(metrics_tracker.compute())
    avg_loss = running_loss / max(num_batches, 1)
    metrics["loss"] = avg_loss

    _LOGGER.info(f"[Train][Epoch {epoch}] loss={avg_loss:.6f}")

    return metrics


def evaluate(
    model: torch.nn.Module,
    dataloader: Any,
    loss_fn: torch.nn.Module,
    metrics_tracker: Any,
    optimizer: Any,
    device: torch.device | str,
    task_type: str,
    epoch_fraction: float = 1.0,
) -> Dict[str, float]:
    """Evaluate for one full dataloader pass with DDP-safe metric computation."""
    del optimizer

    model.eval()
    metrics_tracker.reset()

    running_loss = 0.0
    num_batches = 0
    max_batches = _max_batches_from_fraction(dataloader, epoch_fraction)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            tokens = _extract_tokens(batch).to(device, non_blocking=True)
            targets = _extract_targets(batch).to(device, non_blocking=True)

            outputs = _forward_model(model, tokens)
            outputs = outputs.float()
            if task_type.strip().lower() == "regression":
                outputs, targets = _align_regression_tensors(outputs, targets)
            outputs_flat, targets_flat = _flatten_for_task(outputs, targets, task_type)

            if task_type.strip().lower() == "regression":
                loss = loss_fn(outputs, targets)
            else:
                loss = loss_fn(outputs_flat, targets_flat)

            outputs_tracked = outputs_flat.detach()
            targets_tracked = targets_flat.detach()

            metrics_tracker.update(
                outputs_tracked,
                targets_tracked,
                loss=loss.item(), 
            )

            running_loss += loss.item()
            num_batches += 1

    # Aggregate metrics across all ranks for DDP
    if _is_distributed():
        loss_tensor = torch.tensor(running_loss, device="cuda" if torch.cuda.is_available() else "cpu")
        num_batches_tensor = torch.tensor(num_batches, device="cuda" if torch.cuda.is_available() else "cpu")
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
        running_loss = float(loss_tensor.item())
        num_batches = int(num_batches_tensor.item())
    
    metrics = _to_float_dict(metrics_tracker.compute())
    avg_loss = running_loss / max(num_batches, 1)
    metrics["loss"] = avg_loss

    _LOGGER.info(f"[Eval] loss={avg_loss:.6f}")

    return metrics
