from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score, AUROC, AveragePrecision, MatthewsCorrCoef
from torchmetrics.regression import MeanSquaredError, PearsonCorrCoef, MeanAbsoluteError, R2Score
from src.utils.ddp_runtime_utils import _is_main_process
from src.utils.logging_utils import get_benchmark_logger


_LOGGER = get_benchmark_logger()


class TracksMetrics:
    """Metrics to handle multi-track Pearson correlation and losses."""

    def __init__(
        self,
        track_names: Optional[List[str]] = None,
        split: str = "train",
        device: str = "cuda",
    ):
        self.track_names = track_names or []
        self.num_tracks = len(self.track_names)
        self.split = split
        self.device = device

        self.pearson: Optional[PearsonCorrCoef] = None
        self.mse: Optional[MeanSquaredError] = None
        self.mae: Optional[MeanAbsoluteError] = None
        self.r2: Optional[R2Score] = None

        self.losses: List[float] = []

        # Record mean metrics per logging interval
        self.step_idxs: List[int] = []
        self.mean_pearsons: List[float] = []
        self.mean_mses: List[float] = []
        self.mean_maes: List[float] = []
        self.mean_r2s: List[float] = []
        self.mean_losses: List[float] = []

        if self.num_tracks > 0:
            self._init_metrics(self.num_tracks)

    def _init_metrics(self, num_tracks: int) -> None:
        self.num_tracks = int(num_tracks)
        if not self.track_names:
            self.track_names = [f"track_{i}" for i in range(self.num_tracks)]

        # Move back to GPU (self.device) for multi-GPU NCCL syncing
        self.pearson = PearsonCorrCoef(num_outputs=self.num_tracks).to(self.device)
        self.pearson.set_dtype(torch.float64)

        self.mse = MeanSquaredError(num_outputs=self.num_tracks).to(self.device)
        self.mse.set_dtype(torch.float64)

        self.mae = MeanAbsoluteError(num_outputs=self.num_tracks).to(self.device)
        self.mae.set_dtype(torch.float64)

        self.r2 = R2Score(multioutput="raw_values").to(self.device)
        self.r2.set_dtype(torch.float64)

    def reset(self):
        if self.pearson is not None:
            self.pearson.reset()
        if self.mse is not None:
            self.mse.reset()
        if self.mae is not None:
            self.mae.reset()
        if self.r2 is not None:
            self.r2.reset()
        self.losses = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: float):
        if self.pearson is None or self.mse is None or self.mae is None or self.r2 is None:
            self._init_metrics(predictions.shape[-1])

        pred_flat = predictions.detach().reshape(-1, self.num_tracks).to(torch.float64)
        target_flat = targets.detach().reshape(-1, self.num_tracks).to(torch.float64)

        # Update metrics
        self.pearson.update(pred_flat, target_flat)
        self.mse.update(pred_flat, target_flat)
        self.mae.update(pred_flat, target_flat)
        self.r2.update(pred_flat, target_flat)
        self.losses.append(loss)

    def compute(self) -> Dict[str, float]:
        """Compute metrics and return a dictionary of values."""
        if self.pearson is None or self.mse is None or self.mae is None or self.r2 is None:
            raise RuntimeError("Metrics are not initialized. Call update() first.")

        correlations = np.atleast_1d(self.pearson.compute().cpu().numpy())
        mses = np.atleast_1d(self.mse.compute().cpu().numpy())
        maes = np.atleast_1d(self.mae.compute().cpu().numpy())
        r2s = np.atleast_1d(self.r2.compute().cpu().numpy())

        metrics_dict: Dict[str, float] = {
            f"{track_name}/pearson": float(correlations[i])
            for i, track_name in enumerate(self.track_names)
        }
        metrics_dict.update(
            {
                f"{track_name}/mse": float(mses[i])
                for i, track_name in enumerate(self.track_names)
            }
        )
        metrics_dict.update(
            {
                f"{track_name}/mae": float(maes[i])
                for i, track_name in enumerate(self.track_names)
            }
        )
        metrics_dict.update(
            {
                f"{track_name}/r2": float(r2s[i])
                for i, track_name in enumerate(self.track_names)
            }
        )

        metrics_dict["mean/pearson"] = float(np.mean(correlations))
        metrics_dict["mean/mse"] = float(np.mean(mses))
        metrics_dict["mean/mae"] = float(np.mean(maes))
        metrics_dict["mean/r2"] = float(np.mean(r2s))
        metrics_dict["loss"] = float(np.mean(self.losses)) if self.losses else 0.0
        return metrics_dict

    def update_mean_metrics(
        self,
        step_idx: int,
        save_csv: bool = True,
        output_dir: str | Path | None = None,
        filename: str | None = None,
    ):
        """Update mean metrics over the logging interval and optionally save CSV."""
        metrics_dict = self.compute()
        self.step_idxs.append(step_idx)
        self.mean_pearsons.append(metrics_dict["mean/pearson"])
        self.mean_mses.append(metrics_dict["mean/mse"])
        self.mean_maes.append(metrics_dict["mean/mae"])
        self.mean_r2s.append(metrics_dict["mean/r2"])
        self.mean_losses.append(metrics_dict["loss"])

        if save_csv:
            output_path = Path(output_dir) if output_dir is not None else Path(".")
            output_path.mkdir(parents=True, exist_ok=True)
            data = {
                "step": self.step_idxs,
                "mean_loss": self.mean_losses,
                "mean_pearson": self.mean_pearsons,
                "mean_mse": self.mean_mses,
                "mean_mae": self.mean_maes,
                "mean_r2": self.mean_r2s,
            }
            out_name = filename or f"metrics_{self.split}.csv"
            pd.DataFrame(data).to_csv(output_path / out_name, index=False)

    def print_metrics(
        self,
        step_idx: Optional[int] = None,
        total_steps: Optional[int] = None,
        print_per_track: bool = False,
    ):
        """Print a summary of metrics."""
        if not self.step_idxs and step_idx is None:
            raise RuntimeError("No metric history available. Call update_mean_metrics() first.")

        current_step = step_idx if step_idx is not None else self.step_idxs[-1]
        if total_steps is None:
            header = f"Step {current_step}"
        else:
            header = f"Step {current_step}/{total_steps}"

        sep = "-" * 41
        _LOGGER.info(f"[Metrics - {self.split}] {header}")
        _LOGGER.info(sep)
        _LOGGER.info(f"  Loss:           {self.mean_losses[-1]:.4f}")
        _LOGGER.info(f"  Mean Pearson:   {self.mean_pearsons[-1]:.4f}")
        _LOGGER.info(f"  Mean MSE:       {self.mean_mses[-1]:.4f}")
        _LOGGER.info(f"  Mean MAE:       {self.mean_maes[-1]:.4f}")
        _LOGGER.info(f"  Mean R2:        {self.mean_r2s[-1]:.4f}")
        _LOGGER.info(sep)

        if print_per_track:
            metrics_dict = self.compute()
            for metric_key, metric_value in metrics_dict.items():
                _LOGGER.info(f"  {metric_key}: {metric_value:.4f}")


class TracksClassificationMetrics:
    """Multi-track classification metrics wrapper with per-track and mean metrics."""
    
    def __init__(
        self,
        track_names: Optional[List[str]] = None,
        num_classes: int = 2,
        split: str = "train",
        device: str = "cuda",
    ):
        self.track_names = track_names or []
        self.num_tracks = len(self.track_names)
        self.num_classes = num_classes
        self.split = split
        self.device = device
        self.task = "binary" if num_classes == 2 else "multiclass"
        
        # Per-track metrics collections
        self.metrics_per_track: List[MetricCollection] = []
        self.losses: List[float] = []
        
        # History tracking
        self.step_idxs: List[int] = []
        self.mean_losses: List[float] = []
        
        if self.num_tracks > 0:
            self._init_metrics()
    
    def _init_metrics(self) -> None:
        """Initialize per-track metric collections."""
        if not self.track_names:
            self.track_names = [f"track_{i}" for i in range(self.num_tracks)]
        
        for _ in range(self.num_tracks):
            avg_method = "macro" if self.num_classes > 2 else "binary"
            metrics = MetricCollection({
                "accuracy": Accuracy(task=self.task, num_classes=self.num_classes),
                "f1": F1Score(task=self.task, num_classes=self.num_classes, average=avg_method),
                "auroc": AUROC(task=self.task, num_classes=self.num_classes, thresholds=200),
                "auprc": AveragePrecision(task=self.task, num_classes=self.num_classes, thresholds=200),
                "mcc": MatthewsCorrCoef(task=self.task, num_classes=self.num_classes),
            }).to(self.device) 
            self.metrics_per_track.append(metrics)
    
    def reset(self):
        """Reset all metrics and losses."""
        for metrics in self.metrics_per_track:
            metrics.reset()
        self.losses = []
    
    def update(
        self,
        preds: torch.Tensor,  # (B*S, T, C)
        targets: torch.Tensor,  # (B*S, T)
        loss: float = 0.0,
    ):
        """
        Update metrics from model predictions and targets.
        
        Args:
            preds: Model logits of shape (B*S, T, C)
            targets: Class indices of shape (B*S, T)
            loss: Scalar loss value
        """
        assert preds.shape[0] == targets.shape[0], f"Batch size mismatch: {preds.shape[0]} vs {targets.shape[0]}"
        assert preds.shape[1] == self.num_tracks, f"Num tracks mismatch: {preds.shape[1]} vs {self.num_tracks}"
        
        for t in range(self.num_tracks):
            pred_t = preds[:, t, :]  # (B*S, C)
            target_t = targets[:, t]  # (B*S,)

            # For binary metrics (task='binary'), torchmetrics expects
            # predictions to be either probabilities/logits of shape (N,) or
            # class labels of shape (N,). If we have logits with shape
            # (N, 2), convert to positive-class probability vector.
            if self.task == "binary" and pred_t.dim() == 2 and pred_t.size(1) == 2:
                probs_pos = torch.softmax(pred_t, dim=-1)[:, 1]
                self.metrics_per_track[t].update(probs_pos, target_t)
            else:
                self.metrics_per_track[t].update(pred_t, target_t)
        
        self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """Compute current metrics."""
        metrics_dict = {}
        
        for i, (track_name, metrics) in enumerate(zip(self.track_names, self.metrics_per_track)):
            track_metrics = metrics.compute()
            for key, value in track_metrics.items():
                metrics_dict[f"{track_name}/{key}"] = float(value)
        
        # Add mean metrics
        if metrics_dict:
            accuracy_vals = [v for k, v in metrics_dict.items() if "/accuracy" in k]
            f1_vals = [v for k, v in metrics_dict.items() if "/f1" in k]
            auroc_vals = [v for k, v in metrics_dict.items() if "/auroc" in k]
            auprc_vals = [v for k, v in metrics_dict.items() if "/auprc" in k]
            mcc_vals = [v for k, v in metrics_dict.items() if "/mcc" in k]
            
            if accuracy_vals:
                metrics_dict["mean/accuracy"] = float(np.mean(accuracy_vals))
            if f1_vals:
                metrics_dict["mean/f1"] = float(np.mean(f1_vals))
            if auroc_vals:
                metrics_dict["mean/auroc"] = float(np.mean(auroc_vals))
            if auprc_vals:
                metrics_dict["mean/auprc"] = float(np.mean(auprc_vals))
            if mcc_vals:
                metrics_dict["mean/mcc"] = float(np.mean(mcc_vals))
        
        metrics_dict["loss"] = float(np.mean(self.losses)) if self.losses else 0.0
        return metrics_dict
    
    def update_mean_metrics(
        self,
        step_idx: int,
        save_csv: bool = True,
        output_dir: Optional[str | Path] = None,
        filename: Optional[str] = None,
    ):
        """Update and optionally save metrics to CSV."""
        metrics_dict = self.compute()
        self.step_idxs.append(step_idx)
        self.mean_losses.append(metrics_dict["loss"])
        
        # Store per-track and mean metrics for CSV accumulation
        if not hasattr(self, "_metric_history"):
            self._metric_history = {}
        
        # Store all metrics from this step
        for key, value in metrics_dict.items():
            if key not in self._metric_history:
                self._metric_history[key] = []
            self._metric_history[key].append(value)
        
        if save_csv and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / (filename or f"metrics_{self.split}.csv")
            
            # Build DataFrame with accumulated metric history
            data = {"step": self.step_idxs}
            for key in sorted(self._metric_history.keys()):
                if key != "loss":
                    data[key] = self._metric_history[key]
            data["loss"] = self.mean_losses
            
            pd.DataFrame(data).to_csv(csv_path, index=False)
    
    def print_metrics(self, step_idx: Optional[int] = None, print_per_track: bool = False):
        """Print metrics summary."""
        metrics_dict = self.compute()
        
        step_str = f"Step {step_idx}" if step_idx is not None else "Metrics"
        sep = "-" * 41
        _LOGGER.info(f"[Metrics - {self.split}] {step_str}")
        _LOGGER.info(sep)
        _LOGGER.info(f"  Loss:        {metrics_dict['loss']:.4f}")
        _LOGGER.info(f"  Mean Acc:    {metrics_dict.get('mean/accuracy', 0):.4f}")
        _LOGGER.info(f"  Mean F1:     {metrics_dict.get('mean/f1', 0):.4f}")
        _LOGGER.info(f"  Mean AUROC:  {metrics_dict.get('mean/auroc', 0):.4f}")
        _LOGGER.info(f"  Mean AUPRC:  {metrics_dict.get('mean/auprc', 0):.4f}")
        _LOGGER.info(f"  Mean MCC:    {metrics_dict.get('mean/mcc', 0):.4f}")
        _LOGGER.info(sep)

        if print_per_track:
            for key, value in sorted(metrics_dict.items()):
                if key not in ["loss"] and not key.startswith("mean/"):
                    _LOGGER.info(f"  {key}: {value:.4f}")


LossAndMetricType = Tuple[nn.Module, Union[TracksMetrics, TracksClassificationMetrics, MetricCollection]]


class PoissonMultinomialLoss(nn.Module):
    """
    Regression loss for continuous genomic tracks.
    Combines a Poisson loss on the total counts (scale) with a 
    Multinomial loss on the spatial distribution of the signal (shape).
    """
    def __init__(self, shape_loss_coefficient: float = 5.0, epsilon: float = 1e-7):
        super().__init__()
        self.shape_loss_coefficient = shape_loss_coefficient
        self.epsilon = epsilon

    def _poisson_loss(self, ytrue: torch.Tensor, ypred: torch.Tensor) -> torch.Tensor:
        """Poisson loss per element: ypred - ytrue * log(ypred)."""
        return ypred - ytrue * torch.log(ypred + self.epsilon)

    def _safe_for_grad_log(self, x: torch.Tensor) -> torch.Tensor:
        """Guarantees that the log is defined for all x > 0 in a differentiable way."""
        return torch.log(torch.where(x > 0.0, x, torch.ones_like(x)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the combined loss.
        
        Args:
            logits: Predictions of shape (batch, seq_length, num_tracks)
            targets: Ground truth of shape (batch, seq_length, num_tracks)
            
        Returns:
            A scalar tensor containing the combined loss.
        """
        # Ensure tensors are float
        logits = logits.to(torch.float32)
        targets = targets.to(torch.float32)

        batch_size, seq_length, num_tracks = logits.shape

        # ---------------------------------------------------------
        # 1. Scale Loss: Poisson loss on total counts per sequence
        # ---------------------------------------------------------
        # Sum over sequence dimension (axis=1) to get total volume of signal
        sum_pred = logits.sum(dim=1)  # (batch, num_tracks)
        sum_true = targets.sum(dim=1)  # (batch, num_tracks)

        # Compute poisson loss per (batch, track)
        scale_loss = self._poisson_loss(sum_true, sum_pred)

        # Normalize by sequence length and average over batch and tracks
        scale_loss = scale_loss / (seq_length + self.epsilon)
        scale_loss = scale_loss.mean()

        # ---------------------------------------------------------
        # 2. Shape Loss: Multinomial loss on sequence distribution
        # ---------------------------------------------------------
        predicted_counts = logits + self.epsilon
        targets_with_epsilon = targets + self.epsilon

        # Normalize predictions to get probabilities along the sequence
        denom = predicted_counts.sum(dim=1, keepdim=True) + self.epsilon
        p_pred = predicted_counts / denom

        # Compute shape loss: -sum(targets * log(p_pred))
        pl_pred = self._safe_for_grad_log(p_pred)
        shape_loss = -(targets_with_epsilon * pl_pred)

        # Sum over all dimensions and normalize by total number of positions
        shape_denom = batch_size * seq_length * num_tracks + self.epsilon
        shape_loss = shape_loss.sum() / shape_denom

        # ---------------------------------------------------------
        # 3. Combine Losses
        # ---------------------------------------------------------
        loss = shape_loss + (scale_loss / self.shape_loss_coefficient)

        return loss


class FocalLoss(nn.Module):
    """
    Computes weighted focal loss for single or multi-track classification from logits.
    Handles logits of shape (B,S,C) or (B*S,T,C) and targets (B,S) or (B*S,T).
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, epsilon: float = 1e-7):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Expected shape: (C,) for C classes
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        is_multitrack = logits.dim() == 3
        num_classes = logits.shape[-1]
        
        if is_multitrack:
            logits_flat = logits.reshape(-1, num_classes)
            targets_flat = targets.reshape(-1).long()
        else:
            logits_flat = logits.reshape(-1, num_classes)
            targets_flat = targets.reshape(-1).long()
        
        # 1. Compute log probabilities and probabilities
        log_probs = F.log_softmax(logits_flat, dim=-1)
        probs = torch.exp(log_probs)
        
        # 2. Extract pt and log_pt (the probability and log_prob of the TRUE target class)
        pt = torch.gather(probs, dim=-1, index=targets_flat.unsqueeze(-1)).squeeze(-1)
        log_pt = torch.gather(log_probs, dim=-1, index=targets_flat.unsqueeze(-1)).squeeze(-1)
        
        # 3. Core focal loss formula: -(1 - pt)^gamma * log(pt)
        loss = -((1 - pt) ** self.gamma) * log_pt
        
        # 4. Apply alpha weights if provided
        if self.alpha is not None:
            # Ensure alpha is on the same device as the targets to prevent crashes
            if self.alpha.device != targets_flat.device:
                self.alpha = self.alpha.to(targets_flat.device)
                
            # Gather the specific alpha weight for the true class of each position
            at = self.alpha.gather(0, targets_flat)
            loss = loss * at

        # Average loss over all positions
        return loss.sum() / (loss.numel() + self.epsilon)
    

def get_loss_and_metrics(
    task_type: str,
    num_tracks: int = 1,
    num_classes: int = 2,
    track_names: Optional[List[str]] = None,
    device: str = "cuda",
) -> LossAndMetricType:
    """
    Factory for task-specific loss and metric objects supporting multi-track.
    
    Args:
        task_type: "regression" or "classification"
        num_tracks: Number of output tracks (default: 1)
        num_classes: Number of classes for classification (default: 2)
        track_names: Optional list of track names. If None, auto-generated.
        device: Device for metrics ("cuda", "cpu", etc.)
    
    Returns:
        (loss_fn, metrics_tracker) tuple with consistent API
    """
    task_type = task_type.strip().lower()
    
    if not track_names:
        track_names = [f"track_{i}" for i in range(num_tracks)]

    if task_type == "regression":
        loss_fn = PoissonMultinomialLoss()
        metrics_tracker = TracksMetrics(
            track_names=track_names,
            split="train",
            device=device,
        )
        return loss_fn, metrics_tracker

    elif task_type == "classification":
        loss_fn = FocalLoss()
        metrics_tracker = TracksClassificationMetrics(
            track_names=track_names,
            num_classes=num_classes,
            split="train",
            device=device,
        )
        return loss_fn, metrics_tracker

    else:
        raise ValueError(
            f"Unsupported task_type '{task_type}'. Expected 'regression' or 'classification'."
        )
