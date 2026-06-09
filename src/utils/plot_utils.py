from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.logging_utils import get_benchmark_logger

matplotlib.use("Agg")


_LOGGER = get_benchmark_logger()


def _sanitize_dir_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _primary_metric_key(task_type: str, metrics: dict[str, float]) -> str | None:
    if task_type == "regression":
        for key in ("mean/pearson", "mean/spearman", "mean/mse", "loss"):
            if key in metrics:
                return key
    else:
        for key in ("mean/auprc", "mean/accuracy", "mean/f1", "f1", "accuracy", "loss"):
            if key in metrics:
                return key
    return None


def _comparison_plots_dir(results_root: Path, task_name: str) -> Path:
    plots_dir = results_root / "plots" / task_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def _metric_higher_is_better(metric_name: str) -> bool:
    metric = metric_name.lower()
    return not (metric.endswith("loss") or metric.endswith("mse") or metric.endswith("mae"))


def _display_metric_label(metric_name: str) -> str:
    metric = metric_name.lower()
    if metric in {"mean/auprc", "auprc"}:
        return "AUPRC"
    if metric in {"mean/pearson", "pearson"}:
        return "Pearson"
    if metric in {"mean/spearman", "spearman"}:
        return "Spearman"
    if metric in {"mean/accuracy", "accuracy"}:
        return "Accuracy"
    if metric in {"mean/f1", "f1"}:
        return "F1"
    if metric.endswith("loss"):
        return "Loss"
    if metric.endswith("mse"):
        return "MSE"
    if metric.endswith("mae"):
        return "MAE"
    return metric_name.replace("mean/", "").replace("_", " ").replace("/", " ").title()


def _normalize_metric_value(metric_name: str, value: float) -> float:
    """Map a raw metric to [0, 1] using an absolute optimum reference."""
    metric = metric_name.lower()

    if "pearson" in metric or "spearman" in metric:
        return float(max(0.0, min(1.0, (value + 1.0) / 2.0)))

    if "accuracy" in metric or metric.endswith("/f1") or metric == "f1" or metric.endswith("f1"):
        return float(max(0.0, min(1.0, value)))

    if metric.endswith("loss") or metric.endswith("mse") or metric.endswith("mae"):
        if value <= 0.0:
            return 1.0
        return float(1.0 / (1.0 + value))

    return float(max(0.0, min(1.0, value)))


def _normalize_comparison_frame(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data

    normalized = pd.DataFrame(index=data.index)
    for column in data.columns:
        series = pd.to_numeric(data[column], errors="coerce")
        normalized[column] = series.apply(
            lambda value: _normalize_metric_value(str(column), float(value)) if pd.notna(value) else value
        )

    return normalized


def _save_heatmap(
    data: pd.DataFrame,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str = "score",
) -> None:
    if data.empty:
        return

    fig_width = max(6.0, 1.2 * max(1, data.shape[1]))
    fig_height = max(4.0, 0.45 * max(1, data.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(data.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([str(col) for col in data.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([str(idx) for idx in data.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def generate_comparative_plots(
    results_root: Path,
    tasks_cfg: dict[str, Any],
    models_cfg: list[dict[str, Any]],
) -> None:
    for task_name, task_cfg in tasks_cfg.items():
        task_type = str(task_cfg.get("type", "regression")).lower()
        plots_dir = _comparison_plots_dir(results_root, task_name)

        model_names: list[str] = []
        values: list[float] = []
        metric_name: str | None = None
        summary_rows: list[dict[str, Any]] = []
        track_rows: list[dict[str, Any]] = []
        track_metric_name: str | None = None

        for model_cfg in models_cfg:
            model_name = _sanitize_dir_name(str(model_cfg["name"]))
            model_task_dir = results_root / model_name / task_name

            if not model_task_dir.exists():
                continue

            config_dirs = sorted(model_task_dir.glob("config_*"), reverse=True)
            if not config_dirs:
                continue

            latest_config_dir = config_dirs[0]
            final_metrics_path = latest_config_dir / "final_metrics.csv"

            if not final_metrics_path.exists():
                continue

            metrics_df = pd.read_csv(final_metrics_path)
            if metrics_df.empty:
                continue

            metrics_row = {str(key): value for key, value in metrics_df.iloc[-1].to_dict().items()}
            if metric_name is None:
                metric_name = _primary_metric_key(task_type, metrics_row)
            if metric_name is None or metric_name not in metrics_row:
                continue

            model_names.append(str(model_cfg["name"]))
            values.append(float(metrics_row[metric_name]))
            summary_rows.append({"model": str(model_cfg["name"]), **metrics_row})

            if task_type == "regression":
                preferred_suffix = "/pearson"
                fallback_suffixes = ["/pearson", "/spearman", "/mse", "/mae"]
            else:
                preferred_suffix = "/auprc"
                fallback_suffixes = ["/auprc", "/accuracy", "/f1", "/auroc", "/mcc"]

            chosen_suffix = preferred_suffix
            if not any(key.endswith(chosen_suffix) for key in metrics_row):
                for suffix in fallback_suffixes:
                    if any(key.endswith(suffix) for key in metrics_row):
                        chosen_suffix = suffix
                        break

            if track_metric_name is None:
                if task_type == "regression":
                    track_metric_name = chosen_suffix.lstrip("/")
                else:
                    track_metric_name = f"mean/{chosen_suffix.lstrip('/')}" if not chosen_suffix.startswith("/loss") else "loss"

            if chosen_suffix != "/loss":
                track_row: dict[str, Any] = {"model": str(model_cfg["name"])}
                for key, value in metrics_row.items():
                    if key.endswith(chosen_suffix) and pd.notna(value):
                        track_row[key[: -len(chosen_suffix)]] = float(value)
                if len(track_row) > 1:
                    track_rows.append(track_row)

        if not model_names or metric_name is None:
            continue

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df.to_csv(plots_dir / f"{task_name}_comparison_summary.csv", index=False)

            summary_metric_cols = [
                col
                for col in summary_df.columns
                if col != "model"
                and (
                    col == "loss"
                    or col.startswith("mean/")
                    or col.startswith("mean_")
                    or col in {"accuracy", "f1", "pearson", "spearman", "mse", "mae"}
                )
                and pd.api.types.is_numeric_dtype(summary_df[col])
            ]
            if summary_metric_cols:
                summary_matrix = summary_df.set_index("model")[summary_metric_cols]
                summary_matrix = _normalize_comparison_frame(summary_matrix)
                _save_heatmap(
                    summary_matrix,
                    plots_dir / f"{task_name}_comparison_summary_heatmap.png",
                    f"{task_name} summary metrics",
                    xlabel="metric",
                    ylabel="model",
                    cbar_label="score",
                )

        if track_rows:
            track_df = pd.DataFrame(track_rows).set_index("model")
            track_df = track_df.loc[:, sorted(track_df.columns)]
            track_df = _normalize_comparison_frame(track_df)
            track_label = track_metric_name or "selected metric"
            _save_heatmap(
                track_df,
                plots_dir / f"{task_name}_comparison_tracks_heatmap.png",
                f"{task_name} per-track performance",
                xlabel="track",
                ylabel="model",
                cbar_label=str(track_label),
            )

        bar_items = list(zip(model_names, values))
        bar_items.sort(key=lambda item: item[1], reverse=_metric_higher_is_better(metric_name))
        model_names = [name for name, _ in bar_items]
        values = [value for _, value in bar_items]

        best_idx = 0 if values else None
        base_color = "#2f6fed"
        best_color = "#2ca02c"
        bar_colors = [best_color if idx == best_idx else base_color for idx in range(len(values))]

        plt.figure(figsize=(max(6, len(model_names) * 1.5), 4.5))
        bars = plt.bar(model_names, values, color=bar_colors)
        metric_label = _display_metric_label(metric_name)
        plt.ylabel(metric_label)
        plt.title(f"{task_name} comparison ({metric_label})")
        plt.xticks(rotation=20, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.3)

        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        if bars:
            bars[0].set_edgecolor("black")
            bars[0].set_linewidth(1.2)

        plt.tight_layout()
        plt.savefig(plots_dir / f"{task_name}_comparison.png", dpi=200)
        plt.close()


def generate_run_plots(
    run_dir: Path,
    task_type: str,
    track_names: list[str],
    metadata_df: pd.DataFrame | None = None,
) -> None:
    """Generate training-curve and per-track comparison plots for a single run."""
    history_path = run_dir / "metrics_history.csv"
    if history_path.exists():
        try:
            hist = pd.read_csv(history_path)
            if hist.empty:
                return

            epoch_col = None
            for col in ["epoch", "step"]:
                if col in hist.columns:
                    epoch_col = col
                    break
            if epoch_col is None:
                epoch_col = hist.columns[0]

            fig, ax1 = plt.subplots(figsize=(7, 4))

            bounded_metrics = []
            unbounded_metrics = []

            for col in hist.columns:
                if col.startswith("mean/") or col.startswith("mean_"):
                    if col not in ["mean_loss", "mean_losses"]:
                        metric_name = col.replace("mean/", "").replace("mean_", "")
                        if metric_name.lower() in ["mse", "mae"]:
                            unbounded_metrics.append((col, metric_name))
                        else:
                            bounded_metrics.append((col, metric_name))

            if bounded_metrics:
                for col, label in sorted(bounded_metrics):
                    ax1.plot(hist[epoch_col], hist[col], label=label, linewidth=1.5)

            loss_col = None
            for col in ["loss", "mean_loss"]:
                if col in hist.columns:
                    loss_col = col
                    break
            if loss_col:
                ax1.plot(hist[epoch_col], hist[loss_col], label="loss", color="#d9534f", linewidth=2)

            ax1.set_xlabel("epoch" if epoch_col == "epoch" else "step")
            ax1.set_ylabel("Bounded metrics (correlation, accuracy, loss)", color="black")
            ax1.tick_params(axis="y", labelcolor="black")
            ax1.grid(alpha=0.3)

            if unbounded_metrics:
                ax2 = ax1.twinx()
                for col, label in sorted(unbounded_metrics):
                    ax2.plot(hist[epoch_col], hist[col], label=label, linewidth=1.5, linestyle="--")
                ax2.set_ylabel("Unbounded metrics (MSE, MAE)", color="black")
                ax2.tick_params(axis="y", labelcolor="black")
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
            else:
                ax1.legend(fontsize=8, loc="best")

            plt.title("Training curves")
            plt.tight_layout()
            plt.savefig(run_dir / "training_curve.png", dpi=200)
            plt.close()
        except Exception as e:
            _LOGGER.warning("[Warning] Failed to generate training_curve.png")
            _LOGGER.warning(f"  Run Dir: {run_dir}")
            _LOGGER.warning(f"  Error:   {e}")

    final_path = run_dir / "final_metrics.csv"
    if not final_path.exists():
        return
    try:
        final = pd.read_csv(final_path)
        if final.empty:
            return
        row = final.iloc[-1]

        if task_type == "regression":
            suffix = "/pearson"
        else:
            suffix = "/accuracy"
            if not any(c.endswith(suffix) for c in row.index):
                suffix = "/f1"

        labels = []
        values = []
        for t in track_names:
            key = f"{t}{suffix}"
            if key in row.index:
                labels.append(t)
                values.append(float(row[key]))

        if not labels:
            return

        display_labels = []
        if metadata_df is not None and "file_id" in metadata_df.columns:
            meta_map = dict(zip(metadata_df["file_id"], metadata_df.get("assay", metadata_df.get("name", metadata_df["file_id"]))))
            for l in labels:
                display_labels.append(f"{l} ({meta_map.get(l, '')})")
        else:
            display_labels = labels

        plt.figure(figsize=(max(6, len(labels) * 0.4), 4))
        bars = plt.bar(range(len(values)), values, color="#2f6fed")
        plt.xticks(range(len(values)), display_labels, rotation=90, fontsize=7)
        plt.ylabel(suffix.lstrip("/"))
        plt.title("Per-track performance")
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        for bar, val in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )
        plt.tight_layout()
        plt.savefig(run_dir / "track_comparison.png", dpi=200)
        plt.close()
    except Exception:
        pass
