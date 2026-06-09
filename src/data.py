from __future__ import annotations

import fnmatch
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import pyBigWig
import torch
from huggingface_hub import HfApi, snapshot_download
from pyfaidx import Fasta
from torch.utils.data import DataLoader, Dataset

from src.utils.logging_utils import get_benchmark_logger

DEFAULT_WINDOW_BP = 1024


_LOGGER = get_benchmark_logger()


@dataclass(frozen=True)
class GenomicsInputs:
    fasta_path: str
    splits_df: pd.DataFrame
    metadata_df: pd.DataFrame
    bigwig_paths: list[str] = field(default_factory=list)
    bigwig_ids: list[str] = field(default_factory=list)
    bed_paths: list[str] = field(default_factory=list)
    bed_ids: list[str] = field(default_factory=list)
    
    @property
    def num_tracks(self) -> int:
        """Number of output tracks (bigwig files for regression, bed files for classification)."""
        if self.bigwig_ids:
            return len(self.bigwig_ids)
        elif self.bed_ids:
            return len(self.bed_ids)
        else:
            return 1

    def num_tracks_for_task(self, task_type: str) -> int:
        """Return number of tracks appropriate for a given task type.

        Args:
            task_type: 'regression' or 'classification'

        This is a lightweight helper that downstream code can call after
        `prepare_genomics_inputs()` has been invoked for a specific task/species.
        It avoids adding persistent task-specific fields and keeps the
        recomputation/reset semantics simple: call this per task.
        """
        if task_type == "regression":
            return len(self.bigwig_ids) if self.bigwig_ids else 1
        if task_type == "classification":
            return len(self.bed_ids) if self.bed_ids else 1
        raise ValueError(f"Unknown task_type: {task_type}")


def center_window_bounds(center: int, window_size: int = DEFAULT_WINDOW_BP) -> tuple[int, int]:
    """Return half-open [start, end) bounds for a window centered at center."""
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    half = window_size // 2
    start = center - half
    end = start + window_size
    return start, end


def prepare_genomics_inputs(
    species: str,
    data_cache_dir: str | Path = "data",
    hf_repo_id: str = "InstaDeepAI/NTv3_benchmark_dataset",
    bigwig_file_ids: Optional[list[str]] = None,
    bed_file_ids: Optional[list[str]] = None,
) -> GenomicsInputs:
    """
    Download and prepare FASTA, BigWig, splits, and metadata from the HF dataset.
    """
    cache = Path(data_cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    species_dir = cache / species
    bigwig_dir = species_dir / "functional_tracks"
    bed_dir = species_dir / "genome_annotation"

    def _local_track_map(directory: Path, suffix: str) -> dict[str, Path]:
        if not directory.exists():
            return {}
        return {path.stem: path for path in sorted(directory.glob(f"*{suffix}"))}

    metadata_file = "benchmark_metadata.tsv"
    download_patterns = [metadata_file, f"{species}/genome.fasta", f"{species}/splits.bed"]
    use_local_cache = species_dir.exists()

    # If the cache already contains the species directory, stay fully offline and
    # resolve file IDs from the local filesystem.
    if use_local_cache:
        available_bigwig_files = _local_track_map(bigwig_dir, ".bigwig")
        available_bed_files = _local_track_map(bed_dir, ".bed")

        if bigwig_file_ids is not None:
            missing_bw = set(bigwig_file_ids) - set(available_bigwig_files.keys())
            if missing_bw:
                raise FileNotFoundError(
                    "Requested BigWig files are not present in the local cache: "
                    f"{sorted(missing_bw)}. Expected them under {bigwig_dir}."
                )
        if bed_file_ids is not None:
            missing_bed = set(bed_file_ids) - set(available_bed_files.keys())
            if missing_bed:
                raise FileNotFoundError(
                    "Requested BED files are not present in the local cache: "
                    f"{sorted(missing_bed)}. Expected them under {bed_dir}."
                )
    else:
        # If the user requested specific files, resolve their repo paths via the HF API.
        if bigwig_file_ids is not None or bed_file_ids is not None:
            api = HfApi()
            files = api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset")
            species_files = [p for p in files if fnmatch.fnmatch(p, f"{species}/**")]

            available_bigwig_files = {
                Path(p).stem: p for p in species_files if Path(p).suffix == ".bigwig"
            }
            available_bed_files = {
                Path(p).stem: p for p in species_files if Path(p).suffix == ".bed"
            }

            if bigwig_file_ids is not None:
                missing_bw = set(bigwig_file_ids) - set(available_bigwig_files.keys())
                if missing_bw:
                    raise ValueError(
                        "Requested BigWig files not found: "
                        f"{sorted(missing_bw)}. Available files: "
                        f"{sorted(available_bigwig_files.keys())}"
                    )
                for file_id in bigwig_file_ids:
                    download_patterns.append(available_bigwig_files[file_id])
            else:
                download_patterns.append(f"{species}/functional_tracks/*.bigwig")

            if bed_file_ids is not None:
                missing_bed = set(bed_file_ids) - set(available_bed_files.keys())
                if missing_bed:
                    raise ValueError(
                        "Requested BED files not found: "
                        f"{sorted(missing_bed)}. Available files: "
                        f"{sorted(available_bed_files.keys())}"
                    )
                for file_id in bed_file_ids:
                    download_patterns.append(available_bed_files[file_id])
            else:
                download_patterns.append(f"{species}/genome_annotation/*.bed")
        else:
            # No explicit file lists requested: download the canonical patterns.
            download_patterns.append(f"{species}/functional_tracks/*.bigwig")
            download_patterns.append(f"{species}/genome_annotation/*.bed")

    if use_local_cache:
        local_dir = cache
    else:
        local_dir = Path(
            snapshot_download(
                repo_id=hf_repo_id,
                repo_type="dataset",
                allow_patterns=download_patterns,
                local_dir=str(cache),
            )
        )

    fasta_path = str(local_dir / species / "genome.fasta")

    bigwig_dir = local_dir / species / "functional_tracks"
    if bigwig_file_ids is not None:
        bigwig_paths = [str(bigwig_dir / f"{file_id}.bigwig") for file_id in bigwig_file_ids]
        bigwig_ids = list(bigwig_file_ids)
    else:
        bigwig_files = sorted(bigwig_dir.glob("*.bigwig"))
        bigwig_paths = [str(p) for p in bigwig_files]
        bigwig_ids = [p.stem for p in bigwig_files]

    bed_dir = local_dir / species / "genome_annotation"
    if bed_file_ids is not None:
        bed_paths = [str(bed_dir / f"{file_id}.bed") for file_id in bed_file_ids]
        bed_ids = list(bed_file_ids)
    else:
        bed_files = sorted(bed_dir.glob("*.bed"))
        bed_paths = [str(p) for p in bed_files]
        bed_ids = [p.stem for p in bed_files]

    splits_df = pd.read_csv(
        local_dir / species / "splits.bed",
        sep="\t",
        header=None,
        names=["chr_name", "start", "end", "split"],
        dtype={"chr_name": str, "start": int, "end": int, "split": str},
    )

    metadata_df = pd.read_csv(local_dir / metadata_file, sep="\t")
    metadata_df = metadata_df[metadata_df["species_common_name"] == species].reset_index(drop=True)
    # If we have bigwig tracks, re-order metadata to match them; otherwise keep per-species metadata
    if bigwig_ids:
        metadata_df = metadata_df.set_index("file_id").loc[bigwig_ids].reset_index()

    return GenomicsInputs(
        fasta_path=fasta_path,
        splits_df=splits_df,
        metadata_df=metadata_df,
        bigwig_paths=bigwig_paths,
        bigwig_ids=bigwig_ids,
        bed_paths=bed_paths,
        bed_ids=bed_ids,
    )


def create_targets_scaling_fn(metadata_df: pd.DataFrame) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a target scaling function from track means with smooth clipping."""
    if "mean" not in metadata_df.columns:
        raise ValueError("metadata_df must contain a 'mean' column.")

    track_means = metadata_df["mean"].to_numpy(dtype=np.float32)

    finite_track_means = track_means[np.isfinite(track_means)]
    if finite_track_means.size == 0:
        raise ValueError("metadata_df['mean'] must contain at least one finite value.")

    track_means_tensor = torch.tensor(track_means, dtype=torch.float32)

    def transform_fn(x: torch.Tensor) -> torch.Tensor:
        means = track_means_tensor.to(x.device)
        scaled = x / means
        clipped = torch.where(
            scaled > 10.0,
            2.0 * torch.sqrt(scaled * 10.0) - 10.0,
            scaled,
        )
        return clipped

    return transform_fn


def _merge_intervals(intervals: list[tuple[int, int]]) -> int:
    """Merge half-open intervals and return the covered length."""
    cleaned = sorted((int(start), int(end)) for start, end in intervals if int(end) > int(start))
    if not cleaned:
        return 0

    covered = 0
    current_start, current_end = cleaned[0]
    for start, end in cleaned[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            covered += current_end - current_start
            current_start, current_end = start, end
    covered += current_end - current_start
    return covered


def _coverage_fraction_in_regions(
    intervals_df: pd.DataFrame,
    regions_df: pd.DataFrame,
) -> float:
    """Compute the fraction of bases covered by intervals within the given regions."""
    total_region_bases = int((regions_df["end"] - regions_df["start"]).sum())
    if total_region_bases <= 0:
        return 0.0

    region_groups: dict[str, list[tuple[int, int]]] = {}
    for row in regions_df.itertuples(index=False):
        chrom = str(getattr(row, "chr_name"))
        start = int(getattr(row, "start"))
        end = int(getattr(row, "end"))
        if end > start:
            region_groups.setdefault(chrom, []).append((start, end))

    interval_groups: dict[str, list[tuple[int, int]]] = {}
    for row in intervals_df.itertuples(index=False):
        chrom = str(getattr(row, "chr"))
        start = int(getattr(row, "start"))
        end = int(getattr(row, "end"))
        if end > start:
            interval_groups.setdefault(chrom, []).append((start, end))

    covered_intervals: list[tuple[int, int]] = []
    for chrom, chrom_intervals in interval_groups.items():
        chrom_regions = sorted(region_groups.get(chrom, []))
        if not chrom_regions:
            continue

        chrom_intervals = sorted(chrom_intervals)
        interval_idx = 0
        region_idx = 0
        while interval_idx < len(chrom_intervals) and region_idx < len(chrom_regions):
            interval_start, interval_end = chrom_intervals[interval_idx]
            region_start, region_end = chrom_regions[region_idx]

            overlap_start = max(interval_start, region_start)
            overlap_end = min(interval_end, region_end)
            if overlap_start < overlap_end:
                covered_intervals.append((overlap_start, overlap_end))

            if interval_end <= region_end:
                interval_idx += 1
            else:
                region_idx += 1

    covered_bases = _merge_intervals(covered_intervals)
    return covered_bases / float(total_region_bases)


def _print_dataset_summary(
    *,
    task_type: str,
    label: str,
    split: str,
    window_size: int,
    batch_size: int,
    chrom_regions: pd.DataFrame,
    bigwig_paths: list[str] | None = None,
    bigwig_ids: list[str] | None = None,
    bed_paths: list[str] | None = None,
    bed_ids: list[str] | None = None,
    metadata_df: pd.DataFrame | None = None,
) -> None:
    """Print a task-agnostic data summary using actual metadata and annotation files."""
    split_regions = chrom_regions[chrom_regions["split"] == split].copy()
    if split_regions.empty:
        return

    header = f"[Data Summary - {label}]"
    sep = "-" * 41
    total_split_bases = int((split_regions["end"] - split_regions["start"]).sum())

    _LOGGER.info(header)
    _LOGGER.info(f"  Task Type:    {task_type}")
    _LOGGER.info(f"  Split:        {split}")
    _LOGGER.info(sep)
    _LOGGER.info("Data Scope:")
    _LOGGER.info(f"  Window Size:     {window_size:>13,}")
    _LOGGER.info(f"  Batch Size:      {batch_size:>13,}")
    _LOGGER.info(f"  Split Bases:     {total_split_bases:>13,}")

    if task_type == "regression":
        track_ids = list(bigwig_ids or [])
        if metadata_df is None or "mean" not in metadata_df.columns:
            _LOGGER.info("  Tracks:          NA (metadata missing)")
            _LOGGER.info(sep)
            return

        track_means = metadata_df["mean"].to_numpy(dtype=np.float32)
        finite_track_means = track_means[np.isfinite(track_means)]
        _LOGGER.info("Track Means:")
        _LOGGER.info(f"  Tracks:          {len(track_ids) if track_ids else len(finite_track_means)}")
        _LOGGER.info(f"  Min Mean:        {finite_track_means.min():.6g}")
        _LOGGER.info(f"  Mean:            {finite_track_means.mean():.6g}")
        _LOGGER.info(f"  Std Mean:         {finite_track_means.std():.6g}")
        _LOGGER.info(f"  Max Mean:        {finite_track_means.max():.6g}")
        if track_ids and len(track_ids) <= 8:
            for track_id, track_mean in zip(track_ids, track_means):
                _LOGGER.info(f"    {track_id}: {track_mean:.6g}")
        _LOGGER.info(sep)
        return

    track_ids = list(bed_ids or [])
    if not bed_paths:
        _LOGGER.info("  Tracks:          NA (no BED files)")
        _LOGGER.info(sep)
        return

    _LOGGER.info("Class Imbalance:")
    _LOGGER.info(f"  Tracks:          {len(bed_paths)}")
    positive_fractions: list[tuple[str, float]] = []
    for bed_path, track_id in zip(bed_paths, track_ids or [Path(p).stem for p in bed_paths]):
        bed_df = pd.read_csv(
            bed_path,
            sep="\t",
            header=None,
            usecols=[0, 1, 2],
            names=["chr", "start", "end"],
            dtype={"chr": str, "start": int, "end": int},
        )
        pos_fraction = _coverage_fraction_in_regions(bed_df, split_regions)
        positive_fractions.append((track_id, pos_fraction))

    pos_values = np.array([value for _, value in positive_fractions], dtype=np.float64)
    _LOGGER.info(f"  Positive Fraction (min/mean/max): {pos_values.min():.6g} / {pos_values.mean():.6g} / {pos_values.max():.6g}")
    _LOGGER.info(f"  Negative Fraction (mean):        {1.0 - pos_values.mean():.6g}")
    if len(positive_fractions) <= 8:
        for track_id, pos_fraction in positive_fractions:
            neg_fraction = 1.0 - pos_fraction
            ratio = float("inf") if pos_fraction <= 0 else neg_fraction / pos_fraction
            _LOGGER.info(f"    {track_id}: pos={pos_fraction:.6g} neg={neg_fraction:.6g} neg/pos={ratio:.6g}")
    _LOGGER.info(sep)


class GenomeBigWigDataset(Dataset):
    """
    Dataset for genome sequence + BigWig tracks with strict 1024bp centered windows.

    Returned sample keys:
    - tokens: LongTensor of shape (1024,)
    - bigwig_targets: FloatTensor of shape (1024, num_tracks)
    - chrom: str
    - start: int
    - end: int
    - center: int
    """

    def __init__(
        self,
        fasta_path: str,
        bigwig_path_list: list[str],
        chrom_regions: pd.DataFrame,
        split: str,
        tokenizer,
        transform_fn: Callable[[torch.Tensor], torch.Tensor],
        num_samples: int,
        window_size: int = DEFAULT_WINDOW_BP,
        target_center_col: Optional[str] = None,
        seed: Optional[int] = None,
        keep_target_center_fraction: float = 1.0,
        max_sampling_attempts: int = 20,
    ) -> None:
        super().__init__()

        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if split not in set(chrom_regions["split"].astype(str).unique()):
            raise ValueError(f"Split '{split}' was not found in chrom_regions['split'].")
        if keep_target_center_fraction <= 0 or keep_target_center_fraction > 1:
            raise ValueError("keep_target_center_fraction must be in (0, 1].")
        if window_size <= 0:
            raise ValueError("window_size must be positive.")

        self.fasta_path = str(Path(fasta_path).resolve())
        self.bigwig_paths = [str(Path(p).resolve()) for p in bigwig_path_list]
        self.chrom_regions = chrom_regions.copy()
        self.split = split
        self.tokenizer = tokenizer
        self.transform_fn = transform_fn
        self.num_samples = int(num_samples)
        self.window_size = window_size
        self.target_center_col = target_center_col
        self.keep_target_center_fraction = keep_target_center_fraction
        self.max_sampling_attempts = max_sampling_attempts

        # Compute bigwig center window size (centered subset of full window)
        self.bigwig_center_window_size = int(window_size * keep_target_center_fraction)

        self._rng = random.Random(seed)
        self._fasta_handle: Optional[Fasta] = None
        self._bigwig_handles: Optional[list[pyBigWig.pyBigWig]] = None

        split_regions = self.chrom_regions[self.chrom_regions["split"] == split].copy()
        if split_regions.empty:
            raise ValueError(f"No rows found for split='{split}'.")

        self._candidate_rows = self._build_candidates(split_regions)
        if not self._candidate_rows:
            raise ValueError(
                f"No valid regions available for split='{split}' with window_size={self.window_size}."
            )

    def _build_candidates(self, split_regions: pd.DataFrame) -> list[dict[str, int | str]]:
        half = self.window_size // 2
        candidates: list[dict[str, int | str]] = []

        use_target_center = (
            self.target_center_col is not None and self.target_center_col in split_regions.columns
        )

        for row in split_regions.itertuples(index=False):
            chrom = getattr(row, "chr_name")
            region_start = int(getattr(row, "start"))
            region_end = int(getattr(row, "end"))

            if region_end - region_start < self.window_size:
                continue

            if use_target_center:
                center = int(getattr(row, self.target_center_col))
                start, end = center_window_bounds(center, self.window_size)
                if start < region_start or end > region_end:
                    continue
                candidates.append(
                    {
                        "chr_name": chrom,
                        "region_start": region_start,
                        "region_end": region_end,
                        "fixed_center": center,
                    }
                )
            else:
                min_center = region_start + half
                max_center = region_end - (self.window_size - half)
                if min_center > max_center:
                    continue
                candidates.append(
                    {
                        "chr_name": chrom,
                        "region_start": region_start,
                        "region_end": region_end,
                        "min_center": min_center,
                        "max_center": max_center,
                    }
                )

        return candidates

    def _get_fasta_handle(self) -> Fasta:
        if self._fasta_handle is None:
            self._fasta_handle = Fasta(
                self.fasta_path,
                as_raw=True,
                sequence_always_upper=True,
            )
        return self._fasta_handle

    def _get_bigwig_handles(self) -> list[pyBigWig.pyBigWig]:
        if self._bigwig_handles is None:
            handles: list[pyBigWig.pyBigWig] = []
            for path in self.bigwig_paths:
                if not Path(path).exists():
                    raise FileNotFoundError(f"BigWig file not found: {path}")
                bw = pyBigWig.open(path)
                if bw is None:
                    raise RuntimeError(f"Failed to open BigWig file: {path}")
                handles.append(bw)
            self._bigwig_handles = handles
        return self._bigwig_handles

    def __len__(self) -> int:
        return self.num_samples

    def _sample_window(self) -> tuple[str, int, int, int]:
        candidate = self._rng.choice(self._candidate_rows)
        chrom = str(candidate["chr_name"])

        if "fixed_center" in candidate:
            center = int(candidate["fixed_center"])
        else:
            center = self._rng.randint(int(candidate["min_center"]), int(candidate["max_center"]))

        start, end = center_window_bounds(center, self.window_size)
        return chrom, start, end, center

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        del idx

        fasta = self._get_fasta_handle()
        bigwigs = self._get_bigwig_handles()

        for _ in range(self.max_sampling_attempts):
            chrom, start, end, center = self._sample_window()

            seq = fasta[chrom][start:end]
            if len(seq) != self.window_size:
                continue

            tokenized = self.tokenizer(
                seq,
                padding="max_length",
                truncation=True,
                max_length=self.window_size,
                add_special_tokens=False,
                return_tensors="pt",
            )
            tokens = tokenized["input_ids"][0]
            if tokens.shape[0] != self.window_size:
                continue

            # Extract centered window for targets
            center_offset = int(self.window_size * (1 - self.keep_target_center_fraction) // 2)
            bigwig_start = start + center_offset
            bigwig_end = bigwig_start + self.bigwig_center_window_size

            bigwig_targets_np = np.array(
                [bw.values(chrom, bigwig_start, bigwig_end, numpy=True) for bw in bigwigs],
                dtype=np.float32,
            )
            if bigwig_targets_np.ndim != 2:
                continue

            # (num_tracks, center_window_size) -> (center_window_size, num_tracks)
            bigwig_targets_np = bigwig_targets_np.T
            if bigwig_targets_np.shape[0] != self.bigwig_center_window_size:
                continue

            bigwig_targets = torch.tensor(bigwig_targets_np, dtype=torch.float32)
            bigwig_targets = torch.nan_to_num(bigwig_targets, nan=0.0)
            bigwig_targets = self.transform_fn(bigwig_targets)

            # The transform_fn should not change the sequence length.
            if bigwig_targets.shape[0] != self.bigwig_center_window_size:
                continue

            return {
                "tokens": tokens,
                "bigwig_targets": bigwig_targets,
                "chrom": chrom,
                "start": start,
                "end": end,
                "center": center,
            }

        raise RuntimeError(
            "Failed to sample a valid 1024bp centered window after "
            f"{self.max_sampling_attempts} attempts."
        )

    def close(self) -> None:
        if self._bigwig_handles is not None:
            for handle in self._bigwig_handles:
                try:
                    handle.close()
                except Exception:
                    pass
            self._bigwig_handles = None

        if self._fasta_handle is not None:
            try:
                self._fasta_handle.close()
            except Exception:
                pass
            self._fasta_handle = None

    def __del__(self) -> None:
        self.close()




class GenomeBedDataset(Dataset):
    """
    Dataset for genome sequence + BED annotation tracks with discrete binary targets.

    Returned sample keys:
    - tokens: LongTensor of shape (1024,)
    - bed_targets: LongTensor of shape (bed_center_window, num_tracks) with values in {0, 1}
    - chrom: str
    - start: int
    - end: int
    - center: int
    """

    def __init__(
        self,
        fasta_path: str,
        bed_path_list: list[str],
        chrom_regions: pd.DataFrame,
        split: str,
        tokenizer,
        keep_target_center_fraction: float = 0.375,
        num_samples: int = 1000,
        window_size: int = DEFAULT_WINDOW_BP,
        seed: Optional[int] = None,
        max_sampling_attempts: int = 20,
    ) -> None:
        super().__init__()

        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if split not in set(chrom_regions["split"].astype(str).unique()):
            raise ValueError(f"Split '{split}' was not found in chrom_regions['split'].")
        if keep_target_center_fraction <= 0 or keep_target_center_fraction > 1:
            raise ValueError("keep_target_center_fraction must be in (0, 1].")
        if window_size <= 0:
            raise ValueError("window_size must be positive.")

        self.fasta_path = str(Path(fasta_path).resolve())
        self.bed_paths = [str(Path(p).resolve()) for p in bed_path_list]
        self.chrom_regions = chrom_regions.copy()
        self.split = split
        self.tokenizer = tokenizer
        self.keep_target_center_fraction = keep_target_center_fraction
        self.num_samples = int(num_samples)
        self.window_size = window_size
        self.max_sampling_attempts = max_sampling_attempts

        # Compute bed center window size (centered subset of full window)
        self.bed_center_window_size = int(window_size * keep_target_center_fraction)

        self._rng = random.Random(seed)
        self._fasta_handle: Optional[Fasta] = None
        self._bed_handles: Optional[list[pd.DataFrame]] = None

        split_regions = self.chrom_regions[self.chrom_regions["split"] == split].copy()
        if split_regions.empty:
            raise ValueError(f"No rows found for split='{split}'.")

        self._candidate_rows = self._build_candidates(split_regions)
        if not self._candidate_rows:
            raise ValueError(
                f"No valid regions available for split='{split}' with window_size={self.window_size}."
            )

    def _build_candidates(self, split_regions: pd.DataFrame) -> list[dict[str, int | str]]:
        half = self.window_size // 2
        candidates: list[dict[str, int | str]] = []

        for row in split_regions.itertuples(index=False):
            chrom = getattr(row, "chr_name")
            region_start = int(getattr(row, "start"))
            region_end = int(getattr(row, "end"))

            if region_end - region_start < self.window_size:
                continue

            min_center = region_start + half
            max_center = region_end - (self.window_size - half)
            if min_center > max_center:
                continue

            candidates.append(
                {
                    "chr_name": chrom,
                    "region_start": region_start,
                    "region_end": region_end,
                    "min_center": min_center,
                    "max_center": max_center,
                }
            )

        return candidates

    def _get_fasta_handle(self) -> Fasta:
        if self._fasta_handle is None:
            self._fasta_handle = Fasta(
                self.fasta_path,
                as_raw=True,
                sequence_always_upper=True,
            )
        return self._fasta_handle

    def _get_bed_handles(self) -> list[pd.DataFrame]:
        """Load all BED files into DataFrames (cached)."""
        if self._bed_handles is None:
            handles: list[pd.DataFrame] = []
            for path in self.bed_paths:
                if not Path(path).exists():
                    raise FileNotFoundError(f"BED file not found: {path}")
                # Load BED as DataFrame (chr, start, end columns)
                bed_df = pd.read_csv(
                    path,
                    sep="\t",
                    header=None,
                    usecols=[0, 1, 2],
                    names=["chr", "start", "end"],
                    dtype={"chr": str, "start": int, "end": int},
                )
                handles.append(bed_df)
            self._bed_handles = handles
        return self._bed_handles

    def __len__(self) -> int:
        return self.num_samples

    def _sample_window(self) -> tuple[str, int, int, int]:
        candidate = self._rng.choice(self._candidate_rows)
        chrom = str(candidate["chr_name"])
        center = self._rng.randint(int(candidate["min_center"]), int(candidate["max_center"]))
        start, end = center_window_bounds(center, self.window_size)
        return chrom, start, end, center

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        del idx

        fasta = self._get_fasta_handle()
        bed_dfs = self._get_bed_handles()

        for _ in range(self.max_sampling_attempts):
            chrom, start, end, center = self._sample_window()

            # Get sequence
            seq = fasta[chrom][start:end]
            if len(seq) != self.window_size:
                continue

            tokenized = self.tokenizer(
                seq,
                padding="max_length",
                truncation=True,
                max_length=self.window_size,
                add_special_tokens=False,
                return_tensors="pt",
            )
            tokens = tokenized["input_ids"][0]
            if tokens.shape[0] != self.window_size:
                continue

            # Extract centered window for targets
            center_offset = int(self.window_size * (1 - self.keep_target_center_fraction) // 2)
            bed_start = start + center_offset
            bed_end = bed_start + self.bed_center_window_size

            # Create binary targets from BED annotations
            bed_targets_np = np.zeros((self.bed_center_window_size, len(bed_dfs)), dtype=np.int64)

            for bed_idx, bed_df in enumerate(bed_dfs):
                # Find all BED regions that overlap with [bed_start, bed_end)
                overlapping = bed_df[
                    (bed_df["chr"] == chrom)
                    & (bed_df["start"] < bed_end)
                    & (bed_df["end"] > bed_start)
                ]

                # Mark positions covered by overlapping regions
                for _, row in overlapping.iterrows():
                    region_start = max(row["start"], bed_start)
                    region_end = min(row["end"], bed_end)
                    bed_targets_np[region_start - bed_start:region_end - bed_start, bed_idx] = 1

            # Validate shape
            if bed_targets_np.shape != (self.bed_center_window_size, len(bed_dfs)):
                continue

            bed_targets = torch.tensor(bed_targets_np, dtype=torch.int64)

            return {
                "tokens": tokens,
                "bed_targets": bed_targets,
                "chrom": chrom,
                "start": start,
                "end": end,
                "center": center,
            }

        raise RuntimeError(
            "Failed to sample a valid window with BED annotations after "
            f"{self.max_sampling_attempts} attempts."
        )

    def close(self) -> None:
        if self._fasta_handle is not None:
            try:
                self._fasta_handle.close()
            except Exception:
                pass
            self._fasta_handle = None

    def __del__(self) -> None:
        self.close()


def create_dataset_and_loader(
    fasta_path: str,
    chrom_regions: pd.DataFrame,
    split: str,
    tokenizer,
    transform_fn: Callable[[torch.Tensor], torch.Tensor],
    num_samples: int,
    batch_size: int,
    num_workers: int = 0,
    bigwig_path_list: list[str] = None,
    bed_path_list: Optional[list[str]] = None,
    bigwig_ids: list[str] | None = None,
    bed_ids: list[str] | None = None,
    metadata_df: pd.DataFrame | None = None,
    target_center_col: Optional[str] = None,
    seed: Optional[int] = None,
    shuffle: Optional[bool] = None,
    task_type: str = "regression",
    keep_target_center_fraction: float = 0.375,
    window_size: int = DEFAULT_WINDOW_BP,
) -> tuple:
    """Create dataset and loader for regression or classification tasks.
    
    Args:
        task_type: "regression" (BigWig) or "classification" (BED)
        keep_target_center_fraction: For classification, target window fraction
    """
    if task_type not in ("regression", "classification"):
        raise ValueError(f"task_type must be 'regression' or 'classification', got {task_type}")
    
    if task_type == "classification":
        dataset = GenomeBedDataset(
            fasta_path=fasta_path,
            bed_path_list=bed_path_list,
            chrom_regions=chrom_regions,
            split=split,
            tokenizer=tokenizer,
            keep_target_center_fraction=keep_target_center_fraction,
            num_samples=num_samples,
            window_size=window_size,
            seed=seed,
        )
    else:
        dataset = GenomeBigWigDataset(
            fasta_path=fasta_path,
            bigwig_path_list=bigwig_path_list,
            chrom_regions=chrom_regions,
            split=split,
            tokenizer=tokenizer,
            transform_fn=transform_fn,
            num_samples=num_samples,
            window_size=window_size,
            target_center_col=target_center_col,
            keep_target_center_fraction=keep_target_center_fraction,
            seed=seed,
        )

    summary_label = f"{split}/{task_type}"
    _print_dataset_summary(
        task_type=task_type,
        label=summary_label,
        split=split,
        window_size=window_size,
        batch_size=batch_size,
        chrom_regions=chrom_regions,
        bigwig_paths=bigwig_path_list,
        bigwig_ids=bigwig_ids,
        bed_paths=bed_path_list,
        bed_ids=bed_ids,
        metadata_df=metadata_df,
    )

    if shuffle is None:
        shuffle = split == "train"

    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        worker_init_fn=seed_worker if seed is not None else None,
        generator=g if seed is not None else None, 
    )
    
    return dataset, loader
