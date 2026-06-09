from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.ddp_runtime_utils import _is_main_process
from src.utils.logging_utils import get_benchmark_logger


_LOGGER = get_benchmark_logger()

try:
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer  # type: ignore
except Exception:
    AutoConfig = None  # type: ignore
    AutoModelForMaskedLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

_LOCAL_GENOME_LM = Path(__file__).resolve().parents[2] / "genome-lm"
if _LOCAL_GENOME_LM.exists() and str(_LOCAL_GENOME_LM) not in sys.path:
    sys.path.insert(0, str(_LOCAL_GENOME_LM))

try:
    from genomelm.biflash import BiFlashModel  # type: ignore
    from genomelm.bert import BertModel  # type: ignore
    from genomelm.modernbert_pure import ModernBertPureModel  # type: ignore
except Exception:
    from genome_lm.train.models.base import GenomeLM
    from genome_lm.train.models.bert import BertModel
    from genome_lm.train.models.biflash import BiFlashModel
    from genome_lm.train.models.components.biflash import BiFlashConfig, BiFlashDecoder
    from genome_lm.train.models.components.modern_bert_components.modeling import (
        ModernBertConfig,
        ModernBertModel,
    )
    from genome_lm.train.models.modernbert_pure import ModernBertPureModel


if "GenomeLM" not in globals():
    from genome_lm.train.models.base import GenomeLM

# Import FixedSpeciesVocab for extracting species information from checkpoints
try:
    from genome_lm.train.models.components.fixed_species_vocab import FixedSpeciesVocab
except ImportError:
    FixedSpeciesVocab = None  # type: ignore


class RegressionHead(nn.Module):
    def __init__(self, hidden_dim: int, num_tracks: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, num_tracks)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.norm(hidden_states)
        x = self.proj(x)
        x = F.softplus(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int, num_tracks: int):
        super().__init__()
        self.num_tracks = num_tracks
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, num_tracks*2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.norm(hidden_states)
        x = self.proj(x)
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_tracks, 2)
        return x


def crop_center(hidden_states: torch.Tensor, keep_fraction: float) -> torch.Tensor:
    if keep_fraction <= 0.0 or keep_fraction > 1.0:
        raise ValueError("keep_fraction must be in (0, 1].")
    if keep_fraction >= 1.0:
        return hidden_states

    seq_len = hidden_states.shape[1]
    crop_len = max(1, int(seq_len * keep_fraction))
    start = max(0, (seq_len - crop_len) // 2)
    end = start + crop_len
    return hidden_states[:, start:end, ...]


class BenchmarkModelBase(nn.Module, ABC):
    """Shared interface for benchmark models.

    Concrete models should expose a tokenizer, a backbone, and a task-specific
    head while reusing the common token/input handling and output extraction.
    """

    def __init__(self, keep_target_center_fraction: float = 1.0) -> None:
        super().__init__()
        self.keep_target_center_fraction = float(keep_target_center_fraction)
        self.tokenizer = DNACharTokenizer()

    @staticmethod
    def _resolve_input_ids(
        tokens: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ids = input_ids if input_ids is not None else tokens
        if ids is None:
            raise ValueError("Provide either `tokens` or `input_ids`.")
        return ids

    @staticmethod
    def _extract_hidden_states(backbone_output: Any) -> torch.Tensor:
        if isinstance(backbone_output, torch.Tensor):
            return backbone_output

        # 1. Prioritize HF attributes FIRST, as they correctly handle tuple unpacking
        if hasattr(backbone_output, "hidden_states") and backbone_output.hidden_states is not None:
            hidden_states = backbone_output.hidden_states
            if isinstance(hidden_states, (tuple, list)) and hidden_states:
                last = hidden_states[-1]
                if isinstance(last, torch.Tensor):
                    return last

        if hasattr(backbone_output, "last_hidden_state") and backbone_output.last_hidden_state is not None:
            last_hidden_state = backbone_output.last_hidden_state
            if isinstance(last_hidden_state, torch.Tensor):
                return last_hidden_state

        # 2. Fallback Dictionary logic (Modified to unpack tuples if found)
        if isinstance(backbone_output, dict):
            if "hidden_states" in backbone_output:
                hs = backbone_output["hidden_states"]
                if isinstance(hs, (tuple, list)) and hs:
                    last = hs[-1]
                    if isinstance(last, torch.Tensor):
                        return last
                        
            for key in ("last_hidden_state", "x", "output"):
                value = backbone_output.get(key)
                if isinstance(value, torch.Tensor):
                    return value
            raise KeyError("Backbone dict output does not include a valid tensor hidden state.")

        # 3. Fallback Tuple/List logic
        if isinstance(backbone_output, (tuple, list)):
            for candidate in reversed(backbone_output):
                if isinstance(candidate, torch.Tensor):
                    return candidate
            raise TypeError("Backbone tuple/list output does not contain a tensor hidden state.")

        raise TypeError(f"Unsupported backbone output type: {type(backbone_output)}")

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


@dataclass
class DNACharTokenizer:
    max_vocab_size: int = 6

    def __post_init__(self) -> None:
        self.vocab = {
            "A": 0,
            "C": 1,
            "G": 2,
            "T": 3,
            "N": 4,
        }
        self.unk_id = 5 if self.max_vocab_size >= 6 else self.max_vocab_size - 1

    def __call__(
        self,
        sequence: str,
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int = 1024,
        add_special_tokens: bool = False,
        return_tensors: str | None = "pt",
    ) -> dict[str, torch.Tensor]:
        del add_special_tokens

        seq = sequence.upper()
        ids = [self.vocab.get(ch, self.unk_id) for ch in seq]

        if truncation:
            ids = ids[:max_length]
        if padding == "max_length" and len(ids) < max_length:
            ids = ids + [self.unk_id] * (max_length - len(ids))

        tensor = torch.tensor(ids, dtype=torch.long)
        if return_tensors == "pt":
            tensor = tensor.unsqueeze(0)

        return {"input_ids": tensor}


class _BiFlashBackbone(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.config = BiFlashConfig(hidden_size=hidden_dim)

        self.decoders = nn.ModuleList(
            [
                BiFlashDecoder(self.config, is_global=i in self.config.global_layer_list)
                for i in range(self.config.num_layers)
            ]
        )
        self.meta_token = nn.Parameter(
            torch.randn(self.config.num_meta_tokens, self.config.hidden_size)
        )
        self.in_proj = nn.Linear(6, self.config.hidden_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(input_ids.long(), num_classes=6).to(torch.float32)
        hidden = self.in_proj(one_hot)
        hidden = torch.cat([self.meta_token.expand(hidden.size(0), -1, -1), hidden], dim=1)
        for layer in self.decoders:
            hidden = layer(hidden)
        hidden = hidden[:, self.config.num_meta_tokens :]
        return hidden


class _SimpleTransformerBackbone(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        model = BertModel(d_model=hidden_dim)
        model.configure_model()
        if model.transformer is None:
            raise RuntimeError("BertModel failed to build transformer backbone.")
        self.transformer = model.transformer

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(input_ids.long(), num_classes=6).to(torch.float32)
        self.transformer.cache_freqs(one_hot.shape[1])
        _, hidden = self.transformer(one_hot, return_last_hidden=True)
        return hidden


class _ModernBertBackbone(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        num_heads = max(1, hidden_dim // 64)
        if hidden_dim % num_heads != 0:
            num_heads = 1

        config = ModernBertConfig(
            vocab_size=6,
            hidden_size=hidden_dim,
            intermediate_size=max(hidden_dim * 3 // 2, hidden_dim),
            num_hidden_layers=12,
            num_attention_heads=num_heads,
            num_kv_heads=num_heads,
            max_position_embeddings=8192,
            pad_token_id=0,
            eos_token_id=1,
            bos_token_id=None,
            unk_token_id=2,
            mask_token_id=3,
        )
        self.backbone = ModernBertModel(config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone(input_ids.long())


class _LoadedGenomeLMBackbone(nn.Module):
    def __init__(self, loaded_model: Any):
        super().__init__()
        # Disable torch.compile and set to eval mode
        if hasattr(loaded_model, 'torch_compile'):
            loaded_model.torch_compile = False
        loaded_model.eval()
        loaded_model.configure_model()

        self.mode: str
        self.transformer: Any = None
        self.decoders: Any = None
        self.meta_token: Any = None
        self.in_proj: Any = None
        self.num_meta_tokens: int = 0
        self.backbone: Any = None
        self.default_species_id: int | None = None  # Track the actual species ID

        model_getter = getattr(loaded_model, "_get_model", None)
        model_core: Any = model_getter() if callable(model_getter) else None
        if model_core is None:
            raise RuntimeError(
                "Loaded genome-lm checkpoint did not provide a core model via _get_model()."
            )

        # Extract species vocab information from the loaded model if available
        if hasattr(loaded_model, "hparams"):
            self._extract_species_vocab_info(loaded_model.hparams)

        if hasattr(model_core, "cache_freqs") and hasattr(model_core, "out_norm"):
            self.mode = "bert"
            self.transformer = model_core
            # Unwrap torch.compile wrapper if present
            if hasattr(self.transformer, "_orig_mod"):
                self.transformer = self.transformer._orig_mod
            # Disable last_layer_f32 to avoid dtype conflicts during inference
            if hasattr(self.transformer, "last_layer_f32"):
                self.transformer.last_layer_f32 = False
            # Convert to float16 for inference (model was trained in bfloat16)
            self.transformer = self.transformer.to(torch.float16)
            self.hidden_dim = int(self.transformer.out_norm.weight.shape[0])
            return

        if (
            hasattr(model_core, "decoders")
            and hasattr(model_core, "meta_token")
            and hasattr(model_core, "in_proj")
        ):
            self.mode = "biflash"
            self.decoders = model_core.decoders
            self.meta_token = model_core.meta_token
            self.in_proj = model_core.in_proj
            # Convert to float16 for inference
            for i, layer in enumerate(self.decoders):
                self.decoders[i] = layer.to(torch.float16)
            # For parameters, use .data = to avoid reassignment error
            if isinstance(self.meta_token, torch.nn.Parameter):
                self.meta_token.data = self.meta_token.data.to(torch.float16)
            else:
                self.meta_token = self.meta_token.to(torch.float16)
            if isinstance(self.in_proj, torch.nn.Parameter):
                self.in_proj.data = self.in_proj.data.to(torch.float16)
            else:
                self.in_proj = self.in_proj.to(torch.float16)
            self.hidden_dim = int(self.meta_token.shape[-1])
            self.num_meta_tokens = int(self.meta_token.shape[0])
            return

        if hasattr(model_core, "backbone"):
            self.mode = "modernbert"
            self.backbone = model_core.backbone
            # Convert to float16 for inference
            self.backbone = self.backbone.to(torch.float16)
            self.hidden_dim = int(self.backbone.config.hidden_size)
            return

        raise RuntimeError(
            "Loaded genome-lm checkpoint has an unsupported model structure "
            f"('{type(model_core).__name__}')."
        )

    def _extract_species_vocab_info(self, hparams: Any) -> None:
        """Extract species vocab information from model hyperparameters.
        
        This determines the real species ID that should be used during inference.
        Species IDs are indices into a sorted list of species under a taxonomy root.
        """
        if FixedSpeciesVocab is None:
            return
            
        try:
            species_vocab_tree_taxid = getattr(hparams, "species_vocab_tree_taxid", None)
            if species_vocab_tree_taxid is None:
                return
            
            # Reconstruct the FixedSpeciesVocab from the checkpoint's taxid
            # Find the genome-lm data directory for taxonomy.db
            genome_lm_dir = Path(__file__).resolve().parents[2] / "genome-lm"
            taxonomy_path = genome_lm_dir / "data" / "taxonomy.db"
            cache_dir = genome_lm_dir / "data" / "species_vocab"
            
            if taxonomy_path.exists():
                vocab = FixedSpeciesVocab(
                    str(species_vocab_tree_taxid),
                    taxonomy_path=taxonomy_path,
                    cache_dir=cache_dir
                )
                
                # Species ID 0 is the first species in the sorted list
                if vocab.anchor_lineages and len(vocab.anchor_lineages) > 0:
                    first_species_taxid = vocab.anchor_lineages[0][0]
                    self.default_species_id = 0
                    _LOGGER.info("[Species Vocab]")
                    _LOGGER.info(f"  Taxonomy Tree:  {species_vocab_tree_taxid}")
                    _LOGGER.info(f"  Species ID 0:   {first_species_taxid}")
                    _LOGGER.info(f"  Vocab Size:     {vocab.size}")
            else:
                # taxonomy.db not found, just log the taxid
                _LOGGER.info("[Species Vocab]")
                _LOGGER.info(f"  Taxonomy Tree:  {species_vocab_tree_taxid}")
                _LOGGER.info(f"  Taxonomy DB:    missing at {taxonomy_path}")
                _LOGGER.info("  Species ID:     default 0")
                self.default_species_id = 0
        except Exception as e:
            _LOGGER.info("[Species Vocab]")
            _LOGGER.info(f"  Could not extract species vocab info: {e}")
            self.default_species_id = 0  # Fallback to 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.mode == "bert":
            one_hot = F.one_hot(input_ids.long(), num_classes=6).to(torch.float16)
            self.transformer.cache_freqs(one_hot.shape[1])
            # If the transformer has species embeddings, provide the real species ID
            # (extracted from checkpoint metadata); otherwise pass None.
            species = None
            if hasattr(self.transformer, "species_embeddings") and self.transformer.species_embeddings is not None:
                # Use the species ID extracted from checkpoint's vocab, or default to 0
                # Create a batch tensor [batch_size] with the species ID repeated for each sample
                species_id = self.default_species_id if self.default_species_id is not None else 0
                batch_size = input_ids.shape[0]
                species = torch.full((batch_size,), species_id, dtype=torch.long, device=input_ids.device)
            
            _, hidden = self.transformer(one_hot, species=species, return_last_hidden=True)
            return hidden

        if self.mode == "biflash":
            one_hot = F.one_hot(input_ids.long(), num_classes=6).to(torch.float16)
            hidden = self.in_proj(one_hot)
            hidden = torch.cat([self.meta_token.expand(hidden.size(0), -1, -1), hidden], dim=1)
            for layer in self.decoders:
                hidden = layer(hidden)
            hidden = hidden[:, self.num_meta_tokens :]
            return hidden

        return self.backbone(input_ids.long())


def _build_fresh_genome_backbone(backbone_type: str, hidden_dim: int) -> nn.Module:
    """Build a fresh genome-lm backbone without loading from checkpoint."""
    backbone_type = backbone_type.strip().lower()
    
    if backbone_type in {"biflash", "biflashmodel"}:
        try:
            model = BiFlashModel(hidden_size=hidden_dim)
            model.configure_model()
            if model.model is None:
                raise RuntimeError("BiFlashModel failed to build internal model.")
            core = model.model
            if hasattr(core, "in_proj") and hasattr(core, "decoders") and hasattr(core, "meta_token"):
                # Convert the LM core to a hidden-state-producing backbone.
                return _BiFlashBackbone(hidden_dim)
        except Exception:
            return _BiFlashBackbone(hidden_dim)
        return _BiFlashBackbone(hidden_dim)

    if backbone_type in {"bert", "bertmodel"}:
        return _SimpleTransformerBackbone(hidden_dim)

    if backbone_type in {"modernbert", "modernbert_pure", "modernbertpure"}:
        return _ModernBertBackbone(hidden_dim)

    raise ValueError(
        f"Unsupported backbone_type '{backbone_type}'. "
        "Expected one of: 'biflash', 'bert', 'modernbert_pure'."
    )


def _load_genome_backbone_from_checkpoint(backbone_type: str, checkpoint_path: str) -> nn.Module:
    """Load a genome-lm backbone from a checkpoint file."""
    bt = backbone_type.strip().lower()

    if bt in {"biflash", "biflashmodel"}:
        from genome_lm.train.models.biflash import BiFlashModel as _BiFlashModel

        loaded = _BiFlashModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
        return _LoadedGenomeLMBackbone(loaded)

    if bt in {"bert", "bertmodel"}:
        from genome_lm.train.models.bert import BertModel as _BertModel

        # Load with species_vocab parameters set to None to avoid requiring taxonomy.db
        # The actual values will be extracted from hparams during _extract_species_vocab_info
        # Use strict=False because checkpoint may have _orig_mod wrapper from torch.compile
        loaded = _BertModel.load_from_checkpoint(
            checkpoint_path, 
            map_location="cpu",
            species_vocab_tree_taxid=None,
            species_vocab_size=None,
            strict=False
        )
        return _LoadedGenomeLMBackbone(loaded)

    if bt in {"modernbert", "modernbert_pure", "modernbertpure"}:
        from genome_lm.train.models.modernbert_pure import ModernBertPureModel as _ModernModel

        loaded = _ModernModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
        return _LoadedGenomeLMBackbone(loaded)

    raise ValueError(
        f"Unsupported backbone_type for checkpoint loading '{backbone_type}'. "
        "Expected one of: 'biflash', 'bert', 'modernbert_pure'."
    )


def _infer_genome_hidden_dim(backbone: nn.Module) -> int:
    """Infer hidden dimension from a genome-lm backbone."""
    if hasattr(backbone, "hidden_dim"):
        return int(getattr(backbone, "hidden_dim"))
    config = getattr(backbone, "config", None)
    if config is not None and hasattr(config, "hidden_size"):
        return int(getattr(config, "hidden_size"))
    for name in ("norm", "out_norm", "final_norm"):
        module = getattr(backbone, name, None)
        if module is not None and hasattr(module, "weight"):
            return int(module.weight.shape[0])
    raise RuntimeError("Could not infer hidden_dim from loaded backbone.")


def _build_genome_backbone(
    backbone_type: str,
    checkpoint_path: str | None = None,
    hidden_dim: int = 768,
    freeze_backbone: bool = True,
) -> tuple[nn.Module, int, None]:
    """Build a genome-lm backbone (either fresh or from checkpoint).
    
    Returns:
        (backbone, hidden_dim, None)  # tokenizer is None for genome-lm models
    """
    backbone_type = backbone_type.strip().lower()
    
    if checkpoint_path:
        backbone = _load_genome_backbone_from_checkpoint(backbone_type, checkpoint_path)
        inferred_dim = _infer_genome_hidden_dim(backbone)
    else:
        backbone = _build_fresh_genome_backbone(backbone_type, hidden_dim)
        inferred_dim = hidden_dim
    
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
    
    return backbone, inferred_dim, None


def _build_hf_backbone(
    model_name: str,
    freeze_backbone: bool = True,
    compile_backbone: bool = False,
) -> tuple[nn.Module, int, Any]:
    """Build a Hugging Face backbone with proper tokenizer and configuration.
    
    Returns:
        (backbone, hidden_dim, tokenizer)
    """
    if AutoConfig is None or AutoModelForMaskedLM is None or AutoTokenizer is None:
        raise ImportError(
            "transformers is required to use HF models. Install it in the benchmark environment."
        )

    # HF models should use their own tokenizer to preserve training-time tokenization behavior.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif getattr(tokenizer, "unk_token", None) is not None:
            tokenizer.pad_token = tokenizer.unk_token

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    backbone = AutoModelForMaskedLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        config=config,
    ).to(torch.float32)


    if compile_backbone and hasattr(torch, "compile"):
        backbone = torch.compile(backbone)

    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False

    hidden_dim = _infer_hf_hidden_dim(config)
    return backbone, hidden_dim, tokenizer


def _infer_hf_hidden_dim(config: Any) -> int:
    """Infer hidden dimension from a Hugging Face model config."""
    for attr in ("hidden_size", "embed_dim", "n_embd", "d_model"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    raise RuntimeError("Could not infer hidden size from HF config.")


class _HFBackboneWrapper(nn.Module):
    """Wraps an HF backbone to handle device movement and mixed precision."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> Any:
        # Ensure input is on same device as backbone
        first_param = next(self.backbone.parameters(), None)
        if first_param is not None:
            input_ids = input_ids.to(device=first_param.device)

        outputs = self.backbone(
            input_ids=input_ids.long(),
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        return outputs


class GenericModelWithHead(BenchmarkModelBase):
    """Generic wrapper that composes an arbitrary backbone with a task head.

    This class keeps the head logic independent from the backbone implementation.
    Provide an already-built `backbone` module and the `tokenizer` to use
    (or rely on the BenchmarkModelBase default tokenizer).
    """

    def __init__(
        self,
        backbone: nn.Module,
        tokenizer: Any | None,
        task_type: str,
        num_tracks: int,
        hidden_dim: int,
        keep_target_center_fraction: float = 0.375,
    ) -> None:
        super().__init__(keep_target_center_fraction=keep_target_center_fraction)
        self.backbone = backbone
        if tokenizer is not None:
            self.tokenizer = tokenizer
        self.task_type = task_type.strip().lower()
        self.hidden_dim = int(hidden_dim)
        self.num_tracks = int(num_tracks)

        if self.task_type == "regression":
            self.head = RegressionHead(self.hidden_dim, self.num_tracks)
        elif self.task_type == "classification":
            self.head = ClassificationHead(self.hidden_dim, self.num_tracks)
        else:
            raise ValueError("Unsupported task_type for GenericModelWithHead. Expected 'regression' or 'classification'.")

    def forward(self, tokens: torch.Tensor | None = None, input_ids: torch.Tensor | None = None, **kwargs: Any) -> dict[str, torch.Tensor]:
        del kwargs
        ids = self._resolve_input_ids(tokens=tokens, input_ids=input_ids)
        outputs = self.backbone(input_ids=ids.long(), output_hidden_states=True)
        hidden_states = self._extract_hidden_states(outputs)
        if self.keep_target_center_fraction < 1.0:
            hidden_states = crop_center(hidden_states, self.keep_target_center_fraction)

        if self.task_type == "regression":
            bigwig_logits = self.head(hidden_states)
            return {"bigwig_tracks_logits": bigwig_logits}
        return {"logits": self.head(hidden_states)}


def build_benchmark_model(
    model_cfg: dict[str, Any],
    task_type: str,
    num_tracks: int,
) -> nn.Module:
    model_type = str(model_cfg.get("type", "")).strip().lower()
    keep_target_center_fraction = float(model_cfg.get("keep_target_center_fraction", 0.375))
    freeze_backbone = bool(model_cfg.get("freeze_backbone", False))
    hidden_dim = int(model_cfg.get("hidden_dim", 768))

    if model_type in {"hf", "huggingface", "transformers"}:
        model_name = str(
            model_cfg.get(
                "model_name",
                model_cfg.get("checkpoint", model_cfg.get("name", "")),
            )
        )
        if not model_name:
            raise ValueError("HF model configs must provide 'model_name' or 'checkpoint'.")

        # Build HF backbone
        backbone, hf_hidden_dim, tokenizer = _build_hf_backbone(
            model_name=model_name,
            freeze_backbone=freeze_backbone,
            compile_backbone=bool(model_cfg.get("compile_backbone", False)),
        )
        # Wrap for device/dtype handling
        backbone = _HFBackboneWrapper(backbone)
        # Wrap with GenericModelWithHead to attach task-specific head
        return GenericModelWithHead(
            backbone=backbone,
            tokenizer=tokenizer,
            task_type=task_type,
            num_tracks=num_tracks,
            hidden_dim=hf_hidden_dim,
            keep_target_center_fraction=keep_target_center_fraction,
        )

    checkpoint_path = model_cfg.get("checkpoint")
    # Build genome-lm backbone
    backbone, genome_hidden_dim, _ = _build_genome_backbone(
        backbone_type=model_type,
        checkpoint_path=checkpoint_path,
        hidden_dim=hidden_dim,
        freeze_backbone=freeze_backbone,
    )
    # Wrap with GenericModelWithHead to attach task-specific head
    return GenericModelWithHead(
        backbone=backbone,
        tokenizer=None,
        task_type=task_type,
        num_tracks=num_tracks,
        hidden_dim=genome_hidden_dim,
        keep_target_center_fraction=keep_target_center_fraction,
    )


