"""Backbone plus LoRA, and the letter readout at answer slots (docs/API_SPEC.md section 5).

Full-vocabulary logits are never materialised. The decoder returns final hidden
states, the states at the slots are gathered, and letter logits are their product
with the lm_head rows of the allowed letters, in fp32. Results come back as a flat
list aligned with flat_index(): request 0's questions in order, then request 1's.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from jevmark.encode import Encoded, load_tokenizer


@dataclass(frozen=True)
class Batch:
    input_ids: torch.Tensor  # (B, T) long, right-padded
    attention_mask: torch.Tensor  # (B, T) long, 1 for real tokens
    slot_positions: tuple[tuple[int, ...], ...]
    letter_ids: tuple[tuple[tuple[int, ...], ...], ...]


def collate(encoded: Sequence[Encoded], pad_id: int) -> Batch:
    """Right-pad a list of encoded requests. Slot positions index real tokens, so padding leaves them unchanged."""
    if not encoded:
        raise ValueError("encoded_batch: at least one request is required")
    longest = max(len(e.input_ids) for e in encoded)
    input_ids = torch.full((len(encoded), longest), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), longest), dtype=torch.long)
    for row, e in enumerate(encoded):
        input_ids[row, : len(e.input_ids)] = torch.tensor(e.input_ids, dtype=torch.long)
        attention_mask[row, : len(e.input_ids)] = 1
    return Batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        slot_positions=tuple(e.slot_positions for e in encoded),
        letter_ids=tuple(e.letter_ids for e in encoded),
    )


def half_backbone(config: Mapping[str, Any], device: torch.device) -> bool:
    """Frozen backbone weights in fp16 only on CUDA with precision.autocast fp16; fp32 everywhere else."""
    return device.type == "cuda" and config.get("precision", {}).get("autocast") == "fp16"


def keep_lora_fp32(model: torch.nn.Module) -> None:
    """Cast LoRA parameters to fp32, whatever dtype the backbone is in."""
    for name, param in model.named_parameters():
        if "lora_" in name and param.dtype != torch.float32:
            param.data = param.data.float()


@dataclass(frozen=True)
class LoadSource:
    """What JevMark.load was called with, so the fp32 fallback can reload the same model."""

    config: Mapping[str, Any]
    checkpoint: Path | None
    device: torch.device


def flat_index(encoded: Sequence[Encoded]) -> list[tuple[int, int]]:
    """(request index, question index) for each entry of the flat per-question lists."""
    return [(r, q) for r, e in enumerate(encoded) for q in range(len(e.slot_positions))]


class JevMark:
    """A causal LM (optionally wrapped in a peft LoRA model) read out at answer slots."""

    def __init__(
        self,
        model: PreTrainedModel | PeftModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_tokens: int,
        temperature: float = 1.0,
        model_id: str = "jevmark",
        autocast_dtype: torch.dtype | None = None,
        source: LoadSource | None = None,
    ) -> None:
        self.model = model
        self.source = source
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_id = model_id
        self.autocast_dtype = autocast_dtype
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has neither a pad token nor an eos token to pad with")
        self.pad_id: int = pad_id

    @classmethod
    def load(
        cls,
        config: Mapping[str, Any],
        checkpoint: str | Path | None = None,
        device: str | torch.device | None = None,
        half: bool | None = None,
    ) -> JevMark:
        """Backbone and max_tokens from config, plus the adapter, calibration.json and model_id.txt of a run directory.

        Raises RuntimeError if checkpoint is given but has no adapter directory.
        Precision (decision 28): on CUDA with precision.autocast fp16, the frozen backbone
        loads in fp16 and runs under fp16 autocast; LoRA parameters stay fp32 and the
        letter readout is fp32. Elsewhere everything is fp32. half overrides the policy
        (the tests use it to exercise the fp16 path on CPU).
        """
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        use_half = half_backbone(config, device) if half is None else half
        tokenizer = load_tokenizer(config)
        backbone = config["backbone"]
        model: PreTrainedModel | PeftModel = AutoModelForCausalLM.from_pretrained(
            backbone["id"], revision=backbone.get("revision"), dtype=torch.float16 if use_half else torch.float32
        )
        temperature = 1.0
        model_id = f"jevmark-{config['run_name']}"
        if checkpoint is not None:
            run_dir = Path(checkpoint)
            adapter_dir = run_dir / "adapter"
            if not adapter_dir.is_dir():
                raise RuntimeError(f"checkpoint missing: no adapter directory at {adapter_dir}")
            model = PeftModel.from_pretrained(model, adapter_dir)
            keep_lora_fp32(model)
            calibration = run_dir / "calibration.json"
            if calibration.is_file():
                temperature = float(json.loads(calibration.read_text())["temperature"])
            model_id_file = run_dir / "model_id.txt"
            if model_id_file.is_file():
                model_id = model_id_file.read_text().strip()
        model.to(device).eval()
        return cls(
            model,
            tokenizer,
            max_tokens=int(config["max_tokens"]),
            temperature=temperature,
            model_id=model_id,
            autocast_dtype=torch.float16 if use_half else None,
            source=LoadSource(config, Path(checkpoint) if checkpoint is not None else None, device),
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def use_fp32(self) -> None:
        """The fallback when the first batch has NaN or inf slot logits: reload the same model in fp32.

        Reloading reads the checkpoint's original weights; casting fp16 weights up would
        keep their fp16 rounding. A model built without JevMark.load has no source to
        reload from and is cast to fp32 in place. The reloaded model is in eval mode.
        """
        if self.source is not None:
            reloaded = JevMark.load(self.source.config, self.source.checkpoint, self.source.device, half=False)
            self.model = reloaded.model
        else:
            self.model.float()
        self.autocast_dtype = None

    def _autocast(self) -> contextlib.AbstractContextManager:
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def slot_logits(self, encoded_batch: Sequence[Encoded]) -> list[torch.Tensor]:
        """Per-question fp32 letter logits, gradients enabled, flat in flat_index() order."""
        batch = collate(encoded_batch, self.pad_id)
        device = self.device
        # get_decoder() reaches the inner Qwen3Model through a PeftModel too, and
        # LoRA layers are injected into that module, so the adapter is active.
        decoder = self.model.get_decoder()
        with self._autocast():
            hidden = decoder(
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
                use_cache=False,
            ).last_hidden_state

        rows = [r for r, slots in enumerate(batch.slot_positions) for _ in slots]
        positions = [p for slots in batch.slot_positions for p in slots]
        question_letters = [ids for per_request in batch.letter_ids for ids in per_request]
        vocab_rows = sorted({i for ids in question_letters for i in ids})
        column = {token_id: c for c, token_id in enumerate(vocab_rows)}

        head = self.model.get_output_embeddings()
        rows_index = torch.tensor(vocab_rows, device=device)
        with torch.autocast(device_type=device.type, enabled=False):
            slot_hidden = hidden[torch.tensor(rows, device=device), torch.tensor(positions, device=device)].float()
            scores = slot_hidden @ head.weight[rows_index].float().T  # (questions, distinct letters)
            if getattr(head, "bias", None) is not None:
                scores = scores + head.bias[rows_index].float()
        return [scores[n, [column[i] for i in ids]] for n, ids in enumerate(question_letters)]

    def forward_distributions(self, encoded_batch: Sequence[Encoded]) -> list[torch.Tensor]:
        """Per-question answer distributions: softmax(letter logits / temperature), no gradients, flat order."""
        with torch.no_grad():
            return [torch.softmax(x / self.temperature, dim=-1) for x in self.slot_logits(encoded_batch)]
