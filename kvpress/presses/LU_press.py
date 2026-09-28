# kvpress/presses/lu_press.py
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

import numpy as np
import torch
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)


@dataclass
class LUPress(BasePress):
    press: ScorerPress
    budget_curve_path: Optional[str] = None

    sink: int = 4
    window: int = 32

    # Gate-0 hook: exact logical keep counts per KV head.
    # Mapping: layer_idx -> sequence of length num_kv_heads.
    # When a layer is present here, these counts replace LU-KV's static profile
    # for that layer only. This lets us construct budget-preserving swaps while
    # keeping the token scorer unchanged.
    keep_counts_override: Optional[dict[int, list[int]]] = None

    _budget_curves: Optional[np.ndarray] = field(init=False, repr=False, default=None)
    _global_kept_tokens: int = field(init=False, repr=False, default=0)
    _global_total_tokens: int = field(init=False, repr=False, default=0)
    _last_keep_counts: dict[int, list[int]] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "Only support `ScorerPress`"

    def _post_setup_init(self):
        if self.budget_curve_path:
            try:
                print(f"Wrapped Press: {self.press.__class__.__name__}")
                print(f"Budget Curves Path: {self.budget_curve_path}")

                self._budget_curves = np.load(self.budget_curve_path)
                print(f"Sink={self.sink}, Window={self.window}\n")
            except Exception as e:
                raise IOError(f"Loading Budget Curves Failed: {e}")
        else:
            print("No Budget Curves Loaded")

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    @property
    def last_keep_counts(self) -> dict[int, list[int]]:
        """Exact per-layer KV-head keep counts used in the latest prefill."""
        return {layer: counts.copy() for layer, counts in self._last_keep_counts.items()}

    def set_keep_counts_override(self, override: Optional[dict[int, list[int]]]):
        """Set an exact per-layer/per-KV-head logical budget override."""
        self.keep_counts_override = override

    def clear_keep_counts_override(self):
        self.keep_counts_override = None

    def _protected_count(self, seq_len: int) -> int:
        """Number of positions protected by sink + recent-window constraints."""
        return min(seq_len, max(0, self.sink) + max(0, self.window))

    def _curve_keep_counts(
        self,
        layer_idx: int,
        num_heads: int,
        seq_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self._budget_curves is None:
            return None

        target_idx = int(round(self.compression_ratio * 100)) - 1
        target_idx = max(0, min(98, target_idx))

        try:
            local_prune_ratios = torch.as_tensor(
                self._budget_curves[target_idx, layer_idx],
                device=device,
                dtype=torch.float32,
            )
        except IndexError:
            return None

        if local_prune_ratios.numel() != num_heads:
            raise ValueError(
                f"LU-KV curve head count mismatch at layer {layer_idx}: "
                f"curve has {local_prune_ratios.numel()} heads, runtime has {num_heads}."
            )

        head_keep_rates = 1.0 - local_prune_ratios
        ideal_keep_counts = head_keep_rates * seq_len

        total_keep_target = int(torch.round(ideal_keep_counts.sum()).item())
        keep_counts = torch.floor(ideal_keep_counts).long()
        remainder = total_keep_target - int(keep_counts.sum().item())

        if remainder > 0:
            fractional_parts = ideal_keep_counts - keep_counts
            num_to_distribute = min(remainder, num_heads)
            if num_to_distribute > 0:
                top_k_indices = torch.topk(fractional_parts, k=num_to_distribute).indices
                keep_counts[top_k_indices] += 1

        return keep_counts.clamp(min=1, max=seq_len)

    def _override_keep_counts(
        self,
        layer_idx: int,
        num_heads: int,
        seq_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not self.keep_counts_override or layer_idx not in self.keep_counts_override:
            return None

        keep_counts = torch.as_tensor(
            self.keep_counts_override[layer_idx],
            device=device,
            dtype=torch.long,
        )

        if keep_counts.numel() != num_heads:
            raise ValueError(
                f"keep_counts_override[{layer_idx}] must contain {num_heads} KV-head counts; "
                f"got {keep_counts.numel()}."
            )

        min_keep = max(1, self._protected_count(seq_len))

        # The official LU-KV static curve can assign fewer than sink+window
        # positions to some heads at high compression ratios. In that case the
        # sink/window settings are priority boosts in the token scorer, not a
        # hard per-head allocation floor. Gate 0 must therefore allow an
        # unchanged (or upward-adjusted) official LU budget below min_keep.
        #
        # We only forbid a *donor move* from reducing a head below the
        # sink/window floor. This preserves the registered donor constraint
        # without rejecting the LU baseline itself.
        curve_counts = self._curve_keep_counts(
            layer_idx=layer_idx,
            num_heads=num_heads,
            seq_len=seq_len,
            device=device,
        )
        if curve_counts is not None:
            is_donor_move = keep_counts < curve_counts
            invalid_donor = is_donor_move & (keep_counts < min_keep)
            if torch.any(invalid_donor):
                bad = keep_counts[invalid_donor].tolist()
                raise ValueError(
                    f"Gate-0 donor override violates sink/window protection at layer {layer_idx}. "
                    f"A donor head must keep at least {min_keep} positions; invalid counts: {bad}."
                )
        elif torch.any(keep_counts < min_keep):
            bad = keep_counts[keep_counts < min_keep].tolist()
            raise ValueError(
                f"Gate-0 override violates sink/window protection at layer {layer_idx}. "
                f"Each head must keep at least {min_keep} positions when no LU curve is available; "
                f"invalid counts: {bad}."
            )

        if torch.any(keep_counts > seq_len):
            raise ValueError(
                f"Gate-0 override exceeds prefix length {seq_len} at layer {layer_idx}."
            )

        return keep_counts

    def get_keep_counts(
        self,
        layer_idx: int,
        num_heads: int,
        seq_len: int,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        """
        Return the exact logical keep counts that will be used for one layer.

        Gate-0 overrides take precedence over the LU-KV static profile.
        """
        device = device or torch.device("cpu")
        override = self._override_keep_counts(layer_idx, num_heads, seq_len, device)
        if override is not None:
            return override

        return self._curve_keep_counts(layer_idx, num_heads, seq_len, device)

    @contextmanager
    def __call__(self, model: nn.Module) -> Generator:
        self._global_kept_tokens = 0
        self._global_total_tokens = 0
        self._last_keep_counts = {}
        with super().__call__(model):
            yield

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compression_ratio <= 0:
            return keys, values

        bsz, num_heads, seq_len, _ = keys.shape
        if bsz != 1:
            raise ValueError("LUPress head-wise masking currently expects batch size 1.")

        layer_idx = module.layer_idx
        final_keep_per_head = self.get_keep_counts(
            layer_idx=layer_idx,
            num_heads=num_heads,
            seq_len=seq_len,
            device=keys.device,
        )
        if final_keep_per_head is None:
            return keys, values

        self._last_keep_counts[layer_idx] = [int(x) for x in final_keep_per_head.tolist()]

        num_to_prune_per_head = seq_len - final_keep_per_head
        current_layer_kept = int(final_keep_per_head.sum().item())
        current_layer_total = seq_len * num_heads
        self._global_kept_tokens += current_layer_kept
        self._global_total_tokens += current_layer_total

        total_layers = getattr(module.config, "num_hidden_layers", None)
        if total_layers is not None and layer_idx == total_layers - 1:
            total_ratio = self._global_kept_tokens / self._global_total_tokens
            print(
                f"[LUPress] Final Global Keep Ratio: {total_ratio:.6f} "
                f"(Target: {1.0 - self.compression_ratio:.2f})"
            )

        if torch.all(num_to_prune_per_head <= 0):
            if hasattr(module, "masked_key_indices"):
                module.masked_key_indices = None
            return keys, values

        # Keep the original LU-KV token scorer fixed. Gate 0 changes only the
        # per-head budget, not the ranking within a head.
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)

        if self.sink > 0:
            safe_sink = min(self.sink, seq_len)
            scores[..., :safe_sink] = scores.max().item()
        if self.window > 0:
            start_idx = max(0, seq_len - self.window)
            scores[..., start_idx:] = scores.max().item()

        sorted_indices = torch.argsort(scores.squeeze(0), dim=-1, descending=True, stable=True)
        rank = torch.arange(seq_len, device=scores.device).expand_as(sorted_indices)
        keep_mask = rank < final_keep_per_head.unsqueeze(1)
        prune_mask = ~keep_mask

        pruned_seq_indices = sorted_indices[prune_mask]
        head_indices = (
            torch.arange(num_heads, device=scores.device)
            .unsqueeze(1)
            .expand_as(sorted_indices)[prune_mask]
        )
        batch_indices = torch.zeros_like(head_indices)

        module.masked_key_indices = (batch_indices, head_indices, pruned_seq_indices)
        return keys, values
