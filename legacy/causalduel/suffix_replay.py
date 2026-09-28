from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import DynamicCache

from kvpress.presses.base_press import BasePress


@dataclass
class ReplayResult:
    logits: torch.Tensor
    prefix_length: int
    suffix_length: int


@torch.inference_mode()
def replay_suffix(
    model,
    prefix_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    press: Optional[BasePress] = None,
) -> ReplayResult:
    """
    Prefill a prefix, optionally apply KV compression, then teacher-force the
    known suffix and return next-token logits for suffix positions.

    To score suffix token t, compare logits[:, t-1] with suffix_ids[:, t].
    The returned tensor therefore has length suffix_len - 1.
    """
    if prefix_ids.dim() != 2 or suffix_ids.dim() != 2:
        raise ValueError("prefix_ids and suffix_ids must have shape [batch, time]")
    if prefix_ids.shape[0] != 1 or suffix_ids.shape[0] != 1:
        raise ValueError("Gate-0 POC currently supports batch size 1")
    if suffix_ids.shape[1] < 2:
        raise ValueError("suffix must contain at least 2 tokens")

    device = next(model.parameters()).device
    prefix_ids = prefix_ids.to(device)
    suffix_ids = suffix_ids.to(device)
    prefix_len = prefix_ids.shape[1]

    cache = DynamicCache()

    ctx = press(model) if press is not None else _null_context()
    with ctx:
        model.model(
            input_ids=prefix_ids,
            past_key_values=cache,
            use_cache=True,
        )

    replay_inputs = suffix_ids[:, :-1]
    position_ids = torch.arange(
        prefix_len,
        prefix_len + replay_inputs.shape[1],
        device=device,
    ).unsqueeze(0)

    outputs = model(
        input_ids=replay_inputs,
        past_key_values=cache,
        position_ids=position_ids,
        use_cache=True,
    )

    return ReplayResult(
        logits=outputs.logits.detach(),
        prefix_length=prefix_len,
        suffix_length=suffix_ids.shape[1],
    )


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False
