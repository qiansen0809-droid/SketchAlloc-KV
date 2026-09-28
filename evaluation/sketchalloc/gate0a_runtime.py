"""Minimal runtime helpers for Gate 0A fixed-action replication."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from transformers import DynamicCache

from evaluation.sketchalloc.metrics import continuation_nll
from kvpress.presses.base_press import BasePress


@dataclass
class PromptState:
    cache: DynamicCache
    first_answer_logits: torch.Tensor


@torch.inference_mode()
def prefill_context(model, context_ids: torch.Tensor, press: BasePress | None = None) -> DynamicCache:
    if context_ids.dim() != 2 or context_ids.shape[0] != 1:
        raise ValueError("context_ids must have shape [1, time]")

    device = next(model.parameters()).device
    context_ids = context_ids.to(device)
    cache = DynamicCache()

    if press is None:
        model.model(input_ids=context_ids, past_key_values=cache, use_cache=True)
    else:
        with press(model):
            model.model(input_ids=context_ids, past_key_values=cache, use_cache=True)
    return cache


def clone_cache(cache: DynamicCache) -> DynamicCache:
    return copy.deepcopy(cache)


@torch.inference_mode()
def build_full_prompt_state(model, context_cache: DynamicCache, query_ids: torch.Tensor) -> PromptState:
    if query_ids.dim() != 2 or query_ids.shape[0] != 1:
        raise ValueError("query_ids must have shape [1, time]")
    if query_ids.shape[1] == 0:
        raise ValueError("query_ids must not be empty")

    device = next(model.parameters()).device
    cache = clone_cache(context_cache)
    outputs = model(
        input_ids=query_ids.to(device),
        past_key_values=cache,
        use_cache=True,
    )
    return PromptState(
        cache=cache,
        first_answer_logits=outputs.logits[:, -1, :].detach(),
    )


@torch.inference_mode()
def answer_nll(model, prompt_state: PromptState, answer_ids: torch.Tensor) -> float:
    if answer_ids.dim() != 2 or answer_ids.shape[0] != 1:
        raise ValueError("answer_ids must have shape [1, time]")
    if answer_ids.shape[1] == 0:
        raise ValueError("answer_ids must not be empty")

    device = next(model.parameters()).device
    answer_ids = answer_ids.to(device)
    cache = clone_cache(prompt_state.cache)
    first_logits = prompt_state.first_answer_logits.unsqueeze(1)

    if answer_ids.shape[1] == 1:
        logits = first_logits
    else:
        outputs = model(
            input_ids=answer_ids[:, :-1],
            past_key_values=cache,
            use_cache=True,
        )
        logits = torch.cat([first_logits, outputs.logits], dim=1)

    return float(continuation_nll(logits, answer_ids).item())


@torch.inference_mode()
def best_gold_answer_nll(model, tokenizer, prompt_state: PromptState, answers: list[str]) -> tuple[float, list[float]]:
    scores: list[float] = []
    for answer in answers:
        ids = tokenizer(
            str(answer),
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids
        if ids.shape[1] == 0:
            continue
        scores.append(answer_nll(model, prompt_state, ids))

    if not scores:
        raise ValueError("no non-empty gold answers to score")
    return min(scores), scores
