from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import DynamicCache

from evaluation.gate0.metrics import continuation_nll
from kvpress.presses.base_press import BasePress


@dataclass
class ProbeResult:
    logits: torch.Tensor
    targets: torch.Tensor
    probe_tokens: int


@dataclass
class PromptState:
    cache: DynamicCache
    first_answer_logits: torch.Tensor


@torch.inference_mode()
def prefill_context(
    model,
    context_ids: torch.Tensor,
    press: Optional[BasePress] = None,
) -> DynamicCache:
    """
    Prefill only the benchmark context.

    This matches the upstream LU-KV evaluation protocol, where the context is
    compressed first and the question is appended afterwards without further
    compression.
    """
    if context_ids.dim() != 2 or context_ids.shape[0] != 1:
        raise ValueError("context_ids must have shape [1, time]")

    device = next(model.parameters()).device
    context_ids = context_ids.to(device)
    cache = DynamicCache()

    if press is None:
        model.model(
            input_ids=context_ids,
            past_key_values=cache,
            use_cache=True,
        )
    else:
        with press(model):
            model.model(
                input_ids=context_ids,
                past_key_values=cache,
                use_cache=True,
            )

    return cache


def clone_cache(cache: DynamicCache) -> DynamicCache:
    # DynamicCache stores ordinary tensors/lists and is deepcopy-safe in the
    # transformers version used by the Gate-0 environment.
    return copy.deepcopy(cache)


@torch.inference_mode()
def replay_query_probe(
    model,
    context_cache: DynamicCache,
    query_ids: torch.Tensor,
    probe_len: int,
) -> ProbeResult:
    """
    Starting from an already-prefilled context cache, append any query prefix
    and teacher-force the last probe_len query tokens.

    The context cache is cloned, so multiple probe lengths can reuse the same
    expensive context prefill. As in suffix_replay.py, the first probe token is
    not scored; returned logits predict probe[1:].
    """
    if query_ids.dim() != 2 or query_ids.shape[0] != 1:
        raise ValueError("query_ids must have shape [1, time]")
    if probe_len < 2:
        raise ValueError("probe_len must be at least 2")

    device = next(model.parameters()).device
    query_ids = query_ids.to(device)

    actual_probe = min(int(probe_len), int(query_ids.shape[1]))
    if actual_probe < 2:
        raise ValueError("query must contain at least two tokens for behavioral replay")

    query_prefix = query_ids[:, :-actual_probe]
    probe = query_ids[:, -actual_probe:]

    cache = clone_cache(context_cache)

    if query_prefix.shape[1] > 0:
        model(
            input_ids=query_prefix,
            past_key_values=cache,
            use_cache=True,
        )

    outputs = model(
        input_ids=probe[:, :-1],
        past_key_values=cache,
        use_cache=True,
    )

    return ProbeResult(
        logits=outputs.logits.detach(),
        targets=probe[:, 1:].detach(),
        probe_tokens=actual_probe,
    )


@torch.inference_mode()
def build_full_prompt_state(
    model,
    context_cache: DynamicCache,
    query_ids: torch.Tensor,
) -> PromptState:
    """
    Append the full benchmark question + answer prefix to a cloned context
    cache. The returned last-position logits predict the first answer token.
    """
    if query_ids.dim() != 2 or query_ids.shape[0] != 1:
        raise ValueError("query_ids must have shape [1, time]")
    if query_ids.shape[1] == 0:
        raise ValueError("query_ids must not be empty")

    device = next(model.parameters()).device
    query_ids = query_ids.to(device)
    cache = clone_cache(context_cache)

    outputs = model(
        input_ids=query_ids,
        past_key_values=cache,
        use_cache=True,
    )

    return PromptState(
        cache=cache,
        first_answer_logits=outputs.logits[:, -1, :].detach(),
    )


@torch.inference_mode()
def answer_nll(
    model,
    prompt_state: PromptState,
    answer_ids: torch.Tensor,
) -> float:
    """Average teacher-forced NLL of one gold answer, conditioned on full prompt."""
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
def best_gold_answer_nll(
    model,
    tokenizer,
    prompt_state: PromptState,
    answers: list[str],
) -> tuple[float, list[float]]:
    """
    Score all accepted references and use the minimum average NLL. Gold answers
    are used only here as an offline label, never for candidate selection.
    """
    scores = []
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


@torch.inference_mode()
def greedy_generate(
    model,
    tokenizer,
    prompt_state: PromptState,
    max_new_tokens: int,
) -> str:
    """
    Greedy decode directly from a prepared prompt cache.

    We avoid model.generate() here so all conditions share exactly the same
    already-compressed context cache and no hidden re-prefill can occur.
    """
    if max_new_tokens <= 0:
        return ""

    cache = clone_cache(prompt_state.cache)
    logits = prompt_state.first_answer_logits
    generated = []

    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()

    for _ in range(int(max_new_tokens)):
        next_id = int(torch.argmax(logits, dim=-1).item())
        if next_id in eos_ids:
            break
        generated.append(next_id)

        token = torch.tensor(
            [[next_id]],
            dtype=torch.long,
            device=logits.device,
        )
        outputs = model(
            input_ids=token,
            past_key_values=cache,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]

    return tokenizer.decode(generated, skip_special_tokens=True).strip()



@torch.inference_mode()
def greedy_reference_trace(
    model,
    prompt_state: PromptState,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a short greedy continuation from the FullKV prompt state while
    recording the predictive distribution at every answer position.

    Returns:
      token_ids: [1, steps] FullKV-generated pseudo-answer tokens
      logits:    [1, steps, vocab] FullKV logits that predicted those tokens

    This trace uses no gold answer and is intended only as an online behavioral
    teacher for Gate-0-v2.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")

    cache = clone_cache(prompt_state.cache)
    logits = prompt_state.first_answer_logits
    token_ids = []
    logits_trace = []

    for _ in range(int(steps)):
        logits_trace.append(logits.detach())
        next_id = torch.argmax(logits, dim=-1)
        token_ids.append(next_id.detach())

        outputs = model(
            input_ids=next_id.unsqueeze(1),
            past_key_values=cache,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]

    return (
        torch.stack(token_ids, dim=1),
        torch.stack(logits_trace, dim=1),
    )


@torch.inference_mode()
def teacher_force_trace(
    model,
    prompt_state: PromptState,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Score a fixed pseudo-answer continuation from an arbitrary prompt state.

    token_ids are generated once by the FullKV teacher. LU/candidate states
    teacher-force exactly the same tokens, so behavioral metrics compare the
    predictive distributions on a common future-facing trajectory.
    """
    if token_ids.dim() != 2 or token_ids.shape[0] != 1:
        raise ValueError("token_ids must have shape [1, time]")
    if token_ids.shape[1] == 0:
        raise ValueError("token_ids must not be empty")

    token_ids = token_ids.to(prompt_state.first_answer_logits.device)
    first = prompt_state.first_answer_logits.unsqueeze(1)

    if token_ids.shape[1] == 1:
        return first

    cache = clone_cache(prompt_state.cache)
    outputs = model(
        input_ids=token_ids[:, :-1],
        past_key_values=cache,
        use_cache=True,
    )
    return torch.cat([first, outputs.logits], dim=1)
