from __future__ import annotations

import re
import string
from collections import Counter

import torch
import torch.nn.functional as F


def _as_log_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.float(), dim=-1)


def probability_overlap(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Mean full-vocabulary probability overlap.

    overlap(p, q) = sum_v min(p_v, q_v), averaged over probe positions.
    Range: [0, 1], where 1 means identical distributions.
    """
    ref = F.softmax(reference_logits.float(), dim=-1)
    cand = F.softmax(candidate_logits.float(), dim=-1)
    return torch.minimum(ref, cand).sum(dim=-1).mean().clamp(0.0, 1.0)


def teacher_to_candidate_kl(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> torch.Tensor:
    """Mean KL(reference || candidate) across probe positions."""
    log_ref = _as_log_probs(reference_logits)
    log_cand = _as_log_probs(candidate_logits)
    ref = log_ref.exp()
    return (ref * (log_ref - log_cand)).sum(dim=-1).mean()


def jensen_shannon(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> torch.Tensor:
    """Mean Jensen-Shannon divergence across probe positions."""
    log_ref = _as_log_probs(reference_logits)
    log_cand = _as_log_probs(candidate_logits)
    ref = log_ref.exp()
    cand = log_cand.exp()
    mixture = 0.5 * (ref + cand)
    log_mix = torch.log(mixture.clamp_min(1e-12))
    kl_ref = (ref * (log_ref - log_mix)).sum(dim=-1)
    kl_cand = (cand * (log_cand - log_mix)).sum(dim=-1)
    return 0.5 * (kl_ref + kl_cand).mean()


def topk_agreement(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    k: int = 10,
) -> torch.Tensor:
    """
    Fraction of reference top-k tokens also present in candidate top-k, averaged
    over probe positions.
    """
    ref_top = reference_logits.topk(k, dim=-1).indices
    cand_top = candidate_logits.topk(k, dim=-1).indices
    matches = (ref_top.unsqueeze(-1) == cand_top.unsqueeze(-2)).any(dim=-1)
    return matches.float().mean()


def continuation_nll(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Mean teacher-forced token NLL.

    logits shape: [batch, time, vocab] or [time, vocab]
    target_ids shape: [batch, time] or [time]
    """
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    if target_ids.dim() == 1:
        target_ids = target_ids.unsqueeze(0)

    if logits.shape[:-1] != target_ids.shape:
        raise ValueError(
            f"logits/targets mismatch: {logits.shape[:-1]} vs {target_ids.shape}"
        )

    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        target_ids.reshape(-1),
        reduction="mean",
    )



def _normalize_english_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    """LongBench-style English QA token F1."""
    pred_tokens = _normalize_english_answer(prediction).split()
    gold_tokens = _normalize_english_answer(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0

    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return float(2 * precision * recall / (precision + recall))


def max_qa_f1_score(prediction: str, ground_truths: list[str]) -> float:
    if not ground_truths:
        return 0.0
    return max(qa_f1_score(prediction, gt) for gt in ground_truths)


def ruler_string_match_score(prediction: str, references: list[str]) -> float:
    """
    RULER retrieval metric used for non-QA RULER tasks: fraction of required
    reference strings that occur in the prediction, case-insensitively.
    """
    if not references:
        return 0.0
    pred = prediction.strip().lower()
    return float(
        sum(1.0 if str(ref).lower() in pred else 0.0 for ref in references)
        / len(references)
    )


def gate0_task_score(family: str, prediction: str, references: list[str]) -> float:
    if family == "ruler_retrieval":
        return ruler_string_match_score(prediction, references)
    if family in {"longbench_single", "longbench_multi"}:
        return max_qa_f1_score(prediction, references)
    raise ValueError(f"unsupported Gate-0 family: {family}")
