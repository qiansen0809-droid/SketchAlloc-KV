# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re


def extract_answer(response):
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    else:
        match = re.search(r"The correct answer is ([A-D])", response)
        if match:
            return match.group(1)
        else:
            return None


# def calculate_metrics(df):
#     predictions = df["predicted_answer"].tolist()
#     answers = df["answer"].tolist()
#     lengths = df["length"].tolist()
#     difficulties = df["difficulty"].tolist()
#     return scorer(predictions, answers, lengths, difficulties)


# def scorer(predictions, answers, lengths, difficulties):
#     compensated = False
#     easy, hard, short, medium, long = 0, 0, 0, 0, 0
#     easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
#     for pred, answer, length, difficulty in zip(predictions, answers, lengths, difficulties):
#         acc = int(extract_answer(pred) == answer)
#         if compensated and pred["pred"] is None:
#             acc = 0.25  # type:ignore[assignment]
#         if difficulty == "easy":
#             easy += 1
#             easy_acc += acc
#         else:
#             hard += 1
#             hard_acc += acc

#         if length == "short":
#             short += 1
#             short_acc += acc
#         elif length == "medium":
#             medium += 1
#             medium_acc += acc
#         else:
#             long += 1
#             long_acc += acc
#     scores = ["Overall\tEasy\tHard\tShort\tMedium\tLong"]
#     scores.append(
#         str(round(100 * (easy_acc + hard_acc) / len(predictions), 1))
#         + "\t"
#         + str(round(100 * easy_acc / easy, 1))
#         + "\t"
#         + str(round(100 * hard_acc / hard, 1))
#         + "\t"
#         + str(round(100 * short_acc / short, 1))
#         + "\t"
#         + str(round(100 * medium_acc / medium, 1))
#         + "\t"
#         + str(round(100 * long_acc / long, 1))
#     )
#     return scores

import math

def _normalize_pred(pred):
    """
    Normalize a prediction entry into a string answer like 'A'/'B'/'C'/'D' or None.
    Handles cases where pred might be:
    - a plain string like "(A) The correct answer is (A)"
    - a list like ["(A) ...", "(B) ..."]
    - a dict like {"pred": "(A) ..."} or {"answer": "A"}
    - None
    """
    if pred is None:
        return None
    if isinstance(pred, (list, tuple)):
        if len(pred) == 0:
            return None
        pred_item = pred[0]
    elif isinstance(pred, dict):
        if "pred" in pred:
            pred_item = pred["pred"]
        elif "answer" in pred:
            pred_item = pred["answer"]
        else:
            pred_item = next(iter(pred.values())) if len(pred) > 0 else None
    else:
        pred_item = pred

    if pred_item is None:
        return None
    return extract_answer(str(pred_item))


def calculate_metrics(df):
    """
    Public scorer entry used by SCORER_REGISTRY.
    Returns a dict of metric_name -> value.
    """
    predictions = df["predicted_answer"].tolist()
    answers = df["answer"].tolist()
    lengths = df["length"].tolist()
    difficulties = df["difficulty"].tolist()
    return scorer(predictions, answers, lengths, difficulties)


def scorer(predictions, answers, lengths, difficulties):
    """
    Compute metrics and return a dict mapping metric names to numeric scores (floats).
    Example returned dict:
    {
      "Overall": 75.0,
      "Easy": 80.0,
      "Hard": 70.0,
      "Short": 78.0,
      "Medium": 74.0,
      "Long": 70.0
    }
    Values are percentages (0-100).
    """
    counts = {"easy": 0, "hard": 0, "short": 0, "medium": 0, "long": 0}
    correct = {"easy": 0.0, "hard": 0.0, "short": 0.0, "medium": 0.0, "long": 0.0}

    total = 0
    total_correct = 0.0

    for pred_raw, answer, length, difficulty in zip(predictions, answers, lengths, difficulties):
        pred = _normalize_pred(pred_raw)
        is_correct = 1 if (pred is not None and pred == answer) else 0
        total += 1
        total_correct += is_correct

        if difficulty == "easy":
            counts["easy"] += 1
            correct["easy"] += is_correct
        else:
            counts["hard"] += 1
            correct["hard"] += is_correct

        if length == "short":
            counts["short"] += 1
            correct["short"] += is_correct
        elif length == "medium":
            counts["medium"] += 1
            correct["medium"] += is_correct
        else:
            counts["long"] += 1
            correct["long"] += is_correct

    def pct(num, den):
        if den == 0:
            return float("nan")
        return round(100.0 * (num / den), 1)

    metrics = {
        "Overall": pct(total_correct, total),
        "Easy": pct(correct["easy"], counts["easy"]),
        "Hard": pct(correct["hard"], counts["hard"]),
        "Short": pct(correct["short"], counts["short"]),
        "Medium": pct(correct["medium"], counts["medium"]),
        "Long": pct(correct["long"], counts["long"]),
    }

    return metrics
