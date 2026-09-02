# SPDX-License-Identifier: Apache-2.0
"""Pure scoring functions for the SocialOmni benchmark.

The functions in this module deliberately treat unparseable model decisions as
wrong answers while rejecting incomplete judge data. This keeps model failures
in the benchmark denominator without silently changing a three-judge result
into a two-judge result.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from typing import Any

SOCIALOMNI_JUDGE_NAMES = ("gpt-4o", "gemini-2.5-pro", "qwen3-omni")
SOCIALOMNI_SCORE_BUCKETS = frozenset({0, 25, 50, 75, 100})
_LEVEL1_LABELS = (0, 1, 2, 3)
_VISIBILITY_LABELS = ("speaker_visible", "visibility_mismatch")


class JudgeCompletenessError(ValueError):
    """Raised when a judge-required response lacks exactly three valid scores."""


def _field(record: object, name: str, default: Any = ...) -> Any:
    if isinstance(record, Mapping):
        if name in record:
            return record[name]
    elif hasattr(record, name):
        return getattr(record, name)
    if default is ...:
        raise ValueError(f"record is missing required field {name!r}")
    return default


def _sample_id(record: object, index: int) -> str:
    return str(_field(record, "sample_id", f"index-{index}"))


def _normalize_level1_answer(value: object, *, gold: bool) -> int | None:
    if isinstance(value, bool):
        normalized = None
    elif isinstance(value, int) and value in _LEVEL1_LABELS:
        normalized = value
    elif isinstance(value, str) and value.strip().upper() in ("A", "B", "C", "D"):
        normalized = ord(value.strip().upper()) - ord("A")
    else:
        normalized = None
    if gold and normalized is None:
        raise ValueError(f"invalid Level 1 gold answer: {value!r}")
    return normalized


def _normalize_when(value: object, *, gold: bool) -> bool | None:
    if isinstance(value, bool):
        normalized = value
    elif isinstance(value, str) and value.strip().upper() in {"YES", "NO"}:
        normalized = value.strip().upper() == "YES"
    else:
        normalized = None
    if gold and normalized is None:
        raise ValueError(f"invalid Level 2 gold decision: {value!r}")
    return normalized


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> dict[str, float]:
    """Return a two-sided Wilson 95% confidence interval for a proportion."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total")
    if total == 0:
        return {"low": 0.0, "high": 0.0}
    proportion = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (proportion + z2 / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z2 / (4 * total**2))
        / denominator
    )
    return {"low": max(0.0, center - radius), "high": min(1.0, center + radius)}


def _classification_metrics(
    gold: Sequence[int],
    predicted: Sequence[int | None],
    labels: Sequence[int],
) -> dict[str, Any]:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted sequences must have the same length")

    correct = sum(expected == actual for expected, actual in zip(gold, predicted))
    per_class: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []
    for label in labels:
        true_positive = sum(g == label and p == label for g, p in zip(gold, predicted))
        false_positive = sum(g != label and p == label for g, p in zip(gold, predicted))
        false_negative = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = _safe_ratio(true_positive, precision_denominator)
        recall = _safe_ratio(true_positive, recall_denominator)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        f1_values.append(f1)
        per_class[str(label)] = {
            "support": recall_denominator,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "precision_ci95": wilson_interval(true_positive, precision_denominator),
            "recall": recall,
            "recall_ci95": wilson_interval(true_positive, recall_denominator),
            "f1": f1,
        }

    return {
        "total_samples": len(gold),
        "correct": correct,
        "accuracy": _safe_ratio(correct, len(gold)),
        "accuracy_ci95": wilson_interval(correct, len(gold)),
        "macro_f1": _safe_ratio(sum(f1_values), len(labels)),
        "per_class": per_class,
    }


def compute_socialomni_level1_metrics(records: Iterable[object]) -> dict[str, Any]:
    """Compute Level 1 four-choice metrics with fixed sample denominators."""
    materialized = list(records)
    gold: list[int] = []
    predicted: list[int | None] = []
    visibility: list[str] = []

    for index, record in enumerate(materialized):
        sample_id = _sample_id(record, index)
        gold.append(_normalize_level1_answer(_field(record, "gold_answer"), gold=True))
        current_visibility = str(_field(record, "visibility"))
        if current_visibility not in _VISIBILITY_LABELS:
            raise ValueError(
                f"sample {sample_id!r} has invalid visibility {current_visibility!r}; "
                f"expected one of {_VISIBILITY_LABELS}"
            )
        visibility.append(current_visibility)
        if not bool(_field(record, "is_success", True)):
            predicted.append(None)
        else:
            predicted.append(
                _normalize_level1_answer(
                    _field(record, "predicted_answer", None), gold=False
                )
            )

    result = _classification_metrics(gold, predicted, _LEVEL1_LABELS)
    result["unparseable_or_failed"] = sum(value is None for value in predicted)
    strata: dict[str, dict[str, Any]] = {}
    for name in _VISIBILITY_LABELS:
        indices = [i for i, value in enumerate(visibility) if value == name]
        stratum_gold = [gold[i] for i in indices]
        stratum_predictions = [predicted[i] for i in indices]
        stratum = _classification_metrics(
            stratum_gold, stratum_predictions, _LEVEL1_LABELS
        )
        strata[name] = {
            "total_samples": stratum["total_samples"],
            "correct": stratum["correct"],
            "accuracy": stratum["accuracy"],
            "accuracy_ci95": stratum["accuracy_ci95"],
            "macro_f1": stratum["macro_f1"],
        }
    result["strata"] = strata
    result["visible_minus_mismatch_accuracy"] = (
        strata["speaker_visible"]["accuracy"]
        - strata["visibility_mismatch"]["accuracy"]
    )
    return result


def _validate_judges(judge_names: Sequence[str]) -> tuple[str, str, str]:
    normalized = tuple(judge_names)
    if len(normalized) != 3 or len(set(normalized)) != 3:
        raise ValueError("SocialOmni requires exactly three distinct judge names")
    if any(not name for name in normalized):
        raise ValueError("judge names must be non-empty")
    return normalized  # type: ignore[return-value]


def _validated_scores(
    raw_scores: object,
    *,
    judges: tuple[str, str, str],
    sample_id: str,
) -> dict[str, int]:
    if not isinstance(raw_scores, Mapping):
        raise JudgeCompletenessError(
            f"sample {sample_id!r} requires scores from all judges {judges}"
        )
    missing = [judge for judge in judges if judge not in raw_scores]
    extra = [str(name) for name in raw_scores if name not in judges]
    if missing or extra:
        raise JudgeCompletenessError(
            f"sample {sample_id!r} has incomplete judge scores; "
            f"missing={missing}, extra={extra}"
        )
    validated: dict[str, int] = {}
    for judge in judges:
        score = raw_scores[judge]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise JudgeCompletenessError(
                f"sample {sample_id!r} judge {judge!r} returned non-numeric score "
                f"{score!r}"
            )
        if (
            not math.isfinite(float(score))
            or float(score) not in SOCIALOMNI_SCORE_BUCKETS
        ):
            raise JudgeCompletenessError(
                f"sample {sample_id!r} judge {judge!r} returned {score!r}; "
                f"allowed scores are {sorted(SOCIALOMNI_SCORE_BUCKETS)}"
            )
        validated[judge] = int(score)
    return validated


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    seed: int,
    samples: int,
) -> dict[str, float] | None:
    """Return a deterministic percentile-bootstrap 95% interval for a mean."""
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not values:
        return None
    generator = random.Random(seed)
    size = len(values)
    means = [
        sum(values[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    ]
    return {"low": _quantile(means, 0.025), "high": _quantile(means, 0.975)}


def _average(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _ranks(values: Sequence[float]) -> list[float]:
    ranked = sorted(enumerate(values), key=lambda pair: pair[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][1] == ranked[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for index in range(start, end):
            result[ranked[index][0]] = average_rank
        start = end
    return result


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def _judge_subset_metrics(
    quality_rows: Sequence[dict[str, Any]],
    judges: Sequence[str],
    positive_total: int,
) -> dict[str, float | int | None]:
    qgold_values = [
        (
            0.0
            if row["scores"] is None
            else sum(row["scores"][judge] for judge in judges) / len(judges)
        )
        for row in quality_rows
    ]
    qens_values = [
        sum(row["scores"][judge] for judge in judges) / len(judges)
        for row in quality_rows
        if row["covered"]
    ]
    covered = len(qens_values)
    qens = _average(qens_values)
    coverage = _safe_ratio(covered, positive_total)
    return {
        "judge_count": len(judges),
        "qgold": _average(qgold_values) or 0.0,
        "qens": qens,
        "cov_plus": coverage,
        "qens_joint": coverage * qens if qens is not None else 0.0,
    }


def compute_socialomni_level2_metrics(
    records: Iterable[object],
    *,
    judge_names: Sequence[str] = SOCIALOMNI_JUDGE_NAMES,
    judge_model_families: Mapping[str, str] | None = None,
    bootstrap_seed: int = 20260902,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Compute SocialOmni Level 2 decision and three-judge quality metrics.

    ``QGold`` uses every gold-positive sample. A failed or empty response scores
    zero. A non-empty response must have all three valid judge scores, otherwise
    :class:`JudgeCompletenessError` is raised and no formal result is returned.
    ``QEns`` is conditional on a correct YES decision and a non-empty response;
    ``Cov+`` retains the fixed gold-positive denominator.
    """
    judges = _validate_judges(judge_names)
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if judge_model_families is not None:
        unknown = set(judge_model_families) - set(judges)
        missing = set(judges) - set(judge_model_families)
        if unknown or missing:
            raise ValueError(
                "judge_model_families must map every configured judge exactly once; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )

    materialized = list(records)
    gold: list[int] = []
    predicted: list[int | None] = []
    quality_rows: list[dict[str, Any]] = []
    for index, record in enumerate(materialized):
        sample_id = _sample_id(record, index)
        expected = _normalize_when(_field(record, "gold_when"), gold=True)
        actual = _normalize_when(_field(record, "predicted_when", None), gold=False)
        gold.append(int(bool(expected)))
        predicted.append(None if actual is None else int(actual))
        if not expected:
            continue

        response = _field(record, "gold_response", None)
        response_success = bool(
            _field(
                record,
                "gold_response_success",
                _field(record, "is_success", True),
            )
        )
        has_response = (
            response_success and isinstance(response, str) and bool(response.strip())
        )
        scores = None
        if has_response:
            scores = _validated_scores(
                _field(record, "gold_judge_scores", None),
                judges=judges,
                sample_id=sample_id,
            )
        quality_rows.append(
            {
                "sample_id": sample_id,
                "scores": scores,
                "covered": bool(actual) and has_response,
            }
        )

    classification = _classification_metrics(gold, predicted, (0, 1))
    positive = classification["per_class"]["1"]
    negative = classification["per_class"]["0"]
    decision = {
        "total_samples": classification["total_samples"],
        "correct": classification["correct"],
        "accuracy": classification["accuracy"],
        "accuracy_ci95": classification["accuracy_ci95"],
        "unparseable_or_failed": sum(value is None for value in predicted),
        "positive_precision": positive["precision"],
        "positive_precision_ci95": positive["precision_ci95"],
        "positive_recall": positive["recall"],
        "positive_recall_ci95": positive["recall_ci95"],
        "positive_f1": positive["f1"],
        "macro_f1": classification["macro_f1"],
        "confusion": {
            "true_positive": positive["true_positive"],
            "false_positive": positive["false_positive"],
            "false_negative": positive["false_negative"],
            "true_negative": negative["true_positive"],
        },
    }

    positive_total = len(quality_rows)
    quality = _judge_subset_metrics(quality_rows, judges, positive_total)
    qgold_values = [
        0.0 if row["scores"] is None else sum(row["scores"].values()) / len(judges)
        for row in quality_rows
    ]
    qens_values = [
        sum(row["scores"].values()) / len(judges)
        for row in quality_rows
        if row["covered"]
    ]
    joint_values = [
        sum(row["scores"].values()) / len(judges) if row["covered"] else 0.0
        for row in quality_rows
    ]
    covered = len(qens_values)
    quality.update(
        {
            "gold_positive_samples": positive_total,
            "covered_positive_samples": covered,
            "empty_or_failed_positive_responses": sum(
                row["scores"] is None for row in quality_rows
            ),
            "cov_plus_ci95": wilson_interval(covered, positive_total),
            "qgold_ci95": bootstrap_mean_interval(
                qgold_values, seed=bootstrap_seed, samples=bootstrap_samples
            ),
            "qens_ci95": bootstrap_mean_interval(
                qens_values, seed=bootstrap_seed + 1, samples=bootstrap_samples
            ),
            "qens_joint_ci95": bootstrap_mean_interval(
                joint_values, seed=bootstrap_seed + 2, samples=bootstrap_samples
            ),
        }
    )

    judged_rows = [row for row in quality_rows if row["scores"] is not None]
    per_judge: dict[str, dict[str, float | None]] = {}
    for judge in judges:
        judge_qgold = [
            0.0 if row["scores"] is None else float(row["scores"][judge])
            for row in quality_rows
        ]
        judge_qens = [
            float(row["scores"][judge]) for row in quality_rows if row["covered"]
        ]
        per_judge[judge] = {
            "qgold": _average(judge_qgold) or 0.0,
            "qens": _average(judge_qens),
        }

    leave_one_out = {
        omitted: _judge_subset_metrics(
            quality_rows,
            [judge for judge in judges if judge != omitted],
            positive_total,
        )
        for omitted in judges
    }

    pairwise: dict[str, dict[str, float | int | None]] = {}
    for left, right in combinations(judges, 2):
        left_scores = [float(row["scores"][left]) for row in judged_rows]
        right_scores = [float(row["scores"][right]) for row in judged_rows]
        pairwise[f"{left}__vs__{right}"] = {
            "samples": len(judged_rows),
            "spearman": _pearson(_ranks(left_scores), _ranks(right_scores)),
            "mae": _average([abs(a - b) for a, b in zip(left_scores, right_scores)]),
        }

    family_removal: dict[str, dict[str, Any]] = {}
    if judge_model_families is not None:
        for family in sorted(set(judge_model_families.values())):
            remaining = [
                judge for judge in judges if judge_model_families[judge] != family
            ]
            family_removal[family] = {
                "removed_judges": [
                    judge for judge in judges if judge_model_families[judge] == family
                ],
                "remaining_judges": remaining,
                "metrics": (
                    _judge_subset_metrics(quality_rows, remaining, positive_total)
                    if remaining
                    else None
                ),
            }

    return {
        "complete": True,
        "judge_names": list(judges),
        "score_buckets": sorted(SOCIALOMNI_SCORE_BUCKETS),
        "bootstrap": {"seed": bootstrap_seed, "samples": bootstrap_samples},
        "when": decision,
        "quality": quality,
        "per_judge": per_judge,
        "leave_one_out": leave_one_out,
        "pairwise_agreement": pairwise,
        "family_removal": family_removal,
    }
