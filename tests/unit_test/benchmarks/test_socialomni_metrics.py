# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from benchmarks.metrics.socialomni import (
    JudgeCompletenessError,
    bootstrap_mean_interval,
    compute_socialomni_level1_metrics,
    compute_socialomni_level2_metrics,
    wilson_interval,
)

JUDGES = ("gpt-4o", "gemini-2.5-pro", "qwen3-omni")


def _scores(first: int, second: int, third: int) -> dict[str, int]:
    return dict(zip(JUDGES, (first, second, third)))


def test_level1_uses_fixed_denominator_and_reports_strata_gap() -> None:
    records = [
        {
            "sample_id": "visible-a",
            "gold_answer": "A",
            "predicted_answer": "A",
            "visibility": "speaker_visible",
            "is_success": True,
        },
        {
            "sample_id": "visible-b",
            "gold_answer": "B",
            "predicted_answer": None,
            "visibility": "speaker_visible",
            "is_success": True,
        },
        {
            "sample_id": "mismatch-c",
            "gold_answer": 2,
            "predicted_answer": 2,
            "visibility": "visibility_mismatch",
            "is_success": True,
        },
        {
            "sample_id": "mismatch-d",
            "gold_answer": 3,
            "predicted_answer": 3,
            "visibility": "visibility_mismatch",
            "is_success": False,
        },
    ]

    result = compute_socialomni_level1_metrics(records)

    assert result["total_samples"] == 4
    assert result["correct"] == 2
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["macro_f1"] == pytest.approx(0.5)
    assert result["unparseable_or_failed"] == 2
    assert result["strata"]["speaker_visible"]["accuracy"] == pytest.approx(0.5)
    assert result["strata"]["visibility_mismatch"]["accuracy"] == pytest.approx(0.5)
    assert result["visible_minus_mismatch_accuracy"] == pytest.approx(0.0)


def test_level1_macro_f1_always_includes_all_four_answer_positions() -> None:
    result = compute_socialomni_level1_metrics(
        [
            {
                "sample_id": "only-a",
                "gold_answer": "A",
                "predicted_answer": "A",
                "visibility": "speaker_visible",
                "is_success": True,
            }
        ]
    )

    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == pytest.approx(0.25)
    assert set(result["per_class"]) == {"0", "1", "2", "3"}


def test_level2_computes_decision_and_quality_metrics() -> None:
    records = [
        {
            "sample_id": "yes-covered",
            "gold_when": "YES",
            "predicted_when": "YES",
            "gold_response": "Hello",
            "gold_judge_scores": _scores(100, 75, 50),
        },
        {
            "sample_id": "yes-empty",
            "gold_when": "YES",
            "predicted_when": "YES",
            "gold_response": "  ",
            "gold_judge_scores": None,
        },
        {
            "sample_id": "yes-declined",
            "gold_when": "YES",
            "predicted_when": "NO",
            "gold_response": "A forced response",
            "gold_judge_scores": _scores(25, 50, 75),
        },
        {
            "sample_id": "no-correct",
            "gold_when": "NO",
            "predicted_when": "NO",
            "gold_response": None,
            "gold_judge_scores": None,
        },
        {
            "sample_id": "no-false-positive",
            "gold_when": "NO",
            "predicted_when": "YES",
            "gold_response": "Ignored for quality",
            "gold_judge_scores": None,
        },
        {
            "sample_id": "no-unparseable",
            "gold_when": "NO",
            "predicted_when": None,
            "gold_response": None,
            "gold_judge_scores": None,
        },
    ]

    result = compute_socialomni_level2_metrics(
        records, judge_names=JUDGES, bootstrap_seed=7, bootstrap_samples=200
    )

    when = result["when"]
    assert when["accuracy"] == pytest.approx(3 / 6)
    assert when["positive_precision"] == pytest.approx(2 / 3)
    assert when["positive_recall"] == pytest.approx(2 / 3)
    assert when["positive_f1"] == pytest.approx(2 / 3)
    # The unparseable gold-NO prediction remains a false negative for the NO
    # class, so the negative-class F1 is 0.4 rather than being dropped.
    assert when["macro_f1"] == pytest.approx((2 / 3 + 0.4) / 2)
    assert when["unparseable_or_failed"] == 1

    quality = result["quality"]
    assert quality["gold_positive_samples"] == 3
    assert quality["covered_positive_samples"] == 1
    assert quality["qgold"] == pytest.approx((75 + 0 + 50) / 3)
    assert quality["qens"] == pytest.approx(75)
    assert quality["cov_plus"] == pytest.approx(1 / 3)
    assert quality["qens_joint"] == pytest.approx(25)
    assert quality["empty_or_failed_positive_responses"] == 1
    assert result["per_judge"]["gpt-4o"]["qgold"] == pytest.approx(125 / 3)
    assert result["leave_one_out"]["gpt-4o"]["qens"] == pytest.approx(62.5)


@pytest.mark.parametrize(
    "scores",
    [
        {"gpt-4o": 100, "gemini-2.5-pro": 75},
        {"gpt-4o": 100, "gemini-2.5-pro": 75, "qwen3-omni": 80},
        {
            "gpt-4o": 100,
            "gemini-2.5-pro": 75,
            "qwen3-omni": 50,
            "unexpected": 25,
        },
    ],
)
def test_level2_rejects_incomplete_or_invalid_judge_scores(
    scores: dict[str, int],
) -> None:
    with pytest.raises(JudgeCompletenessError, match="sample-yes"):
        compute_socialomni_level2_metrics(
            [
                {
                    "sample_id": "sample-yes",
                    "gold_when": "YES",
                    "predicted_when": "YES",
                    "gold_response": "non-empty",
                    "gold_judge_scores": scores,
                }
            ],
            judge_names=JUDGES,
            bootstrap_samples=10,
        )


def test_level2_empty_or_failed_response_is_zero_without_requiring_judges() -> None:
    result = compute_socialomni_level2_metrics(
        [
            {
                "sample_id": "failed",
                "gold_when": True,
                "predicted_when": True,
                "gold_response": "partial output",
                "gold_judge_scores": _scores(100, 100, 100),
                "is_success": False,
            }
        ],
        judge_names=JUDGES,
        bootstrap_samples=10,
    )

    assert result["quality"]["qgold"] == 0
    assert result["quality"]["qens"] is None
    assert result["quality"]["cov_plus"] == 0
    assert result["quality"]["qens_joint"] == 0


def test_level2_uses_gold_response_success_field() -> None:
    result = compute_socialomni_level2_metrics(
        [
            {
                "sample_id": "failed",
                "gold_when": "YES",
                "predicted_when": "YES",
                "gold_response": "partial output",
                "gold_response_success": False,
                "gold_judge_scores": _scores(100, 100, 100),
            }
        ],
        judge_names=JUDGES,
        bootstrap_samples=10,
    )

    assert result["quality"]["qgold"] == 0
    assert result["quality"]["covered_positive_samples"] == 0


def test_level2_reports_pairwise_and_family_removal() -> None:
    result = compute_socialomni_level2_metrics(
        [
            {
                "sample_id": "one",
                "gold_when": True,
                "predicted_when": True,
                "gold_response": "one",
                "gold_judge_scores": _scores(0, 25, 100),
            },
            {
                "sample_id": "two",
                "gold_when": True,
                "predicted_when": True,
                "gold_response": "two",
                "gold_judge_scores": _scores(25, 50, 75),
            },
            {
                "sample_id": "three",
                "gold_when": True,
                "predicted_when": True,
                "gold_response": "three",
                "gold_judge_scores": _scores(100, 75, 0),
            },
        ],
        judge_names=JUDGES,
        judge_model_families={
            "gpt-4o": "openai",
            "gemini-2.5-pro": "google",
            "qwen3-omni": "qwen",
        },
        bootstrap_samples=20,
    )

    agreement = result["pairwise_agreement"]["gpt-4o__vs__gemini-2.5-pro"]
    assert agreement["samples"] == 3
    assert agreement["spearman"] == pytest.approx(1.0)
    assert agreement["mae"] == pytest.approx(25.0)
    removed = result["family_removal"]["qwen"]
    assert removed["removed_judges"] == ["qwen3-omni"]
    assert removed["remaining_judges"] == ["gpt-4o", "gemini-2.5-pro"]
    assert removed["metrics"]["qgold"] == pytest.approx(275 / 6)


def test_confidence_intervals_are_bounded_and_bootstrap_is_deterministic() -> None:
    assert wilson_interval(0, 0) == {"low": 0.0, "high": 0.0}
    interval = wilson_interval(5, 10)
    assert 0 < interval["low"] < 0.5 < interval["high"] < 1

    first = bootstrap_mean_interval([0.0, 50.0, 100.0], seed=42, samples=100)
    second = bootstrap_mean_interval([0.0, 50.0, 100.0], seed=42, samples=100)
    assert first == second
    assert first is not None
    assert 0 <= first["low"] <= first["high"] <= 100
