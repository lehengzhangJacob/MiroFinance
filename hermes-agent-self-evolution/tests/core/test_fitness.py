"""Tests for mixed-language evolution fitness helpers."""

import dspy

from evolution.core.fitness import (
    _policy_requirements,
    _semantic_units,
    skill_fitness_metric,
)


def test_semantic_units_support_chinese_without_spaces():
    expected = _semantic_units("必须调用成长质量筛选并遵守公告日截断")
    output = _semantic_units("先调用成长质量筛选，再按公告日进行点时截断")

    assert "成长" in expected & output
    assert "筛选" in expected & output
    assert "公告" in expected & output


def test_semantic_units_support_english_words():
    units = _semantic_units("Use point_in_time screening and CASH controls")

    assert {"point_in_time", "screening", "cash"}.issubset(units)


def test_predictor_feedback_names_missing_process_requirements():
    example = dspy.Example(
        task_input="build a portfolio",
        expected_behavior=(
            "Use as_of and ashare_screen_market, then output CASH in boxed format"
        ),
    )
    prediction = dspy.Prediction(output="Choose three diversified stocks.")

    result = skill_fitness_metric(
        example,
        prediction,
        trace=None,
        pred_name="predictor.predict",
        pred_trace=None,
    )

    assert result.score < 1.0
    assert "ashare_screen_market" in result.feedback
    assert "as_of" in result.feedback
    assert "cash" in result.feedback


def test_full_evaluation_remains_numeric():
    example = dspy.Example(task_input="x", expected_behavior="Use as_of")
    prediction = dspy.Prediction(output="Use as_of")

    assert isinstance(skill_fitness_metric(example, prediction), float)


def test_policy_requirements_extract_financial_guardrails():
    requirements = dict(
        _policy_requirements("只从稳健候选选择，并确保稳定锚和一次补筛")
    )

    assert set(requirements) == {
        "robust-only",
        "stability-anchor",
        "single-fallback-screen",
    }


def test_policy_requirements_reduce_score_when_guardrail_is_missing():
    example = dspy.Example(
        expected_behavior="只从稳健候选选择，并确保稳定锚",
    )
    vague = dspy.Prediction(output="选择质量较好的候选")
    explicit = dspy.Prediction(output="只从稳健候选选择，并确保稳定锚")

    assert skill_fitness_metric(example, explicit) > skill_fitness_metric(example, vague)
