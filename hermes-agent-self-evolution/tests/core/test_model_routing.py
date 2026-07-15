"""Tests for explicit OpenAI-compatible provider routing."""

from unittest.mock import patch

import pytest

from evolution.core.config import make_dspy_lm


@pytest.mark.parametrize(
    ("model", "prefix", "extra_kwargs"),
    [
        (
            "openai/glm-5.2",
            "GLM",
            {"extra_body": {"thinking": {"type": "disabled"}}},
        ),
        ("openai/deepseek-v4-pro", "DEEPSEEK", {}),
        ("openai/kimi-k2.5", "KIMI", {}),
        ("openai/gpt-4.1", "OPENAI", {}),
    ],
)
def test_make_dspy_lm_uses_model_specific_credentials(
    monkeypatch, model, prefix, extra_kwargs
):
    for name in ("GLM", "DEEPSEEK", "KIMI", "OPENAI"):
        monkeypatch.setenv(f"{name}_API_KEY", f"{name.lower()}-key")
        monkeypatch.setenv(f"{name}_BASE_URL", f"https://{name.lower()}.example/v1")

    with patch("evolution.core.config.dspy.LM") as lm:
        make_dspy_lm(model, max_tokens=4096)

    lm.assert_called_once_with(
        model,
        max_tokens=4096,
        api_key=f"{prefix.lower()}-key",
        api_base=f"https://{prefix.lower()}.example/v1",
        **extra_kwargs,
    )
