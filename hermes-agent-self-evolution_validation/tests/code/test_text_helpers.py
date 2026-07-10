"""Tests for text_helpers — baseline seed has intentional bugs."""

import sys
from pathlib import Path

# hermes-agent/code on path
_CODE = Path(__file__).resolve().parents[2] / "hermes-agent" / "code"
sys.path.insert(0, str(_CODE))

from text_helpers import slugify, truncate_subject  # noqa: E402


def test_slugify_basic():
    assert slugify("Hello World") == "hello-world"


def test_slugify_collapses_multiple_hyphens():
    assert slugify("foo--bar") == "foo-bar"
    assert slugify("a   b") == "a-b"


def test_slugify_strips_illegal_chars():
    assert slugify("Hello! @World#") == "hello-world"


def test_truncate_subject_short():
    assert truncate_subject("short line") == "short line"


def test_truncate_subject_exactly_72():
    line = "x" * 72
    assert len(truncate_subject(line)) == 72


def test_truncate_subject_over_72():
    line = "x" * 80
    assert len(truncate_subject(line)) <= 72
