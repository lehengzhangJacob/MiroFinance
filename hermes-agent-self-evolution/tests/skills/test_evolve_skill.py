"""Tests for the public Hermes skill-evolution budget."""

import pytest

from evolution.skills.evolve_skill import gepa_budget


def test_iterations_map_to_full_gepa_evaluations():
    assert gepa_budget(3) == {"max_full_evals": 3}


def test_iteration_budget_must_be_positive():
    with pytest.raises(ValueError):
        gepa_budget(0)
