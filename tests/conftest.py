"""Shared fixtures for the drift-sentinel test suite."""

from __future__ import annotations

import math
import random

import pytest


@pytest.fixture
def rng() -> random.Random:
    """A seeded RNG so every test run produces identical numbers."""
    return random.Random(20260101)


@pytest.fixture
def normal_reference() -> list:
    """A stable numeric sample, mean 0, sd 1."""
    generator = random.Random(1)
    return [generator.gauss(0.0, 1.0) for _ in range(600)]


@pytest.fixture
def shifted_current() -> list:
    """The same distribution shifted by 1.0 standard deviation."""
    generator = random.Random(2)
    return [generator.gauss(1.0, 1.0) for _ in range(600)]


@pytest.fixture
def identical_sample() -> list:
    generator = random.Random(3)
    return [generator.gauss(0.0, 1.0) for _ in range(600)]


@pytest.fixture
def categorical_reference() -> list:
    generator = random.Random(4)
    return [str(generator.choice(["a", "b", "c"])) for _ in range(300)]


@pytest.fixture
def llm_entries() -> list:
    """A batch of healthy LLM call records."""
    generator = random.Random(5)
    entries = []
    for _ in range(40):
        entries.append(
            {
                "model": "test-model",
                "prompt_tokens": generator.randint(100, 800),
                "completion_tokens": generator.randint(20, 200),
                "latency_ms": round(generator.gauss(700.0, 90.0), 1),
                "response": "The runbook says to restart the worker first.",
            }
        )
    return entries


@pytest.fixture
def price_table() -> dict:
    return {
        "test-model": {"prompt_per_1k": 0.003, "completion_per_1k": 0.015}
    }
