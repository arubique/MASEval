"""Structural tests for MMLU benchmark (Tier 1: offline, no real data).

Validates that the MMLU benchmark module exposes the expected API.
"""

import pytest

pytestmark = pytest.mark.benchmark


def test_mmlu_module_exposes_load_tasks_and_compute_metrics():
    """MMLU benchmark module exposes load_tasks and compute_benchmark_metrics."""
    pytest.importorskip("torch")
    from maseval.benchmark.mmlu import load_tasks, compute_benchmark_metrics

    assert callable(load_tasks)
    assert callable(compute_benchmark_metrics)
