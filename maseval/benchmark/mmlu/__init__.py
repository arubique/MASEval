"""MMLU Benchmark for MASEval.

Implements MMLU evaluation with anchor point-based task selection for DISCO prediction.

Usage:
    from maseval.benchmark.mmlu import (
        MMLUBenchmark,
        MMLUEnvironment,
        MMLUEvaluator,
        load_tasks,
        AnchorPointsTaskQueue,
    )

    # Load tasks and anchor points
    tasks = load_tasks(
        data_path="path/to/mmlu_prompts_examples.json",
        anchor_points_path="path/to/anchor_points.pkl",  # Optional
    )

    # Create benchmark
    benchmark = MMLUBenchmark()
    results = benchmark.run(tasks=tasks, agent_data={"model_id": "gpt-4"})
"""

from .mmlu import (
    MMLUBenchmark,
    MMLUEnvironment,
    MMLUEvaluator,
    MMLUModelAgent,
    MMLUAgentAdapter,
    AnchorPointsTaskQueue,
    load_tasks,
    compute_benchmark_metrics,
)

__all__ = [
    "MMLUBenchmark",
    "MMLUEnvironment",
    "MMLUEvaluator",
    "MMLUModelAgent",
    "MMLUAgentAdapter",
    "AnchorPointsTaskQueue",
    "load_tasks",
    "compute_benchmark_metrics",
]
