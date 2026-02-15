"""Integration test for MMLU benchmark with DISCO prediction.

Replicates the "Run MMLU Benchmark (Zephyr-7B SFT Disco)" launch configuration:
- model_id=alignment-handbook/zephyr-7b-sft-full
- data_path=arubique/flattened-MMLU
- device=cuda:0
- disco_model_path=arubique/DISCO-MMLU

Asserts that model 0 DISCO predicted accuracy equals the known-good value 0.606739.

Requires: CUDA, network (HuggingFace), disco extra. Marked live + slow + benchmark + mmlu.

Run with::

    pytest -m "mmlu and live and slow" tests/test_benchmarks/test_mmlu/test_integration.py -v
"""

import json
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.slow,
    pytest.mark.benchmark,
    pytest.mark.mmlu,
]

# Known-good DISCO predicted accuracy for model 0 (Zephyr-7B SFT) from launch config
EXPECTED_MODEL_0_PREDICTED_ACCURACY = 0.606739


def _ensure_examples_importable():
    """Ensure repo root is on path so examples.mmlu_benchmark is importable."""
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def test_mmlu_zephyr_disco_predicted_accuracy(tmp_path):
    """Running MMLU benchmark with Zephyr-7B SFT and DISCO yields model 0 accuracy 0.606739."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for MMLU DISCO integration test")

    _ensure_examples_importable()
    from examples.mmlu_benchmark.mmlu_benchmark import main

    output_dir = tmp_path / "mmlu_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "mmlu_benchmark.py",
        "--model_id",
        "alignment-handbook/zephyr-7b-sft-full",
        "--data_path",
        "arubique/flattened-MMLU",
        "--device",
        "cuda:0",
        "--disco_model_path",
        "arubique/DISCO-MMLU",
        "--output_dir",
        str(output_dir),
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        main()
    finally:
        sys.argv = old_argv

    summary_path = output_dir / "summary.json"
    assert summary_path.exists(), f"Expected summary at {summary_path}"
    with open(summary_path) as f:
        summary = json.load(f)

    assert "disco_prediction" in summary, "DISCO prediction should be present in summary"
    predicted = summary["disco_prediction"]["predicted_accuracy"]
    assert predicted == pytest.approx(EXPECTED_MODEL_0_PREDICTED_ACCURACY, rel=1e-5, abs=1e-6), (
        f"Model 0 predicted accuracy: expected {EXPECTED_MODEL_0_PREDICTED_ACCURACY}, got {predicted}"
    )
