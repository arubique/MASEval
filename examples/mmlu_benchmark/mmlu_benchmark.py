"""
MMLU Benchmark Example - Evaluating Models on Anchor Points

This example demonstrates how to evaluate a HuggingFace model on MMLU tasks
using anchor point-based task selection for DISCO prediction.

Equivalent to the disco-public command:
    python scripts/run_lm_eval.py \\
        --anchor_points_path=/path/to/anchor_points_disagreement.pkl \\
        --model=hf \\
        --model_args=pretrained=alignment-handbook/zephyr-7b-sft-full,trust_remote_code=True \\
        --tasks=mmlu_prompts \\
        --skip_non_anchor_points \\
        --use_full_prompt

Usage:
    # Run with default settings (evaluates on all tasks)
    python mmlu_benchmark.py --model_id "meta-llama/Llama-2-7b-hf" --data_path /path/to/mmlu_prompts_examples.json

    # Run with anchor points filtering (for DISCO prediction)
    python mmlu_benchmark.py \\
        --model_id "alignment-handbook/zephyr-7b-sft-full" \\
        --data_path /path/to/mmlu_prompts_examples.json \\
        --anchor_points_path /path/to/anchor_points_disagreement.pkl \\
        --use_full_prompt

    # Run on a subset of tasks for testing
    python mmlu_benchmark.py \\
        --model_id "meta-llama/Llama-2-7b-hf" \\
        --data_path /path/to/mmlu_prompts_examples.json \\
        --limit 10
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

# MASEval imports
from maseval.core.callbacks.result_logger import FileResultLogger

# MMLU benchmark imports
from maseval.benchmark.mmlu import (
    MMLUBenchmark,
    load_tasks,
    compute_benchmark_metrics,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MMLU Benchmark - Evaluate models on MMLU multiple choice questions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace model identifier (e.g., 'meta-llama/Llama-2-7b-hf')",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to MMLU prompts JSON file (mmlu_prompts_examples.json format)",
    )

    # Optional arguments
    parser.add_argument(
        "--anchor_points_path",
        type=str,
        default=None,
        help="Path to anchor points pickle file. If provided, evaluates only anchor tasks.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        default=None,
        help="Path to save predictions tensor as pickle (for DISCO predictor)",
    )
    parser.add_argument(
        "--use_full_prompt",
        action="store_true",
        help="Use full prompt with few-shot examples instead of just the query",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to evaluate (for testing)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation (currently not implemented, reserved for future)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run model on (e.g., 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading model",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for task execution",
    )

    return parser.parse_args()


class HuggingFaceMMLUBenchmark(MMLUBenchmark):
    """MMLU Benchmark using HuggingFace transformers models.

    This concrete implementation loads a HuggingFace model via the
    transformers pipeline and uses it for MCQ evaluation.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda:0",
        trust_remote_code: bool = False,
        use_full_prompt: bool = False,
        **kwargs,
    ):
        """Initialize HuggingFace MMLU benchmark.

        Args:
            model_id: HuggingFace model identifier.
            device: Device to run model on.
            trust_remote_code: Trust remote code when loading model.
            use_full_prompt: Use full prompt with few-shot examples.
            **kwargs: Additional arguments passed to MMLUBenchmark.
        """
        super().__init__(use_full_prompt=use_full_prompt, **kwargs)
        self._model_id = model_id
        self._device = device
        self._trust_remote_code = trust_remote_code
        self._pipeline = None

    def _get_pipeline(self):
        """Lazy load the transformers pipeline."""
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-generation",
                model=self._model_id,
                device=self._device if "cuda" in self._device else -1,
                trust_remote_code=self._trust_remote_code,
            )
        return self._pipeline

    def get_model_adapter(self, model_id: str, **kwargs):
        """Provide a HuggingFace ModelAdapter.

        Args:
            model_id: Model identifier (ignored, uses instance model_id).
            **kwargs: Additional arguments (e.g., register_name).

        Returns:
            HuggingFaceModelAdapter instance.
        """
        from maseval.interface.inference import HuggingFaceModelAdapter

        pipe = self._get_pipeline()
        adapter = HuggingFaceModelAdapter(
            model=pipe,
            model_id=self._model_id,
            default_generation_params={
                "max_new_tokens": 32,
                "do_sample": False,
                "return_full_text": False,
            },
        )

        # Register for tracing if requested
        register_name = kwargs.get("register_name")
        if register_name:
            self.register("models", register_name, adapter)

        return adapter


def save_predictions_for_disco(
    results: list,
    output_path: str,
    anchor_points: Optional[list] = None,
    n_choices: int = 4,
):
    """Save predictions in format compatible with DISCO predictor.

    Creates a predictions tensor of shape (1, n_questions, n_choices)
    where the values are logits (1.0 for predicted choice, 0.0 for others).

    Args:
        results: Benchmark results list.
        output_path: Path to save predictions pickle.
        anchor_points: Optional anchor points for ordering.
        n_choices: Number of answer choices (default 4 for A/B/C/D).
    """
    # Build predictions array
    # For now, we create simple binary predictions (1.0 for predicted, 0.0 for others)
    # A more sophisticated implementation would capture actual logits

    predictions_list = []

    if anchor_points is not None:
        # Order results by anchor points
        result_by_doc_id = {}
        for res in results:
            if res.get("status") == "success" and res.get("eval"):
                for entry in res["eval"]:
                    doc_id = entry.get("doc_id")
                    if doc_id is not None:
                        result_by_doc_id[doc_id] = entry

        for doc_id in anchor_points:
            entry = result_by_doc_id.get(doc_id, {})
            predicted = entry.get("predicted", -1)
            pred_vec = [0.0] * n_choices
            if 0 <= predicted < n_choices:
                pred_vec[predicted] = 1.0
            predictions_list.append(pred_vec)
    else:
        # Use results in order
        for res in results:
            if res.get("status") == "success" and res.get("eval"):
                for entry in res["eval"]:
                    predicted = entry.get("predicted", -1)
                    pred_vec = [0.0] * n_choices
                    if 0 <= predicted < n_choices:
                        pred_vec[predicted] = 1.0
                    predictions_list.append(pred_vec)

    predictions = np.array(predictions_list)
    predictions = predictions.reshape(1, -1, n_choices)  # (1, n_questions, n_choices)

    with open(output_path, "wb") as f:
        pickle.dump(predictions, f)

    print(f"Saved predictions tensor to {output_path}")
    print(f"  Shape: {predictions.shape}")


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 80)
    print("MMLU Benchmark - MASEval")
    print("=" * 80)
    print(f"Model: {args.model_id}")
    print(f"Data path: {args.data_path}")
    print(f"Anchor points: {args.anchor_points_path or 'None (evaluate all)'}")
    print(f"Use full prompt: {args.use_full_prompt}")
    print(f"Device: {args.device}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)

    # Load tasks
    print("\nLoading tasks...")
    tasks = load_tasks(
        data_path=args.data_path,
        anchor_points_path=args.anchor_points_path,
        limit=args.limit,
    )
    print(f"Loaded {len(tasks)} tasks")

    if args.anchor_points_path:
        anchor_points = tasks._anchor_points
        print(f"Filtering to {len(anchor_points)} anchor points")
    else:
        anchor_points = None

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create result logger
    logger = FileResultLogger(
        output_dir=str(output_dir),
        filename_pattern="mmlu_{timestamp}.jsonl",
        validate_on_completion=False,
    )

    # Create benchmark
    benchmark = HuggingFaceMMLUBenchmark(
        model_id=args.model_id,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        use_full_prompt=args.use_full_prompt,
        callbacks=[logger],
        num_workers=args.num_workers,
    )

    # Run evaluation
    print("\nRunning evaluation...")
    agent_data = {
        "model_id": args.model_id,
        "use_full_prompt": args.use_full_prompt,
    }
    results = benchmark.run(tasks=tasks, agent_data=agent_data)

    # Compute metrics
    metrics = compute_benchmark_metrics(results)

    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(f"Total tasks: {metrics['total_tasks']}")
    print(f"Correct: {metrics['correct_count']}")
    print(f"Accuracy: {metrics['acc']:.4f}")
    print(f"Accuracy (norm): {metrics['acc_norm']:.4f}")

    # Save predictions for DISCO if requested
    if args.predictions_path:
        save_predictions_for_disco(
            results=results,
            output_path=args.predictions_path,
            anchor_points=anchor_points,
        )

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "model_id": args.model_id,
                "data_path": str(args.data_path),
                "anchor_points_path": str(args.anchor_points_path) if args.anchor_points_path else None,
                "use_full_prompt": args.use_full_prompt,
                "metrics": metrics,
            },
            f,
            indent=2,
        )
    print(f"\nSummary saved to: {summary_path}")
    print(f"Full results saved to: {logger.output_dir}")


if __name__ == "__main__":
    main()
