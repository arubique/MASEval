"""LM-Eval Harness wrapper for MMLU evaluation.

This module provides a thin wrapper around lm-evaluation-harness to produce
predictions in the exact same format as disco-public, enabling identical
output for unit testing.

Usage:
    from lm_eval_wrapper import evaluate_with_lm_eval

    predictions, correctness = evaluate_with_lm_eval(
        model_id="alignment-handbook/zephyr-7b-sft-full",
        anchor_points_path="/path/to/anchor_points.pkl",
        data_path="/path/to/mmlu_prompts_examples.json",
        pad_to_size=31,
    )
"""

import pickle
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def pad_predictions(predictions: list, max_num_answers: int = 31) -> list:
    """Pad predictions with -inf to max_num_answers.

    Args:
        predictions: List of log probabilities.
        max_num_answers: Target size for padding.

    Returns:
        Padded list of predictions.
    """
    if len(predictions) >= max_num_answers:
        return predictions[:max_num_answers]
    return predictions + [float("-inf")] * (max_num_answers - len(predictions))


def evaluate_with_lm_eval(
    model_id: str,
    data_path: str,
    anchor_points_path: Optional[str] = None,
    output_path: Optional[str] = None,
    pad_to_size: int = 31,
    device: str = "cuda:0",
    batch_size: int = 8,
    trust_remote_code: bool = True,
    use_full_prompt: bool = True,
    disco_public_path: str = "/home/oh/arubinstein17/github/disco-public",
) -> tuple:
    """Evaluate model using lm-evaluation-harness.

    This produces predictions in the exact same format as disco-public's
    run_lm_eval.py script.

    Args:
        model_id: HuggingFace model identifier.
        data_path: Path to MMLU prompts JSON file.
        anchor_points_path: Path to anchor points pickle file.
        output_path: Path to save predictions pickle.
        pad_to_size: Padding size for predictions (default 31).
        device: Device to run model on.
        batch_size: Batch size for evaluation.
        trust_remote_code: Trust remote code when loading model.
        use_full_prompt: Use full prompt with few-shot examples.
        disco_public_path: Path to disco-public repository.

    Returns:
        Tuple of (predictions, correctness) numpy arrays.
    """
    # Add disco-public to path for imports. Order matters: disco_root must be first
    # so "from scripts import run_lm_eval" resolves to disco-public/scripts, not
    # lm-evaluation-harness/scripts.
    disco_root = Path(disco_public_path)
    disco_scripts = disco_root / "scripts"
    lm_eval_path = disco_root / "external" / "lm-evaluation-harness"

    for path in (disco_scripts, lm_eval_path, disco_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    try:
        import lm_eval
        from lm_eval_postprocess import (
            convert_jsonl_results_to_arrays,
            load_jsonl,
            find_jsonl_file_in_directory,
        )
        from utils import load_pickle
        from utils_for_notebooks import pad_predictions as disco_pad_predictions
    except ImportError as e:
        raise ImportError(f"Could not import lm-eval dependencies. Ensure disco-public is available at {disco_public_path}. Error: {e}") from e

    # Load anchor points if provided
    anchor_points = None
    if anchor_points_path:
        anchor_points = load_pickle(anchor_points_path)
        if isinstance(anchor_points, np.ndarray):
            anchor_points = anchor_points.tolist()

    # Build model args
    model_args = f"pretrained={model_id}"
    if trust_remote_code:
        model_args += ",trust_remote_code=True"

    # Build gen_kwargs
    gen_kwargs = "max_gen_toks=128,output_scores=True,return_dict_in_generate=True"

    # Prepare lm_eval arguments
    eval_args = [
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        "mmlu_prompts",
        "--batch_size",
        str(batch_size),
        "--device",
        device,
        "--gen_kwargs",
        gen_kwargs,
        "--num_fewshot",
        "0",
        "--log_samples",
    ]

    if use_full_prompt:
        eval_args.append("--use_full_prompt")

    if anchor_points_path:
        eval_args.extend(["--anchor_points_path", anchor_points_path])
        eval_args.append("--skip_non_anchor_points")

    # Create temporary output directory
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        eval_args.extend(["--target_outputs_path", str(output_path).replace(".pkl", "_target_outputs.pkl")])
        eval_args.extend(["--output_path", tmpdir])
        eval_args.extend(["--predictions_path", str(output_path)])
        eval_args.append("--metric")
        eval_args.append("acc_norm")
        eval_args.append("--force_recompute")

        # Run disco-public's run_lm_eval.py as a subprocess so scripts.* resolves to disco-public/scripts
        # (in-process import would see lm-evaluation-harness/scripts first).
        print(f"Running run_lm_eval with args: {' '.join(eval_args)}")

        cmd = [sys.executable, str(disco_root / "scripts" / "run_lm_eval.py")] + eval_args
        subprocess.run(cmd, cwd=str(disco_root), check=True)

        # Find and load results
        jsonl_path = find_jsonl_file_in_directory(tmpdir)
        if jsonl_path is None:
            raise RuntimeError(f"Could not find JSONL results in {tmpdir}")

        jsonl_results = load_jsonl(jsonl_path)

        # Convert to arrays
        predictions_2d, correctness_1d, n_questions = convert_jsonl_results_to_arrays(jsonl_results, "acc_norm", anchor_points=anchor_points)

        # Pad predictions
        if pad_to_size is not None and predictions_2d.shape[1] < pad_to_size:
            padded_predictions_list = []
            for row in predictions_2d:
                padded_row = disco_pad_predictions(row.tolist(), max_num_answers=pad_to_size)
                padded_predictions_list.append(padded_row)
            predictions_2d = np.array(padded_predictions_list)

    # Reshape to (1, n_questions, n_choices)
    predictions = predictions_2d.reshape(1, -1, predictions_2d.shape[-1])

    # Save if output path provided
    if output_path:
        with open(output_path, "wb") as f:
            pickle.dump(predictions, f)
        print(f"Saved predictions to {output_path}")
        print(f"  Shape: {predictions.shape}")

    return predictions, correctness_1d


def main():
    """Command-line interface for lm-eval wrapper."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model using lm-evaluation-harness wrapper")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--anchor_points_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--pad_to_size", type=int, default=31)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_full_prompt", action="store_true")
    parser.add_argument(
        "--disco_public_path",
        type=str,
        default="/home/oh/arubinstein17/github/disco-public",
    )

    args = parser.parse_args()

    predictions, correctness = evaluate_with_lm_eval(
        model_id=args.model_id,
        data_path=args.data_path,
        anchor_points_path=args.anchor_points_path,
        output_path=args.output_path,
        pad_to_size=args.pad_to_size,
        device=args.device,
        batch_size=args.batch_size,
        trust_remote_code=args.trust_remote_code,
        use_full_prompt=args.use_full_prompt,
        disco_public_path=args.disco_public_path,
    )

    print("\nResults:")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Correctness shape: {correctness.shape}")
    print(f"  Accuracy: {correctness.mean():.4f}")


if __name__ == "__main__":
    main()
