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

    # Run with DISCO prediction (predict full benchmark performance from anchor points)
    python mmlu_benchmark.py \\
        --model_id "alignment-handbook/zephyr-7b-sft-full" \\
        --data_path /path/to/mmlu_prompts_examples.json \\
        --anchor_points_path /path/to/anchor_points_disagreement.pkl \\
        --use_full_prompt \\
        --disco_prediction \\
        --disco_model_path /path/to/fitted_weights.pkl \\
        --disco_transform_path /path/to/transform.pkl \\
        --pca 256

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

    # DISCO prediction arguments
    parser.add_argument(
        "--disco_prediction",
        action="store_true",
        help="Enable DISCO prediction of full benchmark performance from anchor points",
    )
    parser.add_argument(
        "--disco_model_path",
        type=str,
        default=None,
        help="Path to DISCO fitted weights pickle file (required if --disco_prediction)",
    )
    parser.add_argument(
        "--disco_transform_path",
        type=str,
        default=None,
        help="Path to DISCO PCA transform pickle file (required if --disco_prediction with --pca)",
    )
    parser.add_argument(
        "--pca",
        type=int,
        default=256,
        help="PCA dimension for DISCO embeddings (default: 256)",
    )
    parser.add_argument(
        "--pad_to_size",
        type=int,
        default=None,
        help="Pad predictions to this size with -inf (default: no padding, disco-public uses 31)",
    )
    parser.add_argument(
        "--use_lmeval_batching",
        action="store_true",
        help="Use lm-evaluation-harness batching for exact numerical match. This batches ALL requests together before computing logprobs.",
    )

    return parser.parse_args()


class HuggingFaceMMLUBenchmark(MMLUBenchmark):
    """MMLU Benchmark using HuggingFace transformers models.

    This concrete implementation uses log-likelihood based MCQ evaluation
    with the same optimizations as lm-evaluation-harness:

    1. Single forward pass per question (one-token continuation optimization)
    2. Batching multiple questions together
    3. Efficient log-softmax computation
    4. Proper left-padding for batch processing
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda:0",
        trust_remote_code: bool = False,
        use_full_prompt: bool = False,
        batch_size: int = 8,
        **kwargs,
    ):
        """Initialize HuggingFace MMLU benchmark.

        Args:
            model_id: HuggingFace model identifier.
            device: Device to run model on.
            trust_remote_code: Trust remote code when loading model.
            use_full_prompt: Use full prompt with few-shot examples.
            batch_size: Batch size for evaluation (number of questions per batch).
            **kwargs: Additional arguments passed to MMLUBenchmark.
        """
        super().__init__(use_full_prompt=use_full_prompt, **kwargs)
        self._model_id = model_id
        self._device = device
        self._trust_remote_code = trust_remote_code
        self._batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load the model and tokenizer for log-likelihood computation."""
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading model: {self._model_id}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_id,
                trust_remote_code=self._trust_remote_code,
            )
            self._tokenizer.padding_side = "left"
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Load model with torch_dtype="auto" to match lm-evaluation-harness exactly
            # This uses the model's native dtype (bfloat16 for most modern models)
            # Then move to device manually
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                trust_remote_code=self._trust_remote_code,
                torch_dtype="auto",
            )
            self._model = self._model.to(self._device)
            self._model.eval()

            # Note: We don't pre-cache choice token IDs here because they depend on context.
            # Token IDs are computed dynamically in _get_choice_token_id_in_context()
            # to match lm-evaluation-harness behavior exactly.

        return self._model, self._tokenizer

    def _get_choice_token_id_separate(self, choice: str) -> int:
        """Get the token ID for a choice when tokenized SEPARATELY.

        CRITICAL: lm-evaluation-harness encodes context and continuation separately,
        then concatenates. This means "A" is always tokenized standalone (token 330),
        NOT in context after "Answer:" (which would be token 28741).

        We must match this behavior to get identical log-likelihood values.

        Args:
            choice: The choice string (e.g., "A").

        Returns:
            Token ID for the choice (standalone tokenization).
        """
        _, tokenizer = self._load_model()

        # Tokenize choice ALONE (not in context) - this is how lm-eval does it
        choice_tokens = tokenizer.encode(choice, add_special_tokens=False)

        if len(choice_tokens) == 1:
            return choice_tokens[0]
        else:
            # Multi-token choice - return None to trigger multi-token fallback
            return None

    def _encode_pair(self, context: str, continuation: str) -> tuple:
        """Encode a context-continuation pair like lm-evaluation-harness.

        This matches lm-eval's _encode_pair method exactly:
        1. Encode whole = context + continuation
        2. Encode context alone
        3. continuation_enc = whole[len(context_enc):]

        This handles tokenization boundary effects correctly.

        Args:
            context: The context/prompt string.
            continuation: The continuation string (e.g., " A" with target_delimiter).

        Returns:
            Tuple of (context_enc, continuation_enc) token lists.
        """
        _, tokenizer = self._load_model()

        # Handle trailing spaces in context (move to continuation)
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        # Encode whole string together, then split
        whole_enc = tokenizer.encode(context + continuation, add_special_tokens=True)
        context_enc = tokenizer.encode(context, add_special_tokens=True)

        # Continuation tokens are what's left after context
        continuation_enc = whole_enc[len(context_enc) :]

        return context_enc, continuation_enc

    def _compute_logprobs_single_token(self, prompt: str, choices: list) -> list:
        """Compute log-likelihoods using single-token optimization.

        For MCQ with single-letter answers (A, B, C, D), we can compute all
        choices in one forward pass since they share the same context.

        IMPORTANT: To match lm-evaluation-harness EXACTLY:
        1. Use target_delimiter=" " before choices (e.g., " A" not "A")
        2. Use _encode_pair to handle tokenization boundaries correctly
        3. Input = (context + continuation)[:-1]
        4. Apply log_softmax to get log probabilities

        Args:
            prompt: The prompt/question text.
            choices: List of answer choice strings (e.g., ["A", "B", "C", "D"]).

        Returns:
            List of log-likelihoods, one per choice.
        """
        import torch

        model, _ = self._load_model()

        # lm-eval uses target_delimiter=" " for multiple choice tasks
        target_delimiter = " "

        # Encode first choice to get the shared context
        first_continuation = f"{target_delimiter}{choices[0]}"
        context_enc, first_cont_enc = self._encode_pair(prompt, first_continuation)

        # Build input: (context + continuation)[:-1]
        full_sequence = context_enc + first_cont_enc
        input_tokens = full_sequence[:-1]  # Remove last token

        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self._device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0]  # (seq_len, vocab_size)

            # Select logits at position where continuation is predicted
            # For single-token continuation, this is the last position
            inplen = len(input_tokens)
            contlen = len(first_cont_enc)
            selected_logits = logits[inplen - contlen : inplen]

            # Compute log-softmax
            log_probs = torch.nn.functional.log_softmax(selected_logits, dim=-1)

            # Get log prob for each choice's continuation token
            logprobs = []
            for choice in choices:
                continuation = f"{target_delimiter}{choice}"
                _, cont_enc = self._encode_pair(prompt, continuation)

                # Sum log probs for multi-token continuations
                total = 0.0
                for i, token_id in enumerate(cont_enc):
                    total += log_probs[i, token_id].item()
                logprobs.append(total)

        return logprobs

    def _compute_logprobs_batched(self, prompts: list, choices_list: list) -> list:
        """Compute log-likelihoods for a batch of prompts.

        For exact match with lm-evaluation-harness, we process each prompt
        individually using _compute_logprobs_single_token which uses the
        correct _encode_pair tokenization logic.

        Args:
            prompts: List of prompt strings.
            choices_list: List of choice lists (one per prompt).

        Returns:
            List of log-likelihood lists, one per prompt.
        """
        # For exact match with lm-eval, process individually
        # This ensures correct tokenization via _encode_pair
        all_logprobs = []
        for prompt, choices in zip(prompts, choices_list):
            logprobs = self._compute_logprobs_single_token(prompt, choices)
            all_logprobs.append(logprobs)

        return all_logprobs

    def precompute_all_logprobs_lmeval(self, tasks) -> dict:
        """Precompute log-likelihoods for ALL tasks using lm-eval's batching.

        CRITICAL: lm-evaluation-harness batches ALL requests together and uses
        its Collator class to reorder/group them. This affects floating-point
        precision for some edge cases. To get EXACT matches, we must process
        ALL requests together in a single batch.

        This method:
        1. Creates Instance objects for all task/choice combinations
        2. Calls lm-eval's HFLM.loglikelihood() with ALL instances
        3. Returns a mapping from doc_id to logprobs

        Args:
            tasks: Iterable of Task objects with prompt and choices.

        Returns:
            Dict mapping doc_id -> list of log-likelihoods for each choice.
        """
        import sys

        # Add lm-eval to path
        sys.path.insert(0, "/home/oh/arubinstein17/github/disco-public")
        sys.path.insert(0, "/home/oh/arubinstein17/github/disco-public/external/lm-evaluation-harness")

        from lm_eval.models.huggingface import HFLM
        from lm_eval.api.instance import Instance

        # Create HFLM model (this handles model loading internally)
        lm = HFLM(
            pretrained=self._model_id,
            trust_remote_code=self._trust_remote_code,
            batch_size=self._batch_size,
            device=self._device,
        )

        # lm-eval uses target_delimiter=" " for multiple choice tasks
        target_delimiter = " "
        choices = ["A", "B", "C", "D"]
        continuations = [f"{target_delimiter}{c}" for c in choices]

        # Build ALL instances like lm-eval task system does
        all_instances = []
        instance_map = {}  # (doc_id, choice_idx) -> position in results

        for task in tasks:
            doc_id = task.metadata.get("doc_id")
            # Get prompt from task - use full_prompt from environment_data if available
            if self.use_full_prompt and "full_prompt" in task.environment_data:
                prompt = task.environment_data["full_prompt"]
            else:
                prompt = task.query

            for i, cont in enumerate(continuations):
                inst = Instance(
                    request_type="loglikelihood",
                    doc={"doc_id": doc_id},
                    arguments=(prompt, cont),
                    idx=i,
                    metadata=("mmlu_prompts", doc_id, 1),
                )
                instance_map[(doc_id, i)] = len(all_instances)
                all_instances.append(inst)

        print(f"Precomputing logprobs for {len(all_instances)} instances ({len(all_instances) // len(choices)} tasks)")

        # Call loglikelihood with ALL instances at once - this is the key!
        results = lm.loglikelihood(all_instances)

        # Map results back to doc_ids
        doc_logprobs = {}
        for task in tasks:
            doc_id = task.metadata.get("doc_id")
            logprobs = []
            for i in range(len(choices)):
                pos = instance_map[(doc_id, i)]
                logprob, _ = results[pos]
                logprobs.append(logprob)
            doc_logprobs[doc_id] = logprobs

        # Store for later use
        self._precomputed_logprobs = doc_logprobs

        return doc_logprobs

    def _compute_logprobs_multi_token(self, prompt: str, choices: list) -> list:
        """Compute log-likelihoods for multi-token continuations.

        This is the fallback for when answer choices have multiple tokens.
        Uses _encode_pair to match lm-evaluation-harness exactly.

        Args:
            prompt: The prompt/question text.
            choices: List of answer choice strings.

        Returns:
            List of log-likelihoods, one per choice.
        """
        import torch

        model, _ = self._load_model()

        # lm-eval uses target_delimiter=" " for multiple choice tasks
        target_delimiter = " "

        all_logprobs = []
        for choice in choices:
            continuation = f"{target_delimiter}{choice}"

            # Use _encode_pair for correct tokenization
            context_enc, continuation_enc = self._encode_pair(prompt, continuation)

            # Build input: (context + continuation)[:-1]
            full_sequence = context_enc + continuation_enc
            input_tokens = full_sequence[:-1]

            input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self._device)

            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits[0]  # (seq_len, vocab_size)

                # Select continuation logits
                inplen = len(input_tokens)
                contlen = len(continuation_enc)
                selected = logits[inplen - contlen : inplen]

                # Compute log-softmax
                log_probs = torch.nn.functional.log_softmax(selected, dim=-1)

                # Sum log probs for all continuation tokens
                total = 0.0
                for i, token_id in enumerate(continuation_enc):
                    total += log_probs[i, token_id].item()

                all_logprobs.append(total)

        return all_logprobs

    def run_agents(
        self,
        agents,
        task,
        environment,
        query: str = "",
    ):
        """Execute log-likelihood based MCQ evaluation.

        Uses precomputed logprobs if available (for exact lm-eval match),
        otherwise falls back to single-forward-pass optimization for
        single-token answers, or multi-token batched computation.
        """
        # Get the prompt from environment
        prompt = environment.get_prompt()
        choices = environment.state.get("choices", ["A", "B", "C", "D"])
        doc_id = task.metadata.get("doc_id") if task else None

        # Check if we have precomputed logprobs (for exact lm-eval match)
        if hasattr(self, "_precomputed_logprobs") and doc_id is not None:
            logprobs = self._precomputed_logprobs.get(doc_id)
            if logprobs is not None:
                # Use precomputed values for exact match
                best_idx = logprobs.index(max(logprobs))
                answer = choices[best_idx]

                # Store logprobs in environment for later retrieval
                environment.state["logprobs"] = logprobs
                environment.state["predicted_idx"] = best_idx

                # Record in agent messages for tracing
                agent = agents[0]
                agent.agent._messages.append({"role": "user", "content": prompt})
                agent.agent._messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "logprobs": logprobs,
                    }
                )

                return answer

        # Fall back to computing logprobs on-the-fly
        # Load model
        self._load_model()

        # lm-eval uses target_delimiter=" " for multiple choice tasks
        target_delimiter = " "

        # Check if all choices result in single-token continuations
        # using _encode_pair to get the correct tokenization
        all_single_token = True
        for choice in choices:
            continuation = f"{target_delimiter}{choice}"
            _, cont_enc = self._encode_pair(prompt, continuation)
            if len(cont_enc) != 1:
                all_single_token = False
                break

        if all_single_token:
            # Use optimized single-token path (one forward pass)
            logprobs = self._compute_logprobs_single_token(prompt, choices)
        else:
            # Fall back to multi-token computation
            logprobs = self._compute_logprobs_multi_token(prompt, choices)

        # Select the choice with highest log-probability
        best_idx = logprobs.index(max(logprobs))
        answer = choices[best_idx]

        # Store logprobs in environment for later retrieval if needed
        environment.state["logprobs"] = logprobs
        environment.state["predicted_idx"] = best_idx

        # Record in agent messages for tracing
        agent = agents[0]
        agent.agent._messages.append({"role": "user", "content": prompt})
        agent.agent._messages.append(
            {
                "role": "assistant",
                "content": answer,
                "logprobs": logprobs,
            }
        )

        return answer

    def get_model_adapter(self, model_id: str, **kwargs):
        """Provide a HuggingFace ModelAdapter.

        Note: For logprobs-based evaluation, we don't actually use the adapter
        for generation. This is kept for API compatibility.

        Args:
            model_id: Model identifier (ignored, uses instance model_id).
            **kwargs: Additional arguments (e.g., register_name).

        Returns:
            HuggingFaceModelAdapter instance.
        """
        from maseval.interface.inference import HuggingFaceModelAdapter

        # Create a minimal adapter for compatibility
        # The actual evaluation uses _compute_logprobs_*
        class DummyCallable:
            def __call__(self, prompt, **kwargs):
                return ""

        adapter = HuggingFaceModelAdapter(
            model=DummyCallable(),
            model_id=self._model_id,
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
    pad_to_size: Optional[int] = None,
):
    """Save predictions in format compatible with DISCO predictor.

    Creates a predictions tensor of shape (1, n_questions, pad_to_size)
    where the values are log-probabilities.

    If logprobs are available in the results (from logprobs-based evaluation),
    uses actual log-likelihoods. Otherwise, falls back to 0/-inf format.

    Args:
        results: Benchmark results list.
        output_path: Path to save predictions pickle.
        anchor_points: Optional anchor points for ordering.
        n_choices: Number of answer choices (default 4 for A/B/C/D).
        pad_to_size: Pad predictions to this size with -inf (default: no padding).
    """
    predictions_list = []

    def get_pred_vec(entry, n_choices):
        """Extract prediction vector from entry, using logprobs if available."""
        # Check if actual logprobs are available
        logprobs = entry.get("logprobs")
        if logprobs is not None and len(logprobs) >= n_choices:
            return logprobs[:n_choices]

        # Fall back to 0/-inf format based on predicted index
        predicted = entry.get("predicted", -1)
        pred_vec = [float("-inf")] * n_choices
        if 0 <= predicted < n_choices:
            pred_vec[predicted] = 0.0
        return pred_vec

    def extract_eval_entries(res):
        """Extract evaluation entries from a result dict."""
        eval_data = res.get("eval")
        if eval_data is None:
            return []
        if isinstance(eval_data, list):
            return eval_data
        if isinstance(eval_data, dict):
            return [eval_data]
        return []

    if anchor_points is not None:
        # Order results by anchor points
        result_by_doc_id = {}
        for res in results:
            for entry in extract_eval_entries(res):
                doc_id = entry.get("doc_id")
                if doc_id is not None:
                    result_by_doc_id[doc_id] = entry

        for doc_id in anchor_points:
            entry = result_by_doc_id.get(doc_id, {})
            pred_vec = get_pred_vec(entry, n_choices)
            predictions_list.append(pred_vec)
    else:
        # Use results in order
        for res in results:
            for entry in extract_eval_entries(res):
                pred_vec = get_pred_vec(entry, n_choices)
                predictions_list.append(pred_vec)

    predictions = np.array(predictions_list)

    # Pad to specified size if requested
    if pad_to_size is not None and predictions.shape[1] < pad_to_size:
        padding = np.full(
            (predictions.shape[0], pad_to_size - predictions.shape[1]),
            float("-inf"),
            dtype=predictions.dtype,
        )
        predictions = np.concatenate([predictions, padding], axis=1)

    predictions = predictions.reshape(1, -1, predictions.shape[-1])  # (1, n_questions, n_choices)

    if output_path and output_path != "/dev/null":
        with open(output_path, "wb") as f:
            pickle.dump(predictions, f)
        print(f"Saved predictions tensor to {output_path}")
        print(f"  Shape: {predictions.shape}")
        print(f"  Dtype: {predictions.dtype}")
    else:
        print(f"Built predictions tensor with shape: {predictions.shape}")

    return predictions


def compute_disco_embedding(
    predictions: np.ndarray,
    pca: int,
    transform=None,
    apply_softmax: bool = True,
) -> tuple:
    """Compute DISCO embeddings from predictions.

    This implements the embedding computation from disco-public/experiments.py.

    Args:
        predictions: Predictions tensor of shape (n_models, n_anchor_points, n_classes).
        pca: PCA dimension for dimensionality reduction.
        transform: Pre-fitted PCA transform. If None, a new one will be fitted.
        apply_softmax: Whether to apply softmax to predictions.

    Returns:
        Tuple of (embeddings, transform) where embeddings has shape (n_models, pca).
    """
    try:
        import torch
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError("DISCO prediction requires torch and sklearn. Install with: pip install torch scikit-learn") from e

    # Convert to torch tensor
    preds_tensor = torch.Tensor(predictions)

    # Apply softmax if requested
    if apply_softmax:
        emb_unreduced = preds_tensor.softmax(dim=-1)
    else:
        emb_unreduced = preds_tensor

    # Flatten to (n_models, n_anchor_points * n_classes)
    emb_unreduced = emb_unreduced.reshape(emb_unreduced.shape[0], -1)

    # Apply PCA
    if pca is not None:
        if transform is None:
            # Fit new PCA transform
            transform = PCA(
                n_components=pca,
                svd_solver="full",
                random_state=42,
            ).fit(emb_unreduced.numpy())

        emb = transform.transform(emb_unreduced.numpy())
        emb = torch.Tensor(emb)
    else:
        emb = emb_unreduced

    return emb, transform


def predict_with_disco(
    predictions: np.ndarray,
    model_path: str,
    transform_path: Optional[str] = None,
    pca: int = 256,
) -> dict:
    """Predict full benchmark performance using DISCO.

    This implements the prediction logic from disco-public/scripts/predict_model_performance.py.

    Args:
        predictions: Predictions tensor of shape (n_models, n_anchor_points, n_classes).
        model_path: Path to fitted weights pickle file.
        transform_path: Path to PCA transform pickle file (optional, can be in model_path).
        pca: PCA dimension (default: 256).

    Returns:
        Dict with predicted accuracies and metadata.
    """
    # Load model data
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    if not isinstance(model_data, dict):
        raise ValueError(f"model_path must contain a dict. Got {type(model_data)}")

    # Get transform
    if transform_path is not None:
        with open(transform_path, "rb") as f:
            transform = pickle.load(f)
    elif "transform" in model_data:
        transform = model_data["transform"]
    else:
        raise ValueError("Transform not found. Provide --disco_transform_path or ensure the model file contains a 'transform' key.")

    # Get fitted weights
    if "fitted_weights" in model_data:
        fitted_weights = model_data["fitted_weights"]
    else:
        # Assume the dict itself is fitted_weights (excluding transform)
        fitted_weights = {k: v for k, v in model_data.items() if k != "transform"}
        if not fitted_weights:
            raise ValueError("Could not find fitted_weights in model file.")

    # Get metadata or infer from structure
    if "sampling_name" in model_data:
        sampling_name = model_data["sampling_name"]
    else:
        sampling_name = list(fitted_weights.keys())[0]

    if "number_item" in model_data:
        number_item = model_data["number_item"]
    else:
        number_item = list(fitted_weights[sampling_name].keys())[0]

    if "fitted_model_type" in model_data:
        fitted_model_type = model_data["fitted_model_type"]
    else:
        fitted_model_type = list(fitted_weights[sampling_name][number_item].keys())[0]

    print(f"  Using: sampling={sampling_name}, n_items={number_item}, model={fitted_model_type}")

    # Compute embeddings
    embeddings, _ = compute_disco_embedding(
        predictions,
        pca=pca,
        transform=transform,
        apply_softmax=True,
    )

    # Get the fitted model
    fitted_model = fitted_weights[sampling_name][number_item][fitted_model_type]

    # Predict accuracies for each model
    predicted_accs = {}
    for model_idx in range(embeddings.shape[0]):
        model_embedding = embeddings[model_idx]

        # Convert to numpy if needed
        if hasattr(model_embedding, "numpy"):
            model_embedding_np = model_embedding.numpy()
        else:
            model_embedding_np = np.array(model_embedding)

        # Predict using fitted model
        predicted_acc = fitted_model.predict(model_embedding_np.reshape(1, -1))[0]
        predicted_accs[model_idx] = predicted_acc

    return {
        "predicted_accuracies": predicted_accs,
        "sampling_name": sampling_name,
        "number_item": number_item,
        "fitted_model_type": fitted_model_type,
        "pca": pca,
    }


def main():
    """Main entry point."""
    args = parse_args()

    # Validate DISCO prediction arguments
    if args.disco_prediction:
        if args.disco_model_path is None:
            raise ValueError("--disco_model_path is required when --disco_prediction is enabled")
        if args.anchor_points_path is None:
            raise ValueError("--anchor_points_path is required when --disco_prediction is enabled")
        if args.pca is not None and args.disco_transform_path is None:
            print("Warning: --pca specified without --disco_transform_path. Transform will be loaded from model file if available.")

    print("=" * 80)
    print("MMLU Benchmark - MASEval")
    print("=" * 80)
    print(f"Model: {args.model_id}")
    print(f"Data path: {args.data_path}")
    print(f"Anchor points: {args.anchor_points_path or 'None (evaluate all)'}")
    print(f"Use full prompt: {args.use_full_prompt}")
    print(f"Device: {args.device}")
    print(f"Output dir: {args.output_dir}")
    if args.disco_prediction:
        print("DISCO prediction: ENABLED")
        print(f"  Model path: {args.disco_model_path}")
        print(f"  Transform path: {args.disco_transform_path or '(from model file)'}")
        print(f"  PCA dimension: {args.pca}")
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

    # Optionally precompute logprobs using lm-eval batching for exact match
    if args.use_lmeval_batching:
        print("\nPrecomputing logprobs using lm-eval batching (for exact numerical match)...")
        # Get task list for precomputation
        task_list = list(tasks._anchor_tasks if hasattr(tasks, "_anchor_tasks") else tasks._tasks)
        benchmark.precompute_all_logprobs_lmeval(task_list)

    # Run evaluation
    print("\nRunning evaluation...")
    agent_data = {
        "model_id": args.model_id,
        "use_full_prompt": args.use_full_prompt,
    }
    results = benchmark.run(tasks=tasks, agent_data=agent_data)

    # Compute metrics on evaluated tasks
    metrics = compute_benchmark_metrics(results)

    print("\n" + "=" * 80)
    print("Results Summary (Evaluated Tasks)")
    print("=" * 80)
    print(f"Total tasks: {metrics['total_tasks']}")
    print(f"Correct: {metrics['correct_count']}")
    print(f"Accuracy (on anchor points): {metrics['acc']:.4f}")
    print(f"Accuracy norm (on anchor points): {metrics['acc_norm']:.4f}")

    # Build predictions tensor for DISCO
    predictions = None
    if args.predictions_path or args.disco_prediction:
        predictions = save_predictions_for_disco(
            results=results,
            output_path=args.predictions_path if args.predictions_path else None,
            anchor_points=anchor_points,
            pad_to_size=args.pad_to_size,
        )

    # Run DISCO prediction if enabled
    disco_results = None
    if args.disco_prediction:
        print("\n" + "=" * 80)
        print("DISCO Prediction")
        print("=" * 80)
        print("Computing embeddings and predicting full benchmark accuracy...")

        disco_results = predict_with_disco(
            predictions=predictions,
            model_path=args.disco_model_path,
            transform_path=args.disco_transform_path,
            pca=args.pca,
        )

        print("\n" + "-" * 40)
        print("DISCO Predicted Full Benchmark Accuracy:")
        print("-" * 40)
        for model_idx, acc in disco_results["predicted_accuracies"].items():
            print(f"  Model {model_idx}: {acc:.6f}")

        # Compare with actual anchor accuracy
        print("\n" + "-" * 40)
        print("Comparison:")
        print("-" * 40)
        predicted_acc = disco_results["predicted_accuracies"][0]
        actual_anchor_acc = metrics["acc"]
        print(f"  Actual accuracy (on anchor points): {actual_anchor_acc:.6f}")
        print(f"  DISCO predicted accuracy (full benchmark): {predicted_acc:.6f}")
        print(f"  Difference: {predicted_acc - actual_anchor_acc:+.6f}")

    # Save summary
    summary_data = {
        "model_id": args.model_id,
        "data_path": str(args.data_path),
        "anchor_points_path": str(args.anchor_points_path) if args.anchor_points_path else None,
        "use_full_prompt": args.use_full_prompt,
        "metrics": metrics,
    }

    if disco_results:
        summary_data["disco_prediction"] = {
            "predicted_accuracy": disco_results["predicted_accuracies"][0],
            "sampling_name": disco_results["sampling_name"],
            "number_item": disco_results["number_item"],
            "fitted_model_type": disco_results["fitted_model_type"],
            "pca": disco_results["pca"],
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print(f"Full results saved to: {logger.output_dir}")


if __name__ == "__main__":
    main()
