"""Tests for MMLU benchmark DISCO prediction output identity.

These tests verify that the MASEval MMLU benchmark produces output
that is compatible with (and optionally identical to) disco-public.

To run the full identity test (requires lm-eval-harness and GPU):
    pytest tests/test_benchmark/test_mmlu_disco_identity.py -v -m "slow"

To run format-only tests:
    pytest tests/test_benchmark/test_mmlu_disco_identity.py -v -m "not slow"
"""

import pickle
from pathlib import Path

import numpy as np
import pytest

# Reference files from disco-public
DISCO_PUBLIC_PATH = Path("/home/oh/arubinstein17/github/disco-public")
REFERENCE_PREDICTIONS_PATH = DISCO_PUBLIC_PATH / "data/model_outputs/mmlu/local_predictions_r90.pkl"
ANCHOR_POINTS_PATH = DISCO_PUBLIC_PATH / "data/model_outputs/mmlu/anchor_points_disagreement.pkl"
MMLU_DATA_PATH = DISCO_PUBLIC_PATH / "notebooks/mmlu_prompts_examples.json"
FITTED_WEIGHTS_PATH = DISCO_PUBLIC_PATH / "data/model_outputs/mmlu/fitted_weights.pkl"
TRANSFORM_PATH = DISCO_PUBLIC_PATH / "data/model_outputs/mmlu/transform.pkl"


@pytest.fixture
def reference_predictions():
    """Load reference predictions from disco-public."""
    if not REFERENCE_PREDICTIONS_PATH.exists():
        pytest.skip(f"Reference file not found: {REFERENCE_PREDICTIONS_PATH}")
    with open(REFERENCE_PREDICTIONS_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture
def anchor_points():
    """Load anchor points from disco-public."""
    if not ANCHOR_POINTS_PATH.exists():
        pytest.skip(f"Anchor points file not found: {ANCHOR_POINTS_PATH}")
    with open(ANCHOR_POINTS_PATH, "rb") as f:
        points = pickle.load(f)
        if isinstance(points, np.ndarray):
            return points.tolist()
        return points


class TestPredictionsFormat:
    """Test that predictions have the correct format for DISCO."""

    def test_reference_predictions_shape(self, reference_predictions):
        """Reference predictions should have shape (1, n_anchors, 31)."""
        assert reference_predictions.ndim == 3
        assert reference_predictions.shape[0] == 1
        assert reference_predictions.shape[1] == 100  # n_anchor_points
        assert reference_predictions.shape[2] == 31  # padded size

    def test_reference_predictions_dtype(self, reference_predictions):
        """Reference predictions should be float64."""
        assert reference_predictions.dtype == np.float64

    def test_reference_predictions_contains_logprobs(self, reference_predictions):
        """Reference predictions should contain log probabilities (negative values)."""
        # First 4 columns should be log probs (negative, but not -inf)
        first_4 = reference_predictions[0, :, :4]
        valid_mask = ~np.isinf(first_4)
        assert valid_mask.sum() > 0  # At least some valid values
        assert (first_4[valid_mask] < 0).all()  # All negative (log probs)

    def test_reference_predictions_padding(self, reference_predictions):
        """Reference predictions should be padded with -inf after first 4 columns."""
        padding = reference_predictions[0, :, 4:]
        assert np.all(np.isinf(padding) & (padding < 0))


class TestAnchorPoints:
    """Test anchor points loading and format."""

    def test_anchor_points_count(self, anchor_points):
        """Should have 100 anchor points."""
        assert len(anchor_points) == 100

    def test_anchor_points_are_integers(self, anchor_points):
        """Anchor points should be integers (doc_ids)."""
        assert all(isinstance(p, (int, np.integer)) for p in anchor_points)


class TestDISCOPredictionWorkflow:
    """Test the DISCO prediction workflow with reference files."""

    @pytest.mark.skipif(not all([FITTED_WEIGHTS_PATH.exists(), TRANSFORM_PATH.exists()]), reason="DISCO model files not available")
    def test_disco_prediction_with_reference_predictions(self, reference_predictions):
        """Test DISCO prediction produces reasonable accuracy estimate."""
        # Import DISCO prediction functions
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        # Load model and transform
        with open(FITTED_WEIGHTS_PATH, "rb") as f:
            model_data = pickle.load(f)
        with open(TRANSFORM_PATH, "rb") as f:
            transform = pickle.load(f)

        # Get fitted weights
        if "fitted_weights" in model_data:
            fitted_weights = model_data["fitted_weights"]
        else:
            fitted_weights = {k: v for k, v in model_data.items() if k != "transform"}

        sampling_name = list(fitted_weights.keys())[0]
        number_item = list(fitted_weights[sampling_name].keys())[0]
        fitted_model_type = list(fitted_weights[sampling_name][number_item].keys())[0]
        fitted_model = fitted_weights[sampling_name][number_item][fitted_model_type]

        # Compute embeddings
        preds_tensor = torch.Tensor(reference_predictions)
        emb_unreduced = preds_tensor.softmax(dim=-1)
        emb_unreduced = emb_unreduced.reshape(emb_unreduced.shape[0], -1)
        emb = transform.transform(emb_unreduced.numpy())
        emb = torch.Tensor(emb)

        # Predict accuracy
        predicted_acc = fitted_model.predict(emb.numpy())[0]

        # Check that predicted accuracy is reasonable (between 0 and 1)
        assert 0.0 <= predicted_acc <= 1.0

        # For zephyr-7b-sft-full on MMLU, we expect roughly 50-70% accuracy
        assert 0.4 <= predicted_acc <= 0.8, f"Predicted accuracy {predicted_acc} seems unreasonable"

        print(f"DISCO predicted accuracy: {predicted_acc:.4f}")


@pytest.mark.slow
class TestOutputIdentity:
    """Tests that verify exact output identity with disco-public.

    These tests are marked slow because they require running the full
    lm-evaluation-harness evaluation, which takes significant time and GPU.
    """

    @pytest.mark.skipif(not all([MMLU_DATA_PATH.exists(), ANCHOR_POINTS_PATH.exists()]), reason="Required data files not available")
    def test_lm_eval_wrapper_produces_identical_output(self, reference_predictions, anchor_points, tmp_path):
        """Test that lm_eval_wrapper produces identical output to disco-public."""
        try:
            from examples.mmlu_benchmark.lm_eval_wrapper import evaluate_with_lm_eval
        except ImportError:
            pytest.skip("lm_eval_wrapper not available")

        output_path = tmp_path / "test_predictions.pkl"

        # Run evaluation (lm-eval deps are imported inside evaluate_with_lm_eval)
        try:
            predictions, correctness = evaluate_with_lm_eval(
                model_id="alignment-handbook/zephyr-7b-sft-full",
                data_path=str(MMLU_DATA_PATH),
                anchor_points_path=str(ANCHOR_POINTS_PATH),
                output_path=str(output_path),
                pad_to_size=31,
                device="cuda:0",
                batch_size=8,
                trust_remote_code=True,
                use_full_prompt=True,
            )
        except ImportError as e:
            pytest.skip(f"lm-eval dependencies not available: {e}")

        # Load produced predictions
        with open(output_path, "rb") as f:
            produced_predictions = pickle.load(f)

        # Assert identical shape
        assert produced_predictions.shape == reference_predictions.shape, (
            f"Shape mismatch: {produced_predictions.shape} vs {reference_predictions.shape}"
        )

        # Assert identical dtype
        assert produced_predictions.dtype == reference_predictions.dtype, (
            f"Dtype mismatch: {produced_predictions.dtype} vs {reference_predictions.dtype}"
        )

        # Assert values are very close (allowing for floating point differences)
        # Use a relative tolerance since log probs can have large absolute values
        np.testing.assert_allclose(
            produced_predictions, reference_predictions, rtol=1e-5, atol=1e-10, err_msg="Predictions do not match reference"
        )

        print("Output identity test PASSED!")


def test_format_compatibility_with_maseval_output():
    """Test that MASEval's save_predictions_for_disco produces compatible format.

    Note: This tests format compatibility, not value identity. For identical
    values, use the lm_eval_wrapper which uses lm-evaluation-harness directly.
    """
    import sys
    from pathlib import Path

    # Add examples to path for import
    examples_path = Path(__file__).parent.parent.parent / "examples" / "mmlu_benchmark"
    if str(examples_path.parent) not in sys.path:
        sys.path.insert(0, str(examples_path.parent))

    try:
        from mmlu_benchmark.mmlu_benchmark import save_predictions_for_disco
    except ImportError as e:
        pytest.skip(f"mmlu_benchmark not available: {e}")

    # Create mock results
    mock_results = [
        {
            "status": "success",
            "eval": [
                {"doc_id": 0, "predicted": 0, "acc": 1.0, "acc_norm": 1.0},
            ],
        },
        {
            "status": "success",
            "eval": [
                {"doc_id": 1, "predicted": 2, "acc": 0.0, "acc_norm": 0.0},
            ],
        },
    ]

    predictions = save_predictions_for_disco(
        results=mock_results,
        output_path=None,  # Don't save, just return
        anchor_points=[0, 1],
        n_choices=4,
    )

    # Check format
    assert predictions.ndim == 3
    assert predictions.shape[0] == 1  # n_models
    assert predictions.shape[1] == 2  # n_questions
    assert predictions.shape[2] == 4  # n_choices


# Path to MASEval-produced predictions (set by running mmlu_benchmark.py with --predictions_path)
MASEVAL_PREDICTIONS_PATH = Path("/weka/oh/arubinstein17/github/disco-public/output/mmlu_maseval/maseval_predictions.pkl")


class TestMASEvalPredictionsComparison:
    """Tests that compare MASEval predictions with disco-public reference.

    These tests verify that predictions saved via mmlu_benchmark.py with
    --predictions_path are format-compatible with disco-public output.

    To run these tests, first generate predictions by running:
        python examples/mmlu_benchmark/mmlu_benchmark.py \\
            --model_id alignment-handbook/zephyr-7b-sft-full \\
            --data_path /path/to/mmlu_prompts_examples.json \\
            --anchor_points_path /path/to/anchor_points.pkl \\
            --predictions_path /path/to/maseval_predictions.pkl \\
            --pad_to_size 31
    """

    @pytest.fixture
    def maseval_predictions(self):
        """Load MASEval-produced predictions."""
        if not MASEVAL_PREDICTIONS_PATH.exists():
            pytest.skip(f"MASEval predictions not found: {MASEVAL_PREDICTIONS_PATH}. Run mmlu_benchmark.py with --predictions_path first.")
        with open(MASEVAL_PREDICTIONS_PATH, "rb") as f:
            return pickle.load(f)

    def test_maseval_predictions_shape_matches_reference(self, maseval_predictions, reference_predictions):
        """MASEval predictions should have same shape as reference."""
        assert maseval_predictions.shape == reference_predictions.shape, (
            f"Shape mismatch: MASEval {maseval_predictions.shape} vs reference {reference_predictions.shape}"
        )

    def test_maseval_predictions_has_correct_format(self, maseval_predictions):
        """MASEval predictions should have correct format (padded with -inf)."""
        # Check shape
        assert maseval_predictions.ndim == 3
        assert maseval_predictions.shape[0] == 1  # n_models
        assert maseval_predictions.shape[1] == 100  # n_anchor_points
        assert maseval_predictions.shape[2] == 31  # padded size

        # Check padding (columns 4-31 should be -inf)
        padding = maseval_predictions[0, :, 4:]
        assert np.all(np.isinf(padding) & (padding < 0)), "Padding should be -inf"

    def test_maseval_predictions_match_reference_exactly(self, maseval_predictions, reference_predictions):
        """Compare log-likelihood values between MASEval and disco-public (lm-eval).

        MASEval now uses identical computation to lm-evaluation-harness:
        - Same tokenization (BOS token, separate context/continuation encoding)
        - Same model dtype (torch_dtype="auto" -> bfloat16)
        - Same log-softmax computation

        This test verifies the actual log-likelihood values are nearly identical,
        not just that argmax matches.
        """
        # Get choice columns (A, B, C, D) - exclude padding
        maseval_choices = maseval_predictions[0, :, :4]
        reference_choices = reference_predictions[0, :, :4]

        # Check absolute closeness of log-likelihood values
        # Use rtol=1e-3 and atol=1e-3 to allow for minor floating-point differences
        # is_close = np.allclose(maseval_choices, reference_choices, rtol=1e-3, atol=1e-3)
        is_close = np.allclose(maseval_choices, reference_choices)

        # Compute statistics for debugging
        abs_diff = np.abs(maseval_choices - reference_choices)
        max_diff = abs_diff.max()
        mean_diff = abs_diff.mean()

        # Find where max difference occurs
        max_idx = np.unravel_index(abs_diff.argmax(), abs_diff.shape)

        print(f"Max absolute difference: {max_diff:.6f} at index {max_idx}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print("Sample comparison at max diff location:")
        print(f"  MASEval:   {maseval_choices[max_idx[0], :]}")
        print(f"  Reference: {reference_choices[max_idx[0], :]}")

        # Also check argmax agreement for reference
        maseval_argmax = np.argmax(maseval_choices, axis=1)
        reference_argmax = np.argmax(reference_choices, axis=1)
        argmax_match_rate = (maseval_argmax == reference_argmax).mean()
        print(f"Argmax match rate: {argmax_match_rate:.2%}")

        assert is_close, (
            f"Predictions not close enough. Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}. "
            f"Check model dtype, tokenization, and log-softmax computation match lm-eval."
        )

    def test_disco_prediction_produces_valid_accuracy(self, maseval_predictions, reference_predictions):
        """DISCO prediction from MASEval predictions should produce valid accuracy.

        MASEval now uses the same log-likelihood scoring as lm-evaluation-harness,
        so predictions should contain actual log-likelihoods (not 0/-inf values).

        This test checks that:
        1. MASEval predictions produce a valid (0-1) accuracy prediction
        2. Reference predictions produce a reasonable accuracy for the model
        3. Both predictions are very similar (since MASEval matches lm-eval)
        """
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        if not all([FITTED_WEIGHTS_PATH.exists(), TRANSFORM_PATH.exists()]):
            pytest.skip("DISCO model files not available")

        # Load model and transform
        with open(FITTED_WEIGHTS_PATH, "rb") as f:
            model_data = pickle.load(f)
        with open(TRANSFORM_PATH, "rb") as f:
            transform = pickle.load(f)

        # Get fitted model
        if "fitted_weights" in model_data:
            fitted_weights = model_data["fitted_weights"]
        else:
            fitted_weights = {k: v for k, v in model_data.items() if k != "transform"}

        sampling_name = list(fitted_weights.keys())[0]
        number_item = list(fitted_weights[sampling_name].keys())[0]
        fitted_model_type = list(fitted_weights[sampling_name][number_item].keys())[0]
        fitted_model = fitted_weights[sampling_name][number_item][fitted_model_type]

        def compute_disco_accuracy(predictions):
            """Compute DISCO-predicted accuracy from predictions tensor."""
            preds_tensor = torch.Tensor(predictions)
            emb_unreduced = preds_tensor.softmax(dim=-1)
            emb_unreduced = emb_unreduced.reshape(emb_unreduced.shape[0], -1)
            emb = transform.transform(emb_unreduced.numpy())
            return fitted_model.predict(emb)[0]

        # Reference should work normally
        reference_acc = compute_disco_accuracy(reference_predictions)
        print(f"DISCO predicted accuracy from reference predictions: {reference_acc:.4f}")
        assert 0.4 <= reference_acc <= 0.8, f"Reference accuracy {reference_acc} seems unreasonable"

        # MASEval predictions should now have actual log-likelihoods (matching lm-eval)
        maseval_acc = compute_disco_accuracy(maseval_predictions)
        print(f"DISCO predicted accuracy from MASEval predictions: {maseval_acc:.4f}")
        print(f"Difference: {abs(maseval_acc - reference_acc):.4f}")

        # Both should produce similar predictions since MASEval matches lm-eval
        assert not np.isnan(maseval_acc), "MASEval DISCO prediction is NaN"
        assert abs(maseval_acc - reference_acc) < 0.05, (
            f"DISCO predictions differ too much: MASEval={maseval_acc:.4f}, reference={reference_acc:.4f}. Expected difference < 0.05."
        )
