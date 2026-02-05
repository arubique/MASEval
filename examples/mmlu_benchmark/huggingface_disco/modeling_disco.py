# Copyright 2025 MASEval contributors. DISCO predictor model for Hugging Face Hub.
#
# Self-contained: uses only numpy and huggingface_hub. Load with:
#   from transformers import AutoModel
#   model = AutoModel.from_pretrained("<USERNAME>/my-disco-mmlu", trust_remote_code=True)
#   acc = model.predict(predictions_tensor)  # predictions: (n_models, n_anchor_points, n_classes)

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np


def _pca_transform(X: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Apply PCA transform: (X - mean) @ components.T."""
    return (X - mean) @ components.T


def _predict_tree(
    X: np.ndarray,
    children_left: np.ndarray,
    children_right: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    value: np.ndarray,
) -> np.ndarray:
    """Predict for one tree; X (n_samples, n_features) -> (n_samples,)."""
    out = np.empty(X.shape[0], dtype=np.float64)
    for i in range(X.shape[0]):
        node = 0
        while children_left[node] != -1:
            if X[i, feature[node]] <= threshold[node]:
                node = children_left[node]
            else:
                node = children_right[node]
        out[i] = value[node]
    return out


def _predict_rf(
    X: np.ndarray,
    tree_node_counts: np.ndarray,
    children_left: np.ndarray,
    children_right: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    value: np.ndarray,
) -> np.ndarray:
    """Predict using RF tree arrays; X (n_samples, n_features) -> (n_samples,)."""
    offsets = np.concatenate([[0], np.cumsum(tree_node_counts)])
    n_trees = len(tree_node_counts)
    preds = np.zeros((n_trees, X.shape[0]), dtype=np.float64)
    for t in range(n_trees):
        lo, hi = offsets[t], offsets[t + 1]
        preds[t] = _predict_tree(
            X,
            children_left[lo:hi],
            children_right[lo:hi],
            feature[lo:hi],
            threshold[lo:hi],
            value[lo:hi],
        )
    return np.mean(preds, axis=0)


class DiscoPredictor:
    """
    DISCO predictor: maps anchor-point prediction tensors to full-benchmark accuracy.

    Load from the Hub with:
        from transformers import AutoModel
        model = AutoModel.from_pretrained("<USERNAME>/my-disco-mmlu", trust_remote_code=True)

    Then call model.predict(predictions) where predictions has shape
    (n_models, n_anchor_points, n_classes) (e.g. log-probabilities per choice).
    Returns a 1D array of predicted full-benchmark accuracies, one per model.
    """

    def __init__(
        self,
        components: np.ndarray,
        mean: np.ndarray,
        tree_node_counts: np.ndarray,
        children_left: np.ndarray,
        children_right: np.ndarray,
        feature: np.ndarray,
        threshold: np.ndarray,
        value: np.ndarray,
        config: Optional[Any] = None,
    ):
        self._components = np.asarray(components, dtype=np.float64)
        self._mean = np.asarray(mean, dtype=np.float64)
        self._tree_node_counts = np.asarray(tree_node_counts, dtype=np.int64)
        self._children_left = np.asarray(children_left, dtype=np.int32)
        self._children_right = np.asarray(children_right, dtype=np.int32)
        self._feature = np.asarray(feature, dtype=np.int32)
        self._threshold = np.asarray(threshold, dtype=np.float64)
        self._value = np.asarray(value, dtype=np.float64)
        self.config = config

    @classmethod
    def register_for_auto_class(cls, auto_class=None):
        """No-op: required by transformers when loading custom models with trust_remote_code."""
        pass

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        **kwargs,
    ) -> "DiscoPredictor":
        """Load DISCO weights from a Hugging Face repo or local directory."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError("Loading from Hub requires huggingface_hub: pip install huggingface_hub") from e

        path = Path(pretrained_model_name_or_path)
        if not path.exists() or not path.is_dir():
            path = Path(snapshot_download(pretrained_model_name_or_path))

        # Load config if present
        config = None
        config_path = path / "config.json"
        if config_path.exists():
            try:
                from transformers import AutoConfig

                config = AutoConfig.from_pretrained(str(path), trust_remote_code=True)
            except Exception:
                pass

        # Load PCA (transform)
        transform_data = np.load(path / "disco_transform.npz")
        components = np.asarray(transform_data["components_"])
        mean = np.asarray(transform_data["mean_"])

        # Load RF (model)
        model_data = np.load(path / "disco_model.npz")
        tree_node_counts = np.asarray(model_data["tree_node_counts"], dtype=np.int64)
        children_left = np.asarray(model_data["children_left"], dtype=np.int32)
        children_right = np.asarray(model_data["children_right"], dtype=np.int32)
        feature = np.asarray(model_data["feature"], dtype=np.int32)
        threshold = np.asarray(model_data["threshold"], dtype=np.float64)
        value = np.asarray(model_data["value"], dtype=np.float64)

        return cls(
            components=components,
            mean=mean,
            tree_node_counts=tree_node_counts,
            children_left=children_left,
            children_right=children_right,
            feature=feature,
            threshold=threshold,
            value=value,
            config=config,
        )

    def predict(
        self,
        predictions: np.ndarray,
        apply_softmax: bool = True,
    ) -> np.ndarray:
        """
        Predict full-benchmark accuracy from anchor-point predictions.

        Args:
            predictions: Shape (n_models, n_anchor_points, n_classes), e.g. log-probabilities.
            apply_softmax: If True, apply softmax to predictions before PCA (default True).

        Returns:
            Shape (n_models,) predicted full-benchmark accuracies.
        """
        X = np.asarray(predictions, dtype=np.float64)
        if X.ndim == 2:
            X = X[np.newaxis, ...]
        n_models = X.shape[0]
        # Softmax over last dim
        if apply_softmax:
            X = np.exp(X - X.max(axis=-1, keepdims=True))
            X = X / X.sum(axis=-1, keepdims=True)
        # Flatten to (n_models, n_anchor_points * n_classes)
        X = X.reshape(n_models, -1)
        # PCA
        emb = _pca_transform(X, self._components, self._mean)
        # RF
        return _predict_rf(
            emb,
            self._tree_node_counts,
            self._children_left,
            self._children_right,
            self._feature,
            self._threshold,
            self._value,
        )

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Save DISCO weights and config to a directory (e.g. for uploading to Hub)."""
        from transformers import AutoConfig

        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(
            path / "disco_transform.npz",
            components_=self._components,
            mean_=self._mean,
        )
        np.savez(
            path / "disco_model.npz",
            tree_node_counts=self._tree_node_counts,
            children_left=self._children_left,
            children_right=self._children_right,
            feature=self._feature,
            threshold=self._threshold,
            value=self._value,
        )
        if self.config is not None:
            self.config.save_pretrained(save_directory)
