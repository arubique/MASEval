"""
Extract numpy arrays from DISCO pickle files (fitted_weights.pkl, transform.pkl).

Run this once with the same environment that created the pickles (or ignore
InconsistentVersionWarning). The output .npz files can be loaded without
pickle, avoiding sklearn version mismatch warnings.

Usage:
    python extract_disco_weights.py \\
        --model_path /path/to/fitted_weights.pkl \\
        --transform_path /path/to/transform.pkl \\
        --output_dir /path/to/output

Output:
    {output_dir}/disco_transform.npz   - PCA: components_, mean_
    {output_dir}/disco_model.npz       - RandomForest: tree arrays
    {output_dir}/disco_meta.json       - sampling_name, number_item, fitted_model_type
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np


def _extract_pca(transform) -> dict:
    """Extract numpy arrays from a fitted sklearn PCA object."""
    return {
        "components_": np.asarray(transform.components_, dtype=np.float64),
        "mean_": np.asarray(transform.mean_, dtype=np.float64),
        "n_components_": int(transform.n_components_),
        "n_features_in_": int(transform.n_features_in_),
    }


def _extract_tree(tree) -> dict:
    """Extract numpy arrays from one DecisionTreeRegressor.tree_."""
    return {
        "children_left": np.asarray(tree.children_left, dtype=np.int32),
        "children_right": np.asarray(tree.children_right, dtype=np.int32),
        "feature": np.asarray(tree.feature, dtype=np.int32),
        "threshold": np.asarray(tree.threshold, dtype=np.float64),
        "value": np.asarray(tree.value, dtype=np.float64).reshape(-1),
    }


def _extract_rf(estimator) -> dict:
    """Extract numpy arrays from a RandomForestRegressor (all trees)."""
    trees = []
    for est in estimator.estimators_:
        trees.append(_extract_tree(est.tree_))
    n_trees = len(trees)
    # Concatenate per-tree arrays; store node count per tree for splitting
    node_counts = np.array([len(t["children_left"]) for t in trees], dtype=np.int64)
    return {
        "n_trees": n_trees,
        "tree_node_counts": node_counts,
        "children_left": np.concatenate([t["children_left"] for t in trees]),
        "children_right": np.concatenate([t["children_right"] for t in trees]),
        "feature": np.concatenate([t["feature"] for t in trees]),
        "threshold": np.concatenate([t["threshold"] for t in trees]),
        "value": np.concatenate([t["value"] for t in trees]),
    }


def _get_fitted_model(model_data: dict) -> tuple:
    """Return (fitted_model, sampling_name, number_item, fitted_model_type)."""
    if "fitted_weights" in model_data:
        fitted_weights = model_data["fitted_weights"]
    else:
        fitted_weights = {k: v for k, v in model_data.items() if k != "transform"}
    if not fitted_weights:
        raise ValueError("Could not find fitted_weights in model file.")
    sampling_name = list(fitted_weights.keys())[0]
    number_item = list(fitted_weights[sampling_name].keys())[0]
    fitted_model_type = list(fitted_weights[sampling_name][number_item].keys())[0]
    fitted_model = fitted_weights[sampling_name][number_item][fitted_model_type]
    return fitted_model, sampling_name, number_item, fitted_model_type


def main():
    parser = argparse.ArgumentParser(
        description="Extract numpy arrays from DISCO pickle files for pickle-free loading.",
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to fitted_weights.pkl")
    parser.add_argument("--transform_path", type=str, required=True, help="Path to transform.pkl")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to write disco_transform.npz and disco_model.npz",
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load transform (PCA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        with open(args.transform_path, "rb") as f:
            transform = pickle.load(f)
    pca_data = _extract_pca(transform)
    transform_out = out_dir / "disco_transform.npz"
    np.savez(transform_out, **pca_data)
    print(f"Saved PCA arrays to {transform_out}")

    # Load model and extract RF
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        with open(args.model_path, "rb") as f:
            model_data = pickle.load(f)
    fitted_model, sampling_name, number_item, fitted_model_type = _get_fitted_model(model_data)
    rf_data = _extract_rf(fitted_model)
    model_out = out_dir / "disco_model.npz"
    np.savez(model_out, **rf_data)
    print(f"Saved RF arrays to {model_out}")
    meta = {
        "sampling_name": str(sampling_name),
        "number_item": str(number_item),
        "fitted_model_type": str(fitted_model_type),
    }
    meta_out = out_dir / "disco_meta.json"
    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_out}")
    print(f"  sampling={sampling_name}, n_items={number_item}, model={fitted_model_type}")


if __name__ == "__main__":
    main()
