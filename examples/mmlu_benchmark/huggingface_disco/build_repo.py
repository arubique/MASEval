"""
Build a directory ready to upload to Hugging Face as a DISCO predictor model.

Usage:
    python build_repo.py \\
        --weights_dir /path/to/disco_weights \\
        --output_dir ./my-disco-mmlu

    # Include anchor points and eval config (pca, pad_to_size, use_lmeval_batching)
    python build_repo.py \\
        --weights_dir /path/to/disco_weights \\
        --output_dir ./my-disco-mmlu \\
        --anchor_points_path /path/to/anchor_points_disagreement.pkl \\
        --pca 256 --pad_to_size 31 --use_lmeval_batching

Stored eval_config is applied by the benchmark when loading the model from HF if the user did not override those args.

Then upload with: huggingface-cli upload <USERNAME>/my-disco-mmlu ./my-disco-mmlu .
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path


def _load_anchor_points(path: Path) -> list:
    """Load anchor points from .pkl or .json; return a list."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path) as f:
            out = json.load(f)
    else:
        with open(path, "rb") as f:
            out = pickle.load(f)
    if hasattr(out, "tolist"):
        out = out.tolist()
    return list(out)


def main():
    parser = argparse.ArgumentParser(
        description="Build HF repo directory for DISCO predictor (npz + config + code).",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        required=True,
        help="Directory containing disco_transform.npz, disco_model.npz, disco_meta.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to create (will contain config, code, and weights).",
    )
    parser.add_argument(
        "--anchor_points_path",
        type=str,
        default=None,
        help="Path to anchor points .pkl or .json; saved as anchor_points.json in the repo.",
    )
    parser.add_argument(
        "--pca",
        type=int,
        default=None,
        help="PCA dimension used for this model; stored in eval_config so the benchmark can use it when loading from HF.",
    )
    parser.add_argument(
        "--pad_to_size",
        type=int,
        default=None,
        help="Pad predictions to this size; stored in eval_config for the benchmark when loading from HF.",
    )
    parser.add_argument(
        "--use_lmeval_batching",
        action="store_true",
        help="Whether lm-eval batching was used; stored in eval_config for the benchmark when loading from HF.",
    )
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Required weight files
    for name in ("disco_transform.npz", "disco_model.npz", "disco_meta.json"):
        src = weights_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing {name} in {weights_dir}. Run extract_disco_weights.py first.")
        shutil.copy2(src, output_dir / name)
    print(f"Copied weights from {weights_dir} -> {output_dir}")

    # Copy modeling and config code (so repo is self-contained)
    this_dir = Path(__file__).resolve().parent
    for name in ("configuration_disco.py", "modeling_disco.py"):
        shutil.copy2(this_dir / name, output_dir / name)
        print(f"Copied {name}")

    # config.json with auto_map for AutoModel.from_pretrained(..., trust_remote_code=True)
    meta_path = output_dir / "disco_meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    config = {
        "model_type": "disco",
        "auto_map": {
            "AutoConfig": "configuration_disco.DiscoConfig",
            "AutoModel": "modeling_disco.DiscoPredictor",
        },
        "n_components": 256,
        "sampling_name": meta.get("sampling_name", ""),
        "number_item": meta.get("number_item", ""),
        "fitted_model_type": meta.get("fitted_model_type", ""),
    }
    # Eval config: when the benchmark loads this model from HF, it can apply these if the user did not override
    eval_config = {}
    if args.pca is not None:
        eval_config["pca"] = args.pca
    if args.pad_to_size is not None:
        eval_config["pad_to_size"] = args.pad_to_size
    if args.use_lmeval_batching:
        eval_config["use_lmeval_batching"] = True
    if eval_config:
        config["eval_config"] = eval_config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {config_path}")

    # Optional: anchor points so users can omit --anchor_points_path when loading from HF
    if args.anchor_points_path:
        ap_path = Path(args.anchor_points_path)
        if not ap_path.exists():
            raise FileNotFoundError(f"Anchor points file not found: {ap_path}")
        anchor_points = _load_anchor_points(ap_path)
        ap_out = output_dir / "anchor_points.json"
        with open(ap_out, "w") as f:
            json.dump(anchor_points, f, indent=2)
        print(f"Wrote {ap_out} ({len(anchor_points)} anchor points)")

    print(f"\nRepo ready at: {output_dir.resolve()}")
    print("Upload with: huggingface-cli upload <USERNAME>/my-disco-mmlu ./my-disco-mmlu .")


if __name__ == "__main__":
    main()
