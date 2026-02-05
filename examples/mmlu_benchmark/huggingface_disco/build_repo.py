"""
Build a directory ready to upload to Hugging Face as a DISCO predictor model.

Usage:
    python build_repo.py \\
        --weights_dir /path/to/disco_weights \\
        --output_dir ./my-disco-mmlu

Then upload with: huggingface-cli upload <USERNAME>/my-disco-mmlu ./my-disco-mmlu .
"""

import argparse
import json
import shutil
from pathlib import Path


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
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {config_path}")

    print(f"\nRepo ready at: {output_dir.resolve()}")
    print("Upload with: huggingface-cli upload <USERNAME>/my-disco-mmlu ./my-disco-mmlu .")


if __name__ == "__main__":
    main()
