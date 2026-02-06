"""
Build a directory ready to upload to Hugging Face as a dataset (e.g. MMLU prompts).

Usage:
    python build_data_repo.py \\
        --data_path /path/to/mmlu_prompts_examples.json \\
        --output_dir ./mmlu-prompts-examples

Then upload as a dataset:
    cd mmlu-prompts-examples
    huggingface-cli upload <USERNAME>/mmlu-prompts-examples . . --repo-type dataset

Download in code:
    from huggingface_hub import hf_hub_download
    data_path = hf_hub_download(
        repo_id="<USERNAME>/mmlu-prompts-examples",
        filename="mmlu_prompts_examples.json",
        repo_type="dataset",
    )
    # Then pass data_path to mmlu_benchmark.py --data_path
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Build HF dataset repo directory (e.g. for MMLU prompts JSON).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the prompts JSON file (e.g. mmlu_prompts_examples.json).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to create; will contain the JSON and a README.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="mmlu_prompts_examples.json",
        help="Filename to use in the repo (default: mmlu_prompts_examples.json).",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / args.filename
    shutil.copy2(data_path, out_file)
    print(f"Copied {data_path} -> {out_file}")

    readme = output_dir / "README.md"
    readme.write_text(
        f"""# MMLU prompts examples

Dataset of MMLU (Massive Multitask Language Understanding) prompts in the format expected by MASEval's MMLU benchmark.

## Data

- **{args.filename}** – JSON list of items with `query`, `full_prompt`, `choices`, `gold`, and optional `example`.

## Use with MASEval

Download the file and pass it to the benchmark:

```python
from huggingface_hub import hf_hub_download

data_path = hf_hub_download(
    repo_id="<USERNAME>/mmlu-prompts-examples",
    filename="{args.filename}",
    repo_type="dataset",
)
# Then: python mmlu_benchmark.py --data_path <data_path> ...
```

Or download once to a local path and use that as `--data_path`.
""",
        encoding="utf-8",
    )
    print(f"Wrote {readme}")

    print(f"\nRepo ready at: {output_dir.resolve()}")
    print("Upload with: huggingface-cli upload <USERNAME>/mmlu-prompts-examples . . --repo-type dataset")


if __name__ == "__main__":
    main()
