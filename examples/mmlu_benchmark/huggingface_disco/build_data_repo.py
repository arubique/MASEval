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
        repo_id="arubique/flattened-MMLU",
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
        f"""# MMLU prompts (flattened)

This dataset is a flattened reformatting of the original [MMLU](https://huggingface.co/datasets/cais/mmlu) benchmark, in the format expected by [MASEval](https://github.com/parameterlab/MASEval)'s MMLU benchmark with support for accelerated [DISCO](https://arubique.github.io/disco-site/) evaluation.

## What this dataset is

- **{args.filename}** – JSON list of items with `query`, `full_prompt`, `choices`, `gold`, and optional `example`.
- A flattened structure suitable for anchor-point evaluation and DISCO prediction pipelines.

### What "flattened" means

The original MMLU has 57 subject categories (anatomy, abstract algebra, virology, etc.). This dataset drops the per-category structure and concatenates all questions into a single ordered list of ~14k questions, preserving a fixed ordering for reproducible evaluation.

### The `full_prompt` field

For each question, we randomly select 10 few-shot examples from the same category. The `full_prompt` field includes the question along with these in-context few-shot examples. This design ensures evaluation reproducibility by fixing the few-shot examples for every run.

## Attribution / Acknowledgment

This dataset is derived from the original **MMLU (Massive Multitask Language Understanding)** benchmark:

- **Original authors:** Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt (ICLR 2021).
- **Original paper:** ["Measuring Massive Multitask Language Understanding"](https://arxiv.org/abs/2009.03300) (ICLR 2021).
- **Source dataset:** [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) on Hugging Face.
- **License:** MIT License (inherited from cais/mmlu).

## Use with MASEval

Download the file and pass it to the benchmark:

```python
from huggingface_hub import hf_hub_download

data_path = hf_hub_download(
    repo_id="arubique/flattened-MMLU",
    filename="mmlu_prompts_examples.json",
    repo_type="dataset",
)
# Then: python mmlu_benchmark.py --data_path <data_path> ...
```

Or download once to a local path and use that as `--data_path`.

## License

This dataset includes material derived from the **cais/mmlu** dataset (MIT License).
Original work by: Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt (ICLR 2021).

MIT License terms are provided in the `LICENSE` file.
""",
        encoding="utf-8",
    )
    print(f"Wrote {readme}")

    # Include MIT License from cais/mmlu for compliance
    license_text = """MIT License

Copyright (c) 2020 Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    license_path = output_dir / "LICENSE"
    license_path.write_text(license_text, encoding="utf-8")
    print(f"Wrote {license_path}")

    print(f"\nRepo ready at: {output_dir.resolve()}")
    print("Upload with: huggingface-cli upload <USERNAME>/mmlu-prompts-examples . . --repo-type dataset")


if __name__ == "__main__":
    main()
