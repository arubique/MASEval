# MMLU prompts examples

Dataset of MMLU (Massive Multitask Language Understanding) prompts in the format expected by MASEval's MMLU benchmark.

## Data

- **mmlu_prompts_examples.json** – JSON list of items with `query`, `full_prompt`, `choices`, `gold`, and optional `example`.

## Use with MASEval

Download the file and pass it to the benchmark:

```python
from huggingface_hub import hf_hub_download

data_path = hf_hub_download(
    repo_id="<USERNAME>/mmlu-prompts-examples",
    filename="mmlu_prompts_examples.json",
    repo_type="dataset",
)
# Then: python mmlu_benchmark.py --data_path <data_path> ...
```

Or download once to a local path and use that as `--data_path`.
