# MMLU Benchmark Example

This example demonstrates how to evaluate HuggingFace language models on MMLU (Massive Multitask Language Understanding) using MASEval, with optional anchor point-based task selection for DISCO prediction.

## Overview

The MMLU benchmark evaluates language models on multiple choice questions across 57 subjects including STEM, humanities, social sciences, and more. This implementation is compatible with the [disco-public](https://github.com/parameterlab/disco-public) evaluation methodology.

### Key Features

- **Anchor Point-Based Evaluation**: Evaluate only on selected anchor tasks for efficient DISCO-based performance prediction
- **Full Prompt Support**: Use few-shot examples from `full_prompt` field (like lm-evaluation-harness)
- **HuggingFace Integration**: Works with any HuggingFace transformers model
- **DISCO-Compatible Output**: Saves predictions in format compatible with DISCO predictor

## Installation

```bash
# Install MASEval with all extras (includes transformers)
pip install "maseval[all]"

# Or install with specific extras
pip install "maseval[transformers]"
```

## Data Format

The benchmark expects a JSON file in the `mmlu_prompts_examples.json` format:

```json
[
  {
    "query": "Question text with answer choices...\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer:",
    "full_prompt": "The following are multiple choice questions... [few-shot examples] ... [question]",
    "choices": ["A", "B", "C", "D"],
    "gold": 1
  },
  ...
]
```

## Usage

### Basic Evaluation

Evaluate a model on all MMLU tasks:

```bash
python mmlu_benchmark.py \
    --model_id "meta-llama/Llama-2-7b-hf" \
    --data_path /path/to/mmlu_prompts_examples.json
```

### Anchor Points Evaluation (for DISCO)

Evaluate only on anchor tasks for DISCO prediction:

```bash
python mmlu_benchmark.py \
    --model_id "alignment-handbook/zephyr-7b-sft-full" \
    --data_path /path/to/mmlu_prompts_examples.json \
    --anchor_points_path /path/to/anchor_points_disagreement.pkl \
    --use_full_prompt \
    --predictions_path ./output/predictions.pkl
```

### Quick Test

Run on a small subset for testing:

```bash
python mmlu_benchmark.py \
    --model_id "meta-llama/Llama-2-7b-hf" \
    --data_path /path/to/mmlu_prompts_examples.json \
    --limit 10
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_id` | HuggingFace model identifier (required) | - |
| `--data_path` | Path to MMLU prompts JSON file (required) | - |
| `--anchor_points_path` | Path to anchor points pickle file | None |
| `--output_dir` | Directory to save results | `./results` |
| `--predictions_path` | Path to save predictions pickle (for DISCO) | None |
| `--use_full_prompt` | Use full prompt with few-shot examples | False |
| `--limit` | Limit number of tasks to evaluate | None |
| `--device` | Device to run model on | `cuda:0` |
| `--trust_remote_code` | Trust remote code when loading model | False |
| `--num_workers` | Number of parallel workers | 1 |

## Equivalent disco-public Command

This MASEval benchmark provides equivalent functionality to:

```bash
python scripts/run_lm_eval.py \
    --anchor_points_path=/path/to/anchor_points_disagreement.pkl \
    --batch_size=8 --device=cuda:0 \
    --gen_kwargs=max_gen_toks=128,output_scores=True,return_dict_in_generate=True \
    --metric=acc_norm --model=hf \
    --model_args=pretrained=alignment-handbook/zephyr-7b-sft-full,trust_remote_code=True \
    --num_fewshot=0 \
    --output_path=/path/to/output \
    --tasks=mmlu_prompts --log_samples --force_recompute \
    --use_full_prompt --skip_non_anchor_points
```

## Programmatic Usage

```python
from maseval.benchmark.mmlu import (
    MMLUBenchmark,
    load_tasks,
    compute_benchmark_metrics,
)
from maseval.interface.inference import HuggingFaceModelAdapter


class MyMMLUBenchmark(MMLUBenchmark):
    def __init__(self, model_id: str, **kwargs):
        super().__init__(**kwargs)
        self._model_id = model_id
        self._pipeline = None

    def get_model_adapter(self, model_id: str, **kwargs):
        from transformers import pipeline

        if self._pipeline is None:
            self._pipeline = pipeline(
                "text-generation",
                model=self._model_id,
                device="cuda:0",
            )

        return HuggingFaceModelAdapter(
            model=self._pipeline,
            model_id=self._model_id,
        )


# Load tasks with anchor points filtering
tasks = load_tasks(
    data_path="/path/to/mmlu_prompts_examples.json",
    anchor_points_path="/path/to/anchor_points.pkl",
)

# Run benchmark
benchmark = MyMMLUBenchmark(
    model_id="meta-llama/Llama-2-7b-hf",
    use_full_prompt=True,
)
results = benchmark.run(tasks=tasks, agent_data={"model_id": "llama-7b"})

# Compute metrics
metrics = compute_benchmark_metrics(results)
print(f"Accuracy: {metrics['acc']:.4f}")
```

## Output Format

### Results (JSONL)

Each task result is saved in JSONL format:

```json
{
  "task_id": "mmlu_42",
  "status": "success",
  "eval": [
    {
      "acc": 1.0,
      "acc_norm": 1.0,
      "predicted": 1,
      "gold": 1,
      "correct": true,
      "doc_id": 42
    }
  ]
}
```

### Predictions (Pickle)

When `--predictions_path` is specified, saves a numpy array of shape `(1, n_questions, n_choices)` for use with DISCO predictor.

## Architecture

The MMLU benchmark implementation follows MASEval patterns:

- `MMLUBenchmark`: Main benchmark class (abstract, requires `get_model_adapter`)
- `MMLUEnvironment`: Simple environment holding task context (no tools needed)
- `MMLUEvaluator`: Evaluates model predictions against gold answers
- `AnchorPointsTaskQueue`: AdaptiveTaskQueue that iterates through anchor tasks
- `MMLUModelAgent`: Simple agent wrapper that forwards prompts to model

## References

- [MMLU Dataset](https://huggingface.co/datasets/cais/mmlu)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [disco-public](https://github.com/parameterlab/disco-public)
