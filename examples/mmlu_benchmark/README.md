# MMLU Benchmark Example

Evaluate language models on MMLU (Massive Multitask Language Understanding) with optional efficient evaluation via DISCO.

## Run without DISCO (full evaluation)

From the project root:

```bash
uv run python examples/mmlu_benchmark/mmlu_benchmark.py --model_id alignment-handbook/zephyr-7b-sft-full
```

Full evaluation results look like:

```
================================================================================
Results Summary (Evaluated Tasks)
================================================================================
Total tasks: 100
Correct: 35
Accuracy (on anchor points): 0.3500
Accuracy norm (on anchor points): 0.3500
Built predictions tensor with shape: (1, 100, 31)
```

## Run with DISCO (predicted full-benchmark score)

From the project root:

```bash
uv run python examples/mmlu_benchmark/mmlu_benchmark.py --model_id alignment-handbook/zephyr-7b-sft-full --disco_model_path arubique/DISCO-MMLU
```

Predicted score output:

```
----------------------------------------
DISCO Predicted Full Benchmark Accuracy:
----------------------------------------
  Model 0: 0.606739
```
