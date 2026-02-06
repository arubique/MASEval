# DISCO predictor as a Hugging Face model

This folder contains the code and layout to upload a DISCO predictor (PCA + Random Forest) to the Hugging Face Hub so it can be loaded with:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("<USERNAME>/my-disco-mmlu", trust_remote_code=True)
# predictions: (n_models, n_anchor_points, n_classes) e.g. log-probabilities
acc = model.predict(predictions)
# acc: (n_models,) predicted full-benchmark accuracies
```

## 1. Extract weights to NumPy (no pickle)

From the MASEval repo, run the extractor (once) to produce `disco_transform.npz`, `disco_model.npz`, and `disco_meta.json`:

```bash
python examples/mmlu_benchmark/extract_disco_weights.py \
  --model_path /path/to/fitted_weights.pkl \
  --transform_path /path/to/transform.pkl \
  --output_dir /path/to/disco_weights
```

## 2. Build the Hugging Face repo

Run the build script to create a directory you can upload:

```bash
python examples/mmlu_benchmark/huggingface_disco/build_repo.py \
  --weights_dir /path/to/disco_weights \
  --output_dir ./my-disco-mmlu
```

To include anchor points so users can run the benchmark without `--anchor_points_path`:

```bash
python examples/mmlu_benchmark/huggingface_disco/build_repo.py \
  --weights_dir /path/to/disco_weights \
  --output_dir ./my-disco-mmlu \
  --anchor_points_path /path/to/anchor_points_disagreement.pkl
```

This copies the NumPy weights and writes `config.json` with `auto_map` so `AutoModel.from_pretrained(..., trust_remote_code=True)` loads `DiscoPredictor`.

## 3. Upload to the Hub

```bash
cd my-disco-mmlu
huggingface-cli upload <USERNAME>/my-disco-mmlu . .
```

Or from Python:

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path="./my-disco-mmlu", repo_id="<USERNAME>/my-disco-mmlu", repo_type="model")
```

## Repo contents

- `config.json` – `model_type: "disco"` and `auto_map` for AutoConfig / AutoModel
- `configuration_disco.py` – `DiscoConfig`
- `modeling_disco.py` – `DiscoPredictor` (loads npz, implements `predict`)
- `disco_transform.npz` – PCA components and mean
- `disco_model.npz` – Random Forest tree arrays
- `disco_meta.json` – sampling_name, number_item, fitted_model_type (optional, for display)
- `anchor_points.json` – list of anchor doc_ids (optional; add with `--anchor_points_path` so users can omit `--anchor_points_path` when running the benchmark)

## Dependencies

Loading from the Hub only needs:

- `transformers`
- `huggingface_hub`
- `numpy`

No `scikit-learn` or pickle at runtime.
