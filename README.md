# STF-TELMo-example

Companion example code for **STF-TELMo** — a deep learning architecture for multivariate time-series forecasting.  
This repository contains an anonymized, runnable example split across two main scripts:

- `Model.py` — training pipeline (data load, preprocess, sequence creation, model definition, training, save model & scalers)  
- `Generate_Forecasts.py` — prediction-only script (loads saved model & scalers, prepares new input sequences, runs inference, saves timestamped step-wise predictions)

---

## Repository layout (recommended)

```
.
├─ README.md
├─ requirements.txt
├─ Model.py                 # training pipeline (anonymized)
├─ Generate_Forecasts.py    # prediction-only script (stand-alone)

```

---

## Quick usage

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# .\venv\Scripts\activate      # Windows PowerShell
pip install -r requirements.txt
```

`requirements.txt` (example):

```
tensorflow
scikit-learn
pandas
numpy
joblib
```


### 3. Train the example model (`Model.py`)

`Model.py` is the anonymized training script. Example usage:

```bash
python Model.py \
  --data_dir data/ \
  --output_dir artifacts/ \
  --sequence_length 2 \
  --epochs 20 \
  --batch_size 128
```

After successful training you should find:

- `artifacts/model.h5` — saved Keras model  
- `artifacts/scaler_features.pkl` — feature scaler (joblib)  
- `artifacts/scaler_targets.pkl` — target scaler (joblib)

---

### 4. Generate forecasts (`Generate_Forecasts.py`)

Use `Generate_Forecasts.py` to run inference with a saved model and scalers. Example:

```bash
python Generate_Forecasts.py \
  --model artifacts/model.h5 \
  --scaler_features artifacts/scaler_features.pkl \
  --scaler_targets artifacts/scaler_targets.pkl \
  --input_csv data/new_input.csv \
  --output_csv results/new_predictions.csv \
  --sequence_length 2
```

Expectations for `Generate_Forecasts.py`:

- Loads the Keras model saved via `model.save(...)`.
- Loads `scaler_features` and `scaler_targets` saved with `joblib.dump(...)`.
- Reads `input_csv` which **must** contain a `timestamp` column and the same feature columns used during training (or adjust `feature_columns` inside the script to match).
- Produces `results/new_predictions.csv` containing columns:

```
timestamp, pred_u_step1, pred_v_step1, pred_u_step2, pred_v_step2, ...
```

Timestamps correspond to the last timestamp of each input sequence (i.e., the time the forecast is made for).

---

## Notes & best practices

- The example is intentionally anonymized — replace placeholder column names and file paths before using with private/real datasets.
- Ensure the exact same preprocessing (feature ordering, scaling, engineered columns) is applied at training and inference.
- For reproducibility, pin dependency versions in `requirements.txt` or provide a `Dockerfile`/`environment.yml`.
- For real experiments, increase epochs and perform hyperparameter tuning; log metrics (e.g., with TensorBoard or MLflow).

---

## Citation & license

If you use STF-TELMo or this code in published work, please cite the associated paper:

```
xxxx
```

This example is released under the **MIT License**. See `LICENSE` for details.

---

## Contact

Found an issue or want to suggest improvements? Open an issue or pull request on GitHub.  
Author: *Your Name* (replace with preferred contact)
