# Breast Cancer MLflow — End-to-End ML Experiment Tracking

A complete, production-style MLflow project built on the **Breast Cancer Wisconsin dataset**.  
This project walks through every major MLflow concept in a single, well-documented notebook — from first experiment to a registered, served model.

## 📂 Project Structure

```
breast-cancer-mlflow/
├── notebooks/
│   └── breast_cancer_mlflow.ipynb   ← main notebook (run this)
├── artifacts/                        ← saved plots & reports (git-ignored in practice)
├── src/
│   └── utils.py                      ← helper functions extracted from the notebook
├── requirements.txt
├── .gitignore
└── README.md
```

---

## What the Notebook Covers

| Section | Topic | Key MLflow APIs |
|---------|-------|-----------------|
| 0 | Setup & imports | — |
| 1 | MLflow tracking server config | `set_tracking_uri`, `set_experiment` |
| 2 | Data loading & EDA | — |
| 3 | Manual tracking (full control) | `log_params`, `log_metrics`, `log_artifact`, `set_tag` |
| 4 | Autologging | `mlflow.sklearn.autolog()` |
| 5 | Hyperparameter tuning + nested runs | `start_run(nested=True)` |
| 6 | Model Registry lifecycle | `register_model`, `transition_model_version_stage` |
| 7 | Loading from the Registry | `models:/name/Production` URI |
| 8 | REST serving & payload validation | `mlflow models serve`, `validate_serving_input` |
| 9 | Custom PyFunc model | `mlflow.pyfunc.PythonModel` |
| 10 | Programmatic run comparison | `MlflowClient`, `search_runs` |

---

## Dataset

**Breast Cancer Wisconsin (Diagnostic)** from `sklearn.datasets`

| Property | Value |
|----------|-------|
| Samples | 569 |
| Features | 30 (cell nucleus measurements) |
| Task | Binary classification |
| Classes | Malignant (0) / Benign (1) |
| Class balance | ~37% malignant, ~63% benign |

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/breast-cancer-mlflow.git
cd breast-cancer-mlflow
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start the MLflow tracking server
```bash
mlflow server --host 127.0.0.1 --port 5000
```
Keep this running in a separate terminal. Open **http://127.0.0.1:6001** in your browser to follow along visually.

### 4. Launch the notebook
```bash
jupyter notebook notebooks/breast_cancer_mlflow.ipynb
```

Run cells top to bottom — no configuration needed.

---

## Serving the Registered Model

After running the notebook, a Production model will be registered. Serve it as a REST API:

```bash
mlflow models serve \
  --model-uri "models:/breast-cancer-classifier/Production" \
  --port 5001 \
  --no-conda
```

Then call it:
```bash
curl http://127.0.0.1:6001/ping

curl -X POST http://127.0.0.1:6001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["mean radius", ...], "data": [[17.99, ...]]}}'
```

---

## 🔑 Key Concepts Demonstrated

**Manual vs Autologging** — The notebook shows both approaches side-by-side so you understand what autologging is actually doing under the hood.

**Nested Runs** — 10 hyperparameter trials across 3 model types (Logistic Regression, Random Forest, Gradient Boosting) are organised as parent → child runs. Easy to compare in the UI.

**Model Registry** — Models are registered and walked through `Staging → Production → Archived`. The production loading pattern (`models:/name/Production`) means code never needs updating when a new version is promoted.

**Custom PyFunc** — A custom wrapper class returns enriched predictions: class label, confidence score, malignancy probability, and a clinical risk level (Low / Medium / High).

---

## 📈 Results

Best model from the hyperparameter sweep (your results may vary slightly):

| Metric | Score |
|--------|-------|
| Accuracy | ~0.97 |
| ROC-AUC | ~0.997 |
| F1 | ~0.97 |
| Precision | ~0.97 |
| Recall | ~0.97 |

---

## Tech Stack

- **MLflow** 2.9+ — experiment tracking, model registry, serving
- **scikit-learn** — models & preprocessing
- **pandas / numpy** — data handling
- **matplotlib / seaborn** — visualisations

---

##  License

MIT
