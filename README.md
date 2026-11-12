
# Breast Cancer Classifier API
 
![CI](https://github.com/sgarciba/ml-api-deployment-template/actions/workflows/ci.yml/badge.svg)
 
🚀 **Live API:** https://ml-api-deployment-template.onrender.com/docs
 
A production-style machine learning service that serves a scikit-learn classification model through a REST API built with **FastAPI**. It is tested with **pytest**, linted with **Ruff**, packaged with **Docker**, built and checked by **GitHub Actions**, deployed on **Render**, and tracked with **MLflow**.
 
The project is also designed as a **reusable template** for serving other ML models. It has been reused for a [California house price regression service](https://github.com/sgarciba/ml-api-deployment-regression).
 
---
 
## Features
 
- **Single pipeline artifact**: preprocessing (`StandardScaler`) and model (`LogisticRegression`) are saved together, preventing training–serving skew
- **Input validation** with Pydantic: malformed or incomplete requests are rejected with a clear `422` error before reaching the model
- **Model loaded once at startup** using FastAPI's lifespan, not on every request
- **Unit and API tests** with pytest and FastAPI's `TestClient`
- **Code quality checks** with Ruff (linting and formatting)
- **CI pipeline** with GitHub Actions: lint, test and Docker build on every push
- **Reproducible Docker image**: the model is trained during the image build, so the scikit-learn version used for training always matches the one used for serving
- **Live deployment** on Render, redeployed automatically on every push to `main`
- **Optional experiment tracking** with MLflow, logging parameters, metrics and model artifacts
- **Interactive API docs**, generated automatically at `/docs`
---
 
## Project Structure
 
```
├── src/app/
│   ├── main.py          # FastAPI app: /health and /predict endpoints
│   ├── schemas.py       # Pydantic request/response models
│   ├── model.py         # Model loading and prediction logic
│   └── config.py        # Paths and model version
├── training/
│   └── train.py         # Trains the pipeline, optional MLflow tracking
├── models/              # Saved model artifacts (not tracked in git)
├── tests/
│   ├── test_model.py    # Unit tests for the prediction logic
│   └── test_api.py      # API tests for the endpoints
├── .github/workflows/
│   └── ci.yml           # CI pipeline: lint, test, build
├── Dockerfile
├── pyproject.toml       # pytest and Ruff configuration
└── requirements.txt
```
 
---
 
## Model
 
| | |
|---|---|
| **Dataset** | Breast Cancer Wisconsin (scikit-learn), 569 samples, 30 features |
| **Pipeline** | `StandardScaler` → `LogisticRegression` |
| **Split** | 80/20, stratified on the target |
| **Test accuracy** | 0.982 |
| **Recall (malignant)** | 0.976 |
 
**Target encoding:** `0` = malignant, `1` = benign (scikit-learn convention).
 
**Why recall?** In cancer diagnosis, missing a malignant tumour (a false negative) is far more costly than a false alarm. Recall on the malignant class measures the share of malignant tumours the model catches, so it is reported alongside accuracy.
 
---
 
## Quickstart
 
### Option 1: Run with Docker (recommended)
 
```bash
docker build -t breast-cancer-api .
docker run -p 8000:8000 breast-cancer-api
```
 
### Option 2: Run locally (Python 3.11)
 
```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
 
python training/train.py         # trains and saves the model
uvicorn app.main:app --reload --app-dir src
```
 
Then open **http://127.0.0.1:8000/docs** to explore the API interactively.
 
The live API is hosted on Render's free tier, so the first request after a period of inactivity may take up to a minute while the service wakes up.
 
---
 
## API Usage
 
### `GET /health`
 
Checks that the service is running.
 
```json
{ "status": "ok" }
```
 
### `POST /predict`
 
Accepts the 30 tumour features and returns the predicted class, its label, and the probability that the tumour is malignant.
 
**Request** (truncated; all 30 features are required):
 
```json
{
  "mean radius": 17.99,
  "mean texture": 10.38,
  "...": "..."
}
```
 
**Response:**
 
```json
{
  "prediction": 0,
  "label": "malignant",
  "probability": 0.9999999938324466,
  "model_version": "0.1.0"
}
```
 
If a feature is missing or has the wrong type, the API returns `422 Unprocessable Entity` with details of the invalid field.
 
---
 
## Testing and Code Quality
 
```bash
python -m pytest -v
ruff check .
ruff format --check .
```
 
The test suite covers:
- **Prediction logic**: output keys, correct class mapping on a known malignant case, valid probability range, and native Python types for JSON serialisation
- **API behaviour**: health check, a valid prediction request end to end, and `422` rejection of a request with a missing feature
The same checks run automatically in CI on every push. The Docker image is only built once linting and tests have passed.
 
---
 
## Experiment Tracking
 
Training runs can be tracked with **MLflow**. Each tracked run logs its parameters (including all model settings), its metrics (accuracy and malignant recall) and the trained model artifact.
 
Tracking is optional, so the Docker build and CI run without MLflow installed:
 
```bash
pip install mlflow
python training/train.py --model logreg --track
python training/train.py --model hgb --track
mlflow ui --port 5001
```
 
Available models are defined in the `MODELS` dictionary in `train.py`. Without `--model`, the production model (`logreg`) is trained.
 
---
 
## Design Decisions
 
- **Why save a pipeline instead of just the model?** The fitted scaler stores the training data's mean and standard deviation. Saving it with the model guarantees that every request is transformed exactly as the training data was.
- **Why explicit request fields instead of a generic dictionary?** Explicit fields let the API fail fast at the boundary, with a `422` naming the exact field at fault, rather than failing inside the model with an unhelpful `500`.
- **Why train inside the Docker build?** A `.joblib` artifact may not load correctly under a different scikit-learn version. Training in the image keeps both versions identical and lets the image be built from the repository alone. For larger models, the artifact would instead be versioned in a model registry.
- **Why is MLflow optional?** MLflow is imported only when `--track` is used, so production builds stay lightweight while experiments remain fully traceable.
- **Why prefix model parameters in MLflow?** Model settings are logged as `model__<name>`, so they never clash with pipeline settings of the same name, such as the split's `random_state`.
---
 
## Using This as a Template
 
The project separates **reusable infrastructure** from **model-specific code**, so a new model can be served by changing only a handful of files.
 
**Reusable (unchanged between models):**
- `src/app/main.py`: API endpoints, lifespan and model loading (only the app title changes)
- `Dockerfile`, `.dockerignore`, `pyproject.toml`, `.github/workflows/ci.yml`
- The structure of the tests: fixtures, `TestClient` setup and the 422 validation pattern
- The training structure: `MODELS` dictionary, command-line options and optional MLflow tracking
**Model-specific (change for each new model):**
 
| File | What to change |
|---|---|
| `training/train.py` | Dataset, candidate models, evaluation metrics, experiment name |
| `src/app/config.py` | Artifact filename, model version |
| `src/app/model.py` | How the raw prediction is turned into the response |
| `src/app/schemas.py` | Input features and output fields |
| `tests/` | Sample data and model-specific assertions |
 
**Steps to create a new service:**
1. Click **"Use this template"** on GitHub to create a new repository.
2. Update the model-specific files listed above.
3. Run `python training/train.py`, then `python -m pytest -v` and `ruff check .`.
4. Build and run the Docker image, then push to trigger CI.