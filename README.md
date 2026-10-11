# Breast Cancer Classifier API

A production-style machine learning service that serves a scikit-learn classification model through a REST API built with **FastAPI**. It is tested with **pytest** and packaged with **Docker**.

The project is also designed as a **reusable template** for serving other ML models.

---

## Features

- **Single pipeline artifact**: preprocessing (`StandardScaler`) and model (`LogisticRegression`) are saved together, preventing training–serving skew
- **Input validation** with Pydantic: malformed or incomplete requests are rejected with a clear `422` error before reaching the model
- **Model loaded once at startup** using FastAPI's lifespan, not on every request
- **Unit and API tests** with pytest and FastAPI's `TestClient`
- **Reproducible Docker image**: the model is trained during the image build, so the scikit-learn version used for training always matches the one used for serving
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
│   └── train.py         # Trains the pipeline and saves the artifact
├── models/              # Saved model artifacts (not tracked in git)
├── tests/
│   ├── test_model.py    # Unit tests for the prediction logic
│   └── test_api.py      # API tests for the endpoints
├── Dockerfile
├── pyproject.toml
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

**Target encoding:** `0` = malignant, `1` = benign (scikit-learn convention).

---

## Quickstart

### Option 1: Run with Docker (recommended)

```bash
docker build -t breast-cancer-api .
docker run -p 8000:8000 breast-cancer-api
```

### Option 2: Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python training/train.py         # trains and saves the model
uvicorn app.main:app --reload --app-dir src
```

Then open **http://127.0.0.1:8000/docs** to explore the API interactively.

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

## Running Tests

```bash
python -m pytest -v
```

The test suite covers:
- **Prediction logic**: output keys, correct class mapping on a known malignant case, valid probability range, and native Python types for JSON serialisation
- **API behaviour**: health check, a valid prediction request end to end, and `422` rejection of a request with a missing feature

---

## Design Decisions

- **Why save a pipeline instead of just the model?** The fitted scaler stores the training data's mean and standard deviation. Saving it with the model guarantees that every request is transformed exactly as the training data was.
- **Why explicit request fields instead of a generic dictionary?** Explicit fields let the API fail fast at the boundary, with a `422` naming the exact field at fault, rather than failing inside the model with an unhelpful `500`.
- **Why train inside the Docker build?** A `.joblib` artifact may not load correctly under a different scikit-learn version. Training in the image keeps both versions identical and lets the image be built from the repository alone. For larger models, the artifact would instead be versioned in a model registry.
