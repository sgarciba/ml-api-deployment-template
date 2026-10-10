import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
    

@pytest.fixture(scope="module")
def valid_payload():
    X, _ = load_breast_cancer(return_X_y=True, as_frame=True)
    return X.iloc[0].to_dict()   # keys like "mean radius", as a real client sends


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    
    
def test_predict_valid_request(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    assert set(response.json()) == {"prediction", "label", "probability", "model_version"}
    assert response.json()["label"] == "malignant"


def test_predict_missing_feature_returns_422(client, valid_payload):

    payload = valid_payload.copy()
    del payload["mean radius"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "mean radius" in response.text