import pytest
from sklearn.datasets import load_breast_cancer

from app.model import load_model, predict


@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.fixture(scope="module")
def malignant_sample():
    X, _ = load_breast_cancer(return_X_y=True, as_frame=True)
    return X.iloc[0].to_dict()


def test_predict_returns_expected_keys(model, malignant_sample):
    result = predict(model, malignant_sample)
    assert set(result) == {"prediction", "label", "probability"}
    
    
def test_known_malignant_case(model, malignant_sample):
    result = predict(model, malignant_sample)
    assert result['label'] == "malignant"


def test_probability_is_valid_float(model, malignant_sample):
    result = predict(model, malignant_sample)
    assert isinstance(result['probability'], float)
    assert 0 <= result['probability'] <= 1

def test_prediction_is_python_int(model, malignant_sample):
    result = predict(model, malignant_sample)
    assert isinstance(result['prediction'], int)
    