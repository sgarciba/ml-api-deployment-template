import joblib
import pandas as pd

from app.config import MODEL_PATH


def load_model():
    # 1: load the pipeline from MODEL_PATH and return it
    pipeline = joblib.load(MODEL_PATH)
    return pipeline


def predict(model, features: dict) -> dict:
    # 2: turn the features dict into a one-row DataFrame
    #         (hint: wrap the dict in a list)
    X = pd.DataFrame([features])
    
    
    # 3: get the predicted class and the probability of MALIGNANT
    y_pred = model.predict(X)[0]
    y_label = "malignant" if y_pred == 0 else "benign"
    y_prob = model.predict_proba(X)

    return {
        "prediction": int(y_pred),     # int: 0 or 1
        "label": y_label,          # "malignant" or "benign"
        "probability": float(y_prob[0,0]),    # float: probability of malignant
    }