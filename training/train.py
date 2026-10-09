from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "classifier.joblib"

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
print(X.info())

def main():
    # 1. Load data as a DataFrame (keeps the feature names!)
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    # TODO 1: split into train/test (20% test, random_state=42, stratify on y)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

    # TODO 2: build a Pipeline with two steps: "scaler" and "model"
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression()),
    ])

    # TODO 3: fit the pipeline on train data, then print test accuracy
    pipeline = pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()