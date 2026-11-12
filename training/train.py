import argparse
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "classifier.joblib"

TEST_SIZE = 0.2
RANDOM_STATE = 42

MODELS = {
    "logreg": LogisticRegression(),
    "hgb": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), default="logreg")
    parser.add_argument("--track", action="store_true", help="Log this run to MLflow")
    return parser.parse_args()


def log_to_mlflow(pipeline, args, accuracy, recall):
    import mlflow
    import mlflow.sklearn

    mlflow.set_experiment("breast-cancer")

    with mlflow.start_run():
        model_params = pipeline.named_steps["model"].get_params()
        mlflow.log_params({f"model__{k}": v for k, v in model_params.items()})
        mlflow.log_params(
            {
                "model_type": args.model,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            }
        )
        mlflow.log_metrics({"accuracy": accuracy, "recall": recall})
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            skops_trusted_types=[
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
            ],
        )


def main():

    args = parse_args()

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", MODELS[args.model]),
        ]
    )

    pipeline = pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, pos_label=0)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Recall (malignant): {recall:.3f}")

    if args.track:
        log_to_mlflow(pipeline, args, accuracy, recall)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
