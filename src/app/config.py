from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "classifier.joblib"
MODEL_VERSION = "0.1.0"