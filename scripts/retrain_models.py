"""
Retrain intent classifiers against the local scikit-learn version to fix
the pickle version mismatch (.joblib files trained on sklearn 1.8.0, this
venv has 1.7.2). Run from the backend directory.
"""

import json
import importlib
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

ml_classifiers = importlib.import_module("app.nlp.ml_classifiers")
TFIDFLogisticIntentModel = ml_classifiers.TFIDFLogisticIntentModel
TFIDFSVMIntentModel = ml_classifiers.TFIDFSVMIntentModel
SentenceTransformerIntentModel = ml_classifiers.SentenceTransformerIntentModel
MODELS_DIR = ml_classifiers.MODELS_DIR

# Data lives at repo root, not backend/ — adjust if your settings.ROOT_DIR differs
TRAIN_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "datasets" / "split_train.json"

def load_training_data():
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # ASSUMPTION — VERIFY BEFORE TRUSTING RESULTS:
    # combining subject + description with a space. If pipeline.py or
    # classify()'s caller combines them differently (e.g. "subject. description",
    # or passes them separately), this won't match inference-time behavior
    # and your retrained model will silently score worse than the original.
    texts = [f"{s['subject']} {s['description']}" for s in samples]
    labels = [s["expected_intent"] for s in samples]

    return texts, labels


def main():
    texts, labels = load_training_data()
    print(f"Loaded {len(texts)} training samples.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Training TFIDFLogisticIntentModel...")
    logistic = TFIDFLogisticIntentModel()
    logistic.fit(texts, labels)
    logistic.save(MODELS_DIR / "tfidf_logistic.joblib")

    print("Training TFIDFSVMIntentModel...")
    svm = TFIDFSVMIntentModel()
    svm.fit(texts, labels)
    svm.save(MODELS_DIR / "tfidf_svm.joblib")

    print("Training SentenceTransformerIntentModel...")
    st_model = SentenceTransformerIntentModel()
    st_model.fit(texts, labels)
    st_model.save(MODELS_DIR / "sentence_transformer.joblib")

    print("Done. All three models retrained and saved to", MODELS_DIR)


if __name__ == "__main__":
    main()