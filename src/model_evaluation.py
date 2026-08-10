from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Determine project base directory using pathlib
base_dir = Path(__file__).resolve().parent.parent

# Input paths
test_features_path = base_dir / "data" / "features" / "test_features.csv"
models_dir = base_dir / "models"

# Load test feature dataset
test_df = pd.read_csv(test_features_path)

# Separate features X_test and target y_test
feature_cols = ["weekly_self_study_hours", "attendance_percentage", "class_participation", "study_participation_ratio"]
target_col = "grade_encoded"

X_test = test_df[feature_cols]
y_test = test_df[target_col]

# Define model names and expected model file paths
model_files = {
    "Decision Tree": models_dir / "decision_tree.pkl",
    "Random Forest": models_dir / "random_forest.pkl",
    "XGBoost": models_dir / "xgboost.pkl"
}

metrics_summary = {}

print("=== Starting Model Evaluation Stage ===")

for model_name, model_path in model_files.items():
    if not model_path.exists():
        print(f"Warning: Model file not found for {model_name} at {model_path}")
        continue
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    # Predict labels and probability scores
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate evaluation metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="binary"))
    rec = float(recall_score(y_test, y_pred, average="binary"))
    f1 = float(f1_score(y_test, y_pred, average="binary"))
    roc = float(roc_auc_score(y_test, y_proba))
    
    metrics_summary[model_name] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc, 4)
    }
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["A", "B"]))

# Save evaluation metrics to JSON inside result directory for DVC tracking
result_dir = base_dir / "result"
result_dir.mkdir(parents=True, exist_ok=True)
metrics_file = result_dir / "metrics.json"

with open(metrics_file, "w", encoding="utf-8") as f:
    json.dump(metrics_summary, f, indent=4)

print(f"\nModel evaluation metrics saved to: {metrics_file}")

