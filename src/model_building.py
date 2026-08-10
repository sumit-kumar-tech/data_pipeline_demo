from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import yaml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# Determine project base directory using pathlib
base_dir = Path(__file__).resolve().parent.parent
params_path = base_dir / "params.yaml"

# Load parameters from params.yaml
with open(params_path, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)["model_building"]

# Extract model hyperparameters
dt_params = params["decision_tree"]
rf_params = params["random_forest"]
xgb_params = params["xgboost"]

# Input feature datasets
features_dir = base_dir / "data" / "features"
train_features_path = features_dir / "train_features.csv"
test_features_path = features_dir / "test_features.csv"

# Load feature datasets
train_df = pd.read_csv(train_features_path)
test_df = pd.read_csv(test_features_path)

# Separate features X and binary target y
feature_cols = ["weekly_self_study_hours", "attendance_percentage", "class_participation", "study_participation_ratio"]
target_col = "grade_encoded"

X_train, y_train = train_df[feature_cols], train_df[target_col]
X_test, y_test = test_df[feature_cols], test_df[target_col]

# Define classification models using parameters from params.yaml
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=dt_params["max_depth"], 
        random_state=dt_params["random_state"]
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=rf_params["n_estimators"], 
        max_depth=rf_params["max_depth"], 
        random_state=rf_params["random_state"]
    ),
    "XGBoost": XGBClassifier(
        n_estimators=xgb_params["n_estimators"], 
        max_depth=xgb_params["max_depth"], 
        learning_rate=xgb_params["learning_rate"], 
        random_state=xgb_params["random_state"], 
        eval_metric="logloss"
    )
}

# Output directory for saved trained models
models_dir = base_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

metrics_summary = []

print("=== Starting Model Building & Training (using params.yaml) ===")
for model_name, model in models.items():
    # Fit model
    model.fit(X_train, y_train)
    
    # Predict labels and probability scores
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary")
    rec = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    roc = roc_auc_score(y_test, y_proba)
    
    metrics_summary.append({
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(roc, 4)
    })
    
    # Save model binary using pickle
    model_filename = model_name.lower().replace(" ", "_") + ".pkl"
    model_filepath = models_dir / model_filename
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)
        
    print(f"\n--- {model_name} Metrics ---")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["A", "B"]))

# Summary DataFrame
summary_df = pd.DataFrame(metrics_summary)
print("\n=== Final Model Comparison Summary ===")
print(summary_df.to_string(index=False))

print(f"\nAll models saved successfully to: {models_dir}")
