from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yaml

# Determine project base directory using pathlib
base_dir = Path(__file__).resolve().parent.parent
params_path = base_dir / "params.yaml"

# Load parameters from params.yaml
with open(params_path, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)["feature_engineering"]

target_pos_class = params.get("target_positive_class", "B")

# Input paths (processed clean datasets from data/processed/)
processed_dir = base_dir / "data" / "processed"
train_proc_path = processed_dir / "train_processed.csv"
test_proc_path = processed_dir / "test_processed.csv"

# Load processed datasets
train_df = pd.read_csv(train_proc_path)
test_df = pd.read_csv(test_proc_path)

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features and encode binary target variable using params.yaml."""
    df_feat = df.copy()
    
    # Feature Engineering: Interaction term between self study hours and class participation
    df_feat["study_participation_ratio"] = df_feat["weekly_self_study_hours"] * df_feat["class_participation"]
    
    # Binary Target Encoding: Map target_pos_class to 1 and all others to 0
    if "grade" in df_feat.columns:
        df_feat["grade_encoded"] = np.where(df_feat["grade"] == target_pos_class, 1, 0)
        df_feat = df_feat.drop(columns=["grade"])
        
    return df_feat

# Apply feature creation to train and test sets
train_feat_df = create_features(train_df)
test_feat_df = create_features(test_df)

# Separate features X and target y
feature_cols = ["weekly_self_study_hours", "attendance_percentage", "class_participation", "study_participation_ratio"]
target_col = "grade_encoded"

X_train = train_feat_df[feature_cols]
y_train = train_feat_df[target_col]

X_test = test_feat_df[feature_cols]
y_test = test_feat_df[target_col]

# Apply StandardScaler to feature columns
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Assemble final model-ready DataFrames
train_features_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
train_features_df[target_col] = y_train.values

test_features_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
test_features_df[target_col] = y_test.values

# Output directory for final model-ready features
features_dir = base_dir / "data" / "features"
features_dir.mkdir(parents=True, exist_ok=True)

# Save final feature datasets
train_features_df.to_csv(features_dir / "train_features.csv", index=False)
test_features_df.to_csv(features_dir / "test_features.csv", index=False)

print(f"Feature engineering complete (positive class '{target_pos_class}'). Saved final feature datasets to: {features_dir}")
