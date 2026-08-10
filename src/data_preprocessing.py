from pathlib import Path
import pandas as pd

# Determine project base directory using pathlib
base_dir = Path(__file__).resolve().parent.parent

# Input paths (raw train and test data from data/raw/)
raw_dir = base_dir / "data" / "raw"
train_raw_path = raw_dir / "train.csv"
test_raw_path = raw_dir / "test.csv"

# Load raw split datasets
train_df = pd.read_csv(train_raw_path)
test_df = pd.read_csv(test_raw_path)

# Preprocessing Step 1: Remove duplicate rows if any exist
train_df = train_df.drop_duplicates()
test_df = test_df.drop_duplicates()

# Preprocessing Step 2: Handle missing values (impute with training set feature means)
numeric_cols = ["weekly_self_study_hours", "attendance_percentage", "class_participation"]
train_means = train_df[numeric_cols].mean()
train_df[numeric_cols] = train_df[numeric_cols].fillna(train_means)
test_df[numeric_cols] = test_df[numeric_cols].fillna(train_means)

# Output directory for processed (clean) datasets
processed_dir = base_dir / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# Save processed clean datasets
train_df.to_csv(processed_dir / "train_processed.csv", index=False)
test_df.to_csv(processed_dir / "test_processed.csv", index=False)

print(f"Data preprocessing complete. Saved processed clean data to: {processed_dir}")
