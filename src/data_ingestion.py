from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

# Determine project base directory using pathlib
base_dir = Path(__file__).resolve().parent.parent
print(base_dir)
sample_data_path = base_dir / "data" / "sample_data.csv"
params_path = base_dir / "params.yaml"

# Load parameters from params.yaml
with open(params_path, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)["data_ingestion"]

test_size = params["test_size"]
random_state = params["random_state"]

# Load raw dataset
df = pd.read_csv(sample_data_path)

# Drop uninformative student_id column
df_clean = df.drop(columns=["student_id"])

# Stratified train/test split using parameters from params.yaml
train_data, test_data = train_test_split(
    df_clean, 
    test_size=test_size, 
    random_state=random_state, 
    stratify=df_clean["grade"]
)

# Output raw directory path
raw_dir = base_dir / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

# Save raw train and test split datasets
train_data.to_csv(raw_dir / "train.csv", index=False)
test_data.to_csv(raw_dir / "test.csv", index=False)

print(f"Data ingestion complete (test_size={test_size}, random_state={random_state}). Saved raw split data to: {raw_dir}")
