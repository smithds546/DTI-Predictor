import pandas as pd
from sklearn.model_selection import train_test_split
import os

# === File paths ===
input_path = "/Users/drs/Projects/DTI/Backend/app/data/processed/bindingdb_data.csv"
output_dir = "/Users/drs/Projects/DTI/Backend/app/data/prepped/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# === Load dataset ===
print(f"Loading dataset from {input_path}...")
df = pd.read_csv(input_path)
print(f"Dataset loaded: {len(df)} rows")

# === Split into Train (65%), Validation (20%), Test (15%) ===
train_df, temp_df = train_test_split(df, test_size=0.35, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=(15 / 35), random_state=42, shuffle=True)

# === Print summary ===
print(f"Train set: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
print(f"Validation set: {len(val_df)} rows ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test set: {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)")

# === Save files ===
train_path = os.path.join(output_dir, "bindingdb_train.csv")
val_path = os.path.join(output_dir, "bindingdb_validation.csv")
test_path = os.path.join(output_dir, "bindingdb_test.csv")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print("\n✅ Data successfully split and saved:")
print(f" - Training: {train_path}")
print(f" - Validation: {val_path}")
print(f" - Test: {test_path}")