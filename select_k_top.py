import pandas as pd
import glob
import math

# ================================
# Parameters
# ================================
input_pattern = "Data-Poisoning/datasets/influence_scores_chunk_*.csv"  # Adjust if needed
output_file = "Data-Poisoning/datasets/influence_scores_merged_top_1percent.csv"
percentage = 0.01  # 1% of the dataset

# ================================
# Step 1: Merge all CSV files
# ================================
csv_files = glob.glob(input_pattern)
if not csv_files:
    raise FileNotFoundError(f"No files found with pattern: {input_pattern}")

df_all = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

# Check required columns
required_cols = ['test_idx', 'unique_id', 'influence']
if not all(col in df_all.columns for col in required_cols):
    raise ValueError(f"The DataFrame must contain columns: {required_cols}")

# ================================
# Step 2: Compute k as 1% of unique training points
# ================================
unique_train_count = df_all['unique_id'].nunique()
k = int(unique_train_count * percentage)
if k < 1:
    k = 1  # Ensure at least 1 if 1% is less than 1 training point

print(f"Total unique training points: {unique_train_count}")
print(f"Selecting top {k} (which is {percentage*100}% of {unique_train_count}) per test_idx.")

# ================================
# Step 3: Group by test_idx and select top-k
# ================================
# Sort by influence descending
df_all = df_all.sort_values(by='influence', ascending=False)

# Select top-k per test_idx
df_topk = df_all.groupby('test_idx').head(k)

# ================================
# Step 4: Save the result
# ================================
df_topk.to_csv(output_file, index=False)
print(f"Top {k} influential training datapoints per test input saved to {output_file}")
