import os
import sys
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import hashlib

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.gcs_utils import upload_to_gcs  # Import your GCS upload utility

# Local paths and GCS configurations
path = "../datasets/"  # Folder to save the files locally
os.makedirs(path, exist_ok=True)  # Ensure the folder exists
bucket_name = "sentiment-analysis-datasets"
embeddings_csv_gcs_blob_name = "embeddings/csv/roberta_embeddings.csv"
embeddings_pt_gcs_blob_name = "embeddings/pytorch/roberta_embeddings.pt"

# Load the pre-processed dataset
data_file = os.path.join(path, "binary_imdb_data_with_ids.csv")
df = pd.read_csv(data_file)

# Ensure required columns exist
if 'unique_id' not in df.columns or 'review' not in df.columns:
    raise ValueError("The required columns ('unique_id', 'review') are missing from the dataset.")

# Load RoBERTa model and tokenizer
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Parameters
batch_size = 8
reviews = df['review'].tolist()
unique_ids = df['unique_id'].tolist()

all_embeddings = []
all_ids = []

try:
    # Process in batches
    for i in range(0, len(reviews), batch_size):
        batch_texts = reviews[i:i + batch_size]
        batch_uids = unique_ids[i:i + batch_size]

        # Tokenization
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Store batch embeddings and IDs in memory
        all_embeddings.append(cls_embeddings)
        all_ids.extend(batch_uids)

        print(f"Processed batch {i // batch_size} ({i + len(batch_texts)}/{len(reviews)} rows).")

        # Free memory
        del encoded, outputs, cls_embeddings
        torch.cuda.empty_cache()

except Exception as e:
    print(f"An error occurred during the embedding process: {e}")
    sys.exit(1)

# Concatenate all embeddings
all_embeddings = np.vstack(all_embeddings)

# Create DataFrame with embedding dimensions and unique_id
num_dims = all_embeddings.shape[1]
columns = [f"embedding_dim_{dim}" for dim in range(num_dims)]
embeddings_df = pd.DataFrame(all_embeddings, columns=columns)
embeddings_df['unique_id'] = all_ids

# Save to CSV once
combined_csv_file = os.path.join(path, "roberta_embeddings.csv")
embeddings_df.to_csv(combined_csv_file, index=False)
print(f"Embeddings CSV saved to: {combined_csv_file}")

# Save combined embeddings as a PyTorch tensor
pt_file = os.path.join(path, "roberta_embeddings.pt")
torch.save(torch.tensor(all_embeddings, dtype=torch.float), pt_file)
print(f"Embeddings saved to PyTorch file: {pt_file}")

# Upload CSV to GCS
try:
    upload_to_gcs(pd.read_csv(combined_csv_file), bucket_name, embeddings_csv_gcs_blob_name)
    print(f"Embeddings CSV uploaded to GCS bucket '{bucket_name}' as '{embeddings_csv_gcs_blob_name}'.")
except Exception as e:
    print(f"Failed to upload CSV to GCS: {e}")

# Upload PyTorch tensor path to GCS (wrapped in a DataFrame for upload_to_gcs)
try:
    upload_to_gcs(pd.DataFrame({'path': [pt_file]}), bucket_name, embeddings_pt_gcs_blob_name)
    print(f"Embeddings PyTorch file uploaded to GCS bucket '{bucket_name}' as '{embeddings_pt_gcs_blob_name}'.")
except Exception as e:
    print(f"Failed to upload PyTorch file path to GCS: {e}")