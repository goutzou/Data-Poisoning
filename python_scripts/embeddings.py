import os
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# ===================================
# Step 1: Data Acquisition and Preparation
# ===================================
# Assuming you've already downloaded the dataset using kagglehub as in your snippet:
# path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
# For this script, just set `path` to the directory containing "IMDB Dataset.csv"
path = ""  # Update this with the actual path
data_file = os.path.join(path, "IMDB Dataset.csv")

# Load the dataset
df = pd.read_csv(data_file)

# Minimal text cleaning (optional)
# Here we remove HTML tags like <br />
def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    return text.strip()

df['review_clean'] = df['review'].apply(clean_text)

# Convert sentiment labels to binary: positive=1, negative=0
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# ===================================
# Step 2: Extracting RoBERTa Embeddings
# ===================================
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Put the model in evaluation mode
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

# For large datasets, consider batching and possibly using a DataLoader.
# This simple example processes data in batches to avoid memory issues.
batch_size = 32
reviews = df['review_clean'].tolist()
labels = df['label'].tolist()

all_embeddings = []
device = "cuda" if torch.cuda.is_available() else "cpu"

for i in range(0, len(reviews), batch_size):
    batch_texts = reviews[i:i+batch_size]
    
    # Tokenize the batch of texts
    encoded = tokenizer(
        batch_texts, 
        padding=True, 
        truncation=True, 
        max_length=256,  # Adjust max_length as needed
        return_tensors='pt'
    )
    
    # Move to device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    with torch.no_grad():
        outputs = model(**encoded)
        # outputs.last_hidden_state: [batch_size, seq_len, hidden_dim]
        # For embeddings, you can use the CLS token (index 0) or mean-pool the sequence.
        
        # CLS pooling (CLS token embedding)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # shape: [batch_size, hidden_dim]
        
        # Alternatively, mean pooling:
        # attention_mask = encoded['attention_mask']
        # sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask.unsqueeze(-1), dim=1)
        # sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9)
        # mean_embeddings = sum_embeddings / sum_mask
        
        # Here we choose CLS for simplicity:
        batch_embeddings = cls_embeddings.cpu().numpy()
        all_embeddings.append(batch_embeddings)

all_embeddings = torch.tensor([emb for batch in all_embeddings for emb in batch], dtype=torch.float)

# Now `all_embeddings` is a tensor with shape [num_examples, hidden_dim]
# and `labels` is a list of corresponding labels.

# Optionally, convert labels to a tensor
labels = torch.tensor(labels, dtype=torch.long)

# You can save these embeddings and labels for further steps:
# torch.save((all_embeddings, labels), "roberta_embeddings_labels.pt")

print("Embeddings shape:", all_embeddings.shape)
print("Labels shape:", labels.shape)
