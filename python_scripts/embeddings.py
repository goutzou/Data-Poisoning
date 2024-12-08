import os
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

path = ""  # Update this with the actual path
data_file = os.path.join(path, "imdb_data_ints.csv")

df = pd.read_csv(data_file)

def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    return text.strip()

df['review_clean'] = df['review'].apply(clean_text)

# If df['sentiment'] is already 0 or 1, just convert directly:
df = df.dropna(subset=['sentiment'])  # Ensure no missing values in sentiment
df['label'] = df['sentiment'].astype(int)
df = df[df['label'].isin([0, 1])]  # Ensure only 0 or 1 present
df = df.dropna(subset=['label']).reset_index(drop=True)

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

batch_size = 32
reviews = df['review_clean'].tolist()
labels = df['label'].tolist()

# Convert labels to a tensor
labels = torch.tensor(labels, dtype=torch.long)

all_embeddings = []

for i in range(0, len(reviews), batch_size):
    batch_texts = reviews[i:i+batch_size]
    
    encoded = tokenizer(
        batch_texts, 
        padding=True, 
        truncation=True, 
        max_length=256,
        return_tensors='pt'
    )
    
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    with torch.no_grad():
        outputs = model(**encoded)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)

import numpy as np
all_embeddings = np.vstack(all_embeddings)
all_embeddings = torch.tensor(all_embeddings, dtype=torch.float)

print("Embeddings shape:", all_embeddings.shape)
print("Labels shape:", labels.shape)
