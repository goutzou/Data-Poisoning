import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

path = "Data-Poisoning/datasets/"  # Folder to save the files
os.makedirs(path, exist_ok=True)  # Ensure the folder exists

data_file = os.path.join(path, "imdb_data_ints.csv")

# Load the dataset
df = pd.read_csv(data_file)

# Ensure 'unique_id' column exists
if 'unique_id' not in df.columns:
    raise ValueError("The 'unique_id' column is missing from the dataset.")

# Clean review text
def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text)
    return text.strip()

df['review_clean'] = df['review'].apply(clean_text)

# Ensure labels are binary (0 or 1) and drop missing values
df = df.dropna(subset=['sentiment'])
df['label'] = df['sentiment'].astype(int)
df = df[df['label'].isin([0, 1])].reset_index(drop=True)

# Load RoBERTa model and tokenizer
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

batch_size = 32
reviews = df['review_clean'].tolist()
unique_ids = df['unique_id'].tolist()  # Use the existing unique_id column

all_embeddings = []

# Generate embeddings
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

# Combine all embeddings into a single matrix
all_embeddings = np.vstack(all_embeddings)

# Create a DataFrame with unique IDs and embeddings
embedding_df = pd.DataFrame(
    data=all_embeddings, 
    columns=[f"dim_{i}" for i in range(all_embeddings.shape[1])]
)
embedding_df['unique_id'] = unique_ids

# Save the embedding DataFrame as a CSV file
csv_file = os.path.join(path, "roberta_embeddings.csv")
embedding_df.to_csv(csv_file, index=False)

# Save the embeddings as a PyTorch tensor file
pt_file = os.path.join(path, "roberta_embeddings.pt")
torch.save(torch.tensor(all_embeddings, dtype=torch.float), pt_file)

print(f"Embeddings saved to CSV: {csv_file}")
print(f"Embeddings saved to PyTorch file: {pt_file}")
