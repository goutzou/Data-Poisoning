import os
import torch
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ===================================
# Step 1: Load Data
# ===================================
# Define the path to the embeddings and labels file
embeddings_file = "Data-Poisoning/datasets/roberta_embeddings_labels.pt"

# Check if the embeddings file exists
if not os.path.exists(embeddings_file):
    raise FileNotFoundError(f"File not found: {embeddings_file}")

# Load embeddings and labels from the .pt file
all_embeddings, labels = torch.load(embeddings_file)

# Convert PyTorch tensors to NumPy arrays
X = all_embeddings.cpu().numpy()  # Embedding vectors
y = labels.cpu().numpy()          # Binary labels (0 or 1)

# ===================================
# Step 2: Train-Test Split
# ===================================
# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(y_train)}")
print(f"Test set size: {len(y_test)}")

# ===================================
# Step 3: Train Logistic Regression Model
# ===================================
# Initialize the logistic regression model
model = LogisticRegression(
    max_iter=1000,         # Allow sufficient iterations for convergence
    random_state=42,       # Ensure reproducibility
    solver='lbfgs',        # Good default solver for dense data
    n_jobs=-1,             # Use all CPU cores for efficiency
    class_weight='balanced'  # Handle potential class imbalance
)

# Train the model
print("Training the logistic regression model...")
model.fit(X_train, y_train)

# Save the trained model to disk
model_file = "Data-Poisoning/datasets/logistic_model.joblib"
joblib.dump(model, model_file)
print(f"Trained logistic regression model saved to: {model_file}")

# ===================================
# Step 4: Evaluate the Model
# ===================================
# Predict labels on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of the positive class (1)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# ===================================
# Step 5: Save Train-Test Split (Optional)
# ===================================
# Save the train-test split for reproducibility
split_file = "Data-Poisoning/datasets/train_test_split.npz"
np.savez(split_file, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
print(f"Train-test split saved to: {split_file}")
