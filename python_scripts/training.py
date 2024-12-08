import torch
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ===================================
# Step 3: Model Setup and Baseline Training (Pure Data)
# ===================================
# Load embeddings and labels from step 2
all_embeddings, labels = torch.load("roberta_embeddings_labels.pt")

# Convert PyTorch tensors to NumPy arrays
X = all_embeddings.cpu().numpy()  # shape: [num_examples, hidden_dim]
y = labels.cpu().numpy()          # shape: [num_examples]

# Split into train/test sets. Adjust test_size as desired.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression(
    max_iter=1000,         # Increase max_iter to ensure convergence
    random_state=42,
    solver='lbfgs',        # Recommended solver for large datasets
    n_jobs=-1               # Utilize all available CPU cores
)

# Train the model on the pure (unpoisoned) data
model.fit(X_train, y_train)

# Save the trained model for later use
joblib.dump(model, "logistic_model.joblib")
print("Trained Logistic Regression model saved as 'logistic_model.joblib'.")

# Evaluate on the test set
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)  # default is binary for binary labels

print("Baseline Performance (Pure Data):")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")

# Optionally, save the train/test split for consistency
np.savez("train_test_split.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
print("Train/Test split saved as 'train_test_split.npz'.")
