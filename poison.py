import torch
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from numpy.linalg import inv

# ===================================
# Step 4: Influence Function Computation and Data Poisoning
# ===================================

# Load the saved train/test split
split = np.load("train_test_split.npz")
X_train = split['X_train']
X_test = split['X_test']
y_train = split['y_train']
y_test = split['y_test']

# Load the trained logistic regression model
model = joblib.load("logistic_model.joblib")
print("Loaded trained Logistic Regression model from 'logistic_model.joblib'.")

# Function to add intercept term
def add_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

# Augment the data with intercept
X_train_aug = add_intercept(X_train)
X_test_aug = add_intercept(X_test)

# Extract model parameters (theta)
# model.coef_ shape: (1, d), model.intercept_ shape: (1,)
theta = np.concatenate([model.intercept_, model.coef_.ravel()])  # shape: (d+1,)
print(f"Model parameters (theta) shape: {theta.shape}")

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Choose a target test review
target_index = 0  # Change this index to select a different test sample
x_test_target = X_test_aug[target_index:target_index+1]  # shape (1, d+1)
y_test_target = y_test[target_index]

# Predict before poisoning
p_test_before = sigmoid(x_test_target.dot(theta))
y_pred_before = int(p_test_before >= 0.5)
print(f"\nTarget Test Review Before Poisoning:")
print(f"True Label: {y_test_target}")
print(f"Predicted Probability: {p_test_before[0]:.4f}")
print(f"Predicted Label: {y_pred_before}")

# Compute gradient wrt theta for target test point
grad_test = (p_test_before - y_test_target) * x_test_target  # shape: (1, d+1)

# Compute gradients for all training points
p_train = sigmoid(X_train_aug.dot(theta))  # shape: (n_train,)
grad_train = ((p_train - y_train)[:, np.newaxis] * X_train_aug)  # shape: (n_train, d+1)

# Compute Hessian: H = Σ p_i(1-p_i) x_i x_i^T
W = p_train * (1 - p_train)  # shape: (n_train,)
# To compute X_train_aug^T * diag(W) * X_train_aug efficiently
H = (X_train_aug.T * W).dot(X_train_aug)  # shape: (d+1, d+1)

# Invert Hessian (add small ridge for numerical stability)
epsilon = 1e-6
try:
    H_inv = inv(H + epsilon * np.eye(H.shape[0]))
except np.linalg.LinAlgError:
    print("Hessian is singular and cannot be inverted. Consider using a regularizer or reducing dimensionality.")
    exit()

print(f"Hessian inverse (H_inv) shape: {H_inv.shape}")

# Compute influence of each training point
# I(z_test, z_i) = - grad_test * H_inv * grad_i^T
# Vectorize the computation for efficiency
influences = -grad_train.dot(H_inv).dot(grad_test.T).flatten()  # shape: (n_train,)

# Identify most influential points
# Sort indices by influence scores in descending order
most_influential_idx = np.argsort(influences)[::-1]

print("\nTop 10 most influential training points for target test review:")
for rank in range(10):
    idx = most_influential_idx[rank]
    influence_score = influences[idx]
    print(f"Rank {rank+1}: Index {idx}, Influence Score: {influence_score:.6f}")

# ===================================
# Data Poisoning Attack
# ===================================

# Define poisoning strategy
# Example: Flip labels of the top N influential training points
N = 200  # Increase N to poison more points
poison_indices = most_influential_idx[:N]

# Display selected indices to poison
print(f"\nPoisoning the top {N} influential training points by flipping their labels.")

# Flip labels: if label=1, make it 0; if label=0, make it 1
y_train_poisoned = y_train.copy()
y_train_poisoned[poison_indices] = 1 - y_train_poisoned[poison_indices]

# Optional: Inspect some of the poisoned points
print("\nSample of poisoned labels (first 5):")
for i in poison_indices[:5]:
    print(f"Index {i}: Original Label = {y_train[i]}, Poisoned Label = {y_train_poisoned[i]}")

# ===================================
# Retrain Logistic Regression on Poisoned Data
# ===================================
poisoned_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs',
    n_jobs=-1
)

# Train on poisoned data
poisoned_model.fit(X_train, y_train_poisoned)
print("\nRetrained Logistic Regression model on poisoned data.")

# Save the poisoned model
joblib.dump(poisoned_model, "logistic_model_poisoned.joblib")
print("Poisoned Logistic Regression model saved as 'logistic_model_poisoned.joblib'.")

# Predict after poisoning
# Extract poisoned model's parameters
theta_poisoned = np.concatenate([poisoned_model.intercept_, poisoned_model.coef_.ravel()])

# Predict on the target test review with poisoned model
p_test_after = sigmoid(x_test_target.dot(theta_poisoned))
y_pred_after = int(p_test_after >= 0.5)
print(f"\nTarget Test Review After Poisoning:")
print(f"True Label: {y_test_target}")
print(f"Predicted Probability: {p_test_after[0]:.4f}")
print(f"Predicted Label: {y_pred_after}")

# Evaluate the poisoned model on the entire test set
y_pred_poisoned = poisoned_model.predict(X_test)
acc_poisoned = accuracy_score(y_test, y_pred_poisoned)
f1_poisoned = f1_score(y_test, y_pred_poisoned)

print("\nPoisoned Model Performance on Test Set:")
print(f"Accuracy: {acc_poisoned:.4f}")
print(f"F1 Score: {f1_poisoned:.4f}")

# ===================================
# Compare Baseline and Poisoned Models
# ===================================

# To compare, you can re-run the baseline prediction on the target test review
# or save baseline predictions before poisoning.

# Alternatively, include baseline prediction in this script
# Recompute baseline prediction
p_test_baseline = sigmoid(x_test_target.dot(theta))
y_pred_baseline = int(p_test_baseline >= 0.5)

print("\nBaseline Model Prediction for Target Test Review:")
print(f"True Label: {y_test_target}")
print(f"Predicted Probability: {p_test_baseline[0]:.4f}")
print(f"Predicted Label: {y_pred_baseline}")

# Now, compare baseline and poisoned predictions
print("\nComparison of Target Test Review Predictions:")
print(f"Baseline Predicted Probability: {p_test_baseline[0]:.4f}, Predicted Label: {y_pred_baseline}")
print(f"Poisoned Predicted Probability: {p_test_after[0]:.4f}, Predicted Label: {y_pred_after}")
