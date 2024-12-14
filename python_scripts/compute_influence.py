import os
import torch
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ===================================
# Step 1: Load Data and Model
# ===================================
# File paths
embeddings_file = "../datasets/roberta_embeddings_labels.pt"
ids_file = "../datasets/roberta_embeddings.csv"  # CSV containing unique_id
model_file = "./logistic_model_poison.joblib"

# Load embeddings and labels
if not os.path.exists(embeddings_file):
    raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model file not found: {model_file}")
if not os.path.exists(ids_file):
    raise FileNotFoundError(f"IDs file not found: {ids_file}")

all_embeddings, labels = torch.load(embeddings_file)
X_train = all_embeddings.cpu().numpy()
y_train = labels.cpu().numpy()

# Load unique_ids from the embeddings CSV
ids_df = pd.read_csv(ids_file)
unique_ids = ids_df['unique_id'].tolist()

# Ensure consistency
if len(unique_ids) != len(X_train):
    raise ValueError("Mismatch between number of unique IDs and embeddings.")

# Load trained logistic regression model
model = joblib.load(model_file)

# ===================================
# Step 2: Prepare a Smaller Subset
# ===================================
# Use fewer training and test points for quick verification
train_subset_size = 20  # Take only first 20 training samples
test_subset_size = 5    # Take only first 5 test samples

X_train = X_train[:train_subset_size]
y_train = y_train[:train_subset_size]
unique_ids = unique_ids[:train_subset_size]

X_test = X_train[:test_subset_size]
y_test = y_train[:test_subset_size]

# Ensure PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# ===================================
# Step 3: Define PyTorch Logistic Regression
# ===================================
class LogisticRegressionTorch(torch.nn.Module):
    def __init__(self, input_dim, sklearn_model):
        super(LogisticRegressionTorch, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        self.linear.weight.data = torch.tensor(sklearn_model.coef_, dtype=torch.float32)
        self.linear.bias.data = torch.tensor(sklearn_model.intercept_, dtype=torch.float32)

    def forward(self, x):
        return self.linear(x)

input_dim = X_train.shape[1]
torch_model = LogisticRegressionTorch(input_dim, model)
torch_model.eval()

# ===================================
# Step 4: Influence Computation Functions
# ===================================
def compute_loss(model, X, y, reduction='sum'):
    logits = model(X)
    return torch.nn.BCEWithLogitsLoss(reduction=reduction)(logits, y)

def compute_gradient(loss, params):
    grads = torch.autograd.grad(loss, params, create_graph=False, retain_graph=True)
    return grads

def hvp(loss, model, params, v):
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    grad_dot_v = sum((g * vv).sum() for g, vv in zip(grads, v))
    Hv = torch.autograd.grad(grad_dot_v, params, retain_graph=True)
    return Hv

def conjugate_gradient(loss_fn, model, params, b, cg_iters=10, cg_tol=1e-10):
    x = [torch.zeros_like(p) for p in params]
    r = [bb.clone() for bb in b]
    p = [rr.clone() for rr in r]
    rr_dot = sum((rr * rr).sum() for rr in r)

    for _ in range(cg_iters):
        Hp = hvp(loss_fn(), model, params, p)
        pHp = sum((pp * hpp).sum() for pp, hpp in zip(p, Hp))
        alpha = rr_dot / pHp

        x = [xx + alpha * pp for xx, pp in zip(x, p)]
        r = [rr - alpha * hpp for rr, hpp in zip(r, Hp)]

        new_rr_dot = sum((rrr * rrr).sum() for rrr in r)
        if new_rr_dot < cg_tol:
            break
        beta = new_rr_dot / rr_dot
        p = [rr + beta * pp for rr, pp in zip(r, p)]
        rr_dot = new_rr_dot

    return x

# ===================================
# Step 5: Compute Influence for Each Test Point
# ===================================
influences = []

params = list(torch_model.parameters())

# Precompute the full training loss closure
def full_train_loss():
    return compute_loss(torch_model, X_train_tensor, y_train_tensor)

for test_idx in range(len(X_test)):
    # Get test point
    x_test_single = X_test_tensor[test_idx:test_idx+1]
    y_test_single = y_test_tensor[test_idx:test_idx+1]

    # Compute gradients for test point
    test_loss = compute_loss(torch_model, x_test_single, y_test_single)
    grad_test = compute_gradient(test_loss, params)

    s_test = conjugate_gradient(full_train_loss, torch_model, params, grad_test, cg_iters=10)
    
    # Compute influence for each (subset of) training points
    for i in range(len(X_train)):
        xi = X_train_tensor[i:i+1]
        yi = y_train_tensor[i:i+1]
        loss_i = compute_loss(torch_model, xi, yi)
        grad_i = compute_gradient(loss_i, params)
        # Influence = - s_test^T grad_i
        influence_i = -sum((ss * vv).sum().item() for ss, vv in zip(s_test, grad_i))
        influences.append({'test_idx': test_idx, 'unique_id': unique_ids[i], 'influence': influence_i})
        print(i)
 
# Save influences to CSV
output_file = "../datasets/temp/influence_scores.csv"
influence_df = pd.DataFrame(influences)
influence_df.to_csv(output_file, index=False)

print(f"Influences saved to {output_file}")
