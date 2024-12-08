import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# We assume you have already defined:
# hvp, conjugate_gradient, compute_gradient, compute_loss, LogisticRegression
# from the previous code snippet you provided.


def hvp(loss, model, params, v):
    """
    Compute Hessian-vector product H v = (d^2 L / d theta^2) v using Pearlmutter's trick.
    """
    # First backprop to get gradients
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    # Dot product of grads with v
    grad_dot_v = 0
    for g, vv in zip(grads, v):
        grad_dot_v += (g * vv).sum()
    # Second backprop to get Hv
    Hv = torch.autograd.grad(grad_dot_v, params, retain_graph=True)
    return Hv

def conjugate_gradient(loss_fn, model, params, b, cg_iters=10, cg_tol=1e-10):
    """
    Use conjugate gradient to solve Hx = b where H is the Hessian of loss_fn at params.
    """
    x = [torch.zeros_like(p) for p in params]
    r = [bb.clone() for bb in b]  # r = b - A x, initially x=0 so r=b
    p = [rr.clone() for rr in r]
    rr_dot = sum((rr * rr).sum() for rr in r)

    for i in range(cg_iters):
        # Compute H p
        # We'll use a small trick: define a closure that returns the same loss for computing Hvp
        def closure():
            return loss_fn()

        Hp = hvp(loss_fn(), model, params, p)

        pHp = sum((pp * hpp).sum() for pp, hpp in zip(p, Hp))
        alpha = rr_dot / pHp

        # x <- x + alpha p
        x = [xx + alpha * pp for xx, pp in zip(x, p)]
        # r <- r - alpha H p
        r = [rr - alpha * hpp for rr, hpp in zip(r, Hp)]

        new_rr_dot = sum((rrr * rrr).sum() for rrr in r)
        if new_rr_dot < cg_tol:
            break
        beta = new_rr_dot / rr_dot
        # p <- r + beta p
        p = [rr + beta * pp for rr, pp in zip(r, p)]
        rr_dot = new_rr_dot

    return x


# Simple logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
    def forward(self, x):
        return self.linear(x)

def compute_gradient(loss, params):
    grads = torch.autograd.grad(loss, params, create_graph=False, retain_graph=True)
    return grads

def compute_loss(model, X, y):
    logits = model(X)
    loss = nn.BCEWithLogitsLoss()(logits, y)
    return loss

# Load your train/test split (these should already be RoBERTa embeddings)
data = np.load('train_test_split.npz')
X_train_np = data['X_train']   # shape: (n_train, embedding_dim)
X_test_np = data['X_test']     # shape: (n_test, embedding_dim)
y_train_np = data['y_train']   # shape: (n_train, 1)
y_test_np = data['y_test']     # shape: (n_test, 1)

# Convert numpy arrays to torch tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32)

# We assume that steps 1 and 2 have already been done:
# - You have transformed data to embeddings (already loaded here)
# - You have trained a logistic regression model (we'll just retrain it here for completeness)
n_features = X_train.shape[1]
model = LogisticRegression(n_features)

optimizer = optim.SGD(model.parameters(), lr=0.1)

# Retrain your logistic regression model (if needed)
# If you already have a trained model, you can skip this part and just load it.
for _ in range(100):
    optimizer.zero_grad()
    loss = compute_loss(model, X_train, y_train)
    loss.backward()
    optimizer.step()

params = list(model.parameters())

# Define a function to get full training loss (sum or mean over entire training set)
def full_train_loss():
    return compute_loss(model, X_train, y_train)

# Steps 3 and 4:
# Step 3: Randomly sample test datapoints from test set
num_test_samples = 5  # for example, choose 5 random test points
random_indices = torch.randint(low=0, high=X_test.shape[0], size=(num_test_samples,))
sampled_X_test = X_test[random_indices]
sampled_y_test = y_test[random_indices]

# Step 4: Compute the influence of each training datapoint on each of our sampled test inputs
# We'll store the results in a dictionary:
# influences_dict[test_index] = list of influences of all training points on this test input
influences_dict = {}

for idx_in_batch, test_idx in enumerate(random_indices):
    x_test_i = sampled_X_test[idx_in_batch].unsqueeze(0)
    y_test_i = sampled_y_test[idx_in_batch].unsqueeze(0)

    # Compute gradients for this test point
    test_loss = compute_loss(model, x_test_i, y_test_i)
    grad_test = compute_gradient(test_loss, params)

    # Compute influence for each training point
    influences = []
    for i in range(X_train.shape[0]):
        xi = X_train[i:i+1]
        yi = y_train[i:i+1]
        loss_i = compute_loss(model, xi, yi)
        grad_i = compute_gradient(loss_i, params)

        # Solve H v = grad_i using conjugate gradient
        def loss_fn():
            return full_train_loss()

        v = conjugate_gradient(loss_fn, model, params, grad_i, cg_iters=10)

        # influence = - grad_test^T v
        influence_i = 0.0
        for gt, vv in zip(grad_test, v):
            influence_i += (gt * vv).sum()
        influence_i = -influence_i.item()
        influences.append(influence_i)
    
    influences_dict[test_idx.item()] = influences

print("Influences computed for sampled test points:")
for test_idx, infl in influences_dict.items():
    print(f"Test Index {test_idx}: {infl[:10]} ...")  # print first 10 influences for brevity

