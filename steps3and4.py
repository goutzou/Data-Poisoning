import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.linear_model import LogisticRegression

# ============================
# Define the PyTorch Logistic Regression Model
# ============================
class PyTorchLogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(PyTorchLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        return self.linear(x)

# ============================
# Define Utility Functions for Influence Computation
# ============================
def compute_gradient(loss, params):
    """
    Compute gradients of the loss with respect to the parameters.
    """
    grads = torch.autograd.grad(loss, params, create_graph=False, retain_graph=True)
    return grads

def compute_loss(model, X, y):
    """
    Compute the binary cross-entropy loss with logits.
    """
    logits = model(X)
    loss = nn.BCEWithLogitsLoss()(logits, y)
    return loss

def hvp(loss, model, params, v):
    """
    Compute Hessian-vector product H v using Pearlmutter's trick.
    """
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    grad_dot_v = 0
    for g, vv in zip(grads, v):
        grad_dot_v += (g * vv).sum()
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
        Hp = hvp(loss_fn(), model, params, p)

        pHp = sum((pp * hpp).sum() for pp, hpp in zip(p, Hp))
        if pHp == 0:
            break
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

# ============================
# Main Function to Execute Steps 3 and 4
# ============================
def main():
    #######################
    # Step 3: Randomly Sample Test Datapoints
    #######################
    # Load train/test split
    split_data = np.load("train_test_split.npz")
    X_train = split_data["X_train"]  # Shape: [num_train, hidden_dim]
    y_train = split_data["y_train"]  # Shape: [num_train]
    X_test = split_data["X_test"]    # Shape: [num_test, hidden_dim]
    y_test = split_data["y_test"]    # Shape: [num_test]

    # Number of test points to sample
    num_test_points = 10
    np.random.seed(42)  # For reproducibility
    test_indices = np.random.choice(len(X_test), size=num_test_points, replace=False)

    # Save sampled test indices
    np.save("sampled_test_indices.npy", test_indices)
    print(f"Saved {num_test_points} sampled test indices to 'sampled_test_indices.npy'")

    #######################
    # Step 4: Compute Influence of Each Training Datapoint on Each Sampled Test Input
    #######################

    # Load the trained logistic regression model
    sk_model = joblib.load("logistic_model.joblib")

    # Extract parameters from the sklearn model
    # For binary classification, sk_model.coef_ has shape (1, n_features) and sk_model.intercept_ has shape (1,)
    w = sk_model.coef_.flatten()
    b = sk_model.intercept_.item()

    # Initialize the PyTorch logistic regression model
    input_dim = X_train.shape[1]
    model = PyTorchLogisticRegression(input_dim)

    # Copy weights and biases from scikit-learn model to PyTorch model
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor(w, dtype=torch.float).unsqueeze(0))
        model.linear.bias.copy_(torch.tensor(b, dtype=torch.float))

    # Ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad = True

    # Set model to training mode to enable gradient tracking
    model.train()

    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Convert data to PyTorch tensors and move to device
    X_train_t = torch.tensor(X_train, dtype=torch.float).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float).unsqueeze(1).to(device)  # Shape: [num_train, 1]
    X_test_t = torch.tensor(X_test, dtype=torch.float).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float).unsqueeze(1).to(device)    # Shape: [num_test, 1]

    # Define the full training loss function
    def full_train_loss():
        logits = model(X_train_t)
        loss = nn.BCEWithLogitsLoss()(logits, y_train_t)
        return loss

    # Extract model parameters
    params = list(model.parameters())

    # Load sampled test indices
    sampled_test_indices = np.load("sampled_test_indices.npy")

    # Initialize dictionary to store influence scores
    test_influences = {}

    # Number of conjugate gradient iterations
    cg_iters = 10

    # Iterate over each sampled test index
    for idx in sampled_test_indices:
        x_test_i = X_test_t[idx:idx+1]  # Shape: [1, hidden_dim]
        y_test_i = y_test_t[idx:idx+1]  # Shape: [1, 1]

        # Compute test loss
        test_loss = compute_loss(model, x_test_i, y_test_i)

        # Compute gradient of test loss w.r.t. model parameters
        grad_test = compute_gradient(test_loss, params)

        # Initialize list to store influences for this test point
        influences = []

        # Define closure for full training loss
        def loss_fn():
            return full_train_loss()

        # Iterate over each training datapoint
        for i in range(len(X_train_t)):
            xi = X_train_t[i:i+1]      # Shape: [1, hidden_dim]
            yi = y_train_t[i:i+1]      # Shape: [1, 1]

            # Compute training loss for datapoint i
            loss_i = compute_loss(model, xi, yi)

            # Compute gradient of training loss w.r.t. model parameters
            grad_i = compute_gradient(loss_i, params)

            # Solve H v = grad_i using conjugate gradient
            v = conjugate_gradient(loss_fn, model, params, grad_i, cg_iters=cg_iters)

            # Compute influence = - grad_test^T v
            influence_i = 0.0
            for gt, vv in zip(grad_test, v):
                influence_i += (gt * vv).sum().item()
            influence_i = -influence_i
            influences.append(influence_i)

        # Store the influences for this test index
        test_influences[idx] = np.array(influences)

        print(f"Computed influences for test index {idx}")

    # Save the influence scores
    # Since direct saving of dictionaries with numpy arrays can be tricky, we'll save them as separate files
    # Save the list of test indices
    np.save("influence_test_indices.npy", np.array(list(test_influences.keys())))
    
    # Save the influence values as a list of arrays
    # Each entry corresponds to the influence scores of training datapoints on a specific test datapoint
    influence_values = [test_influences[k] for k in test_influences.keys()]
    np.save("influence_values.npy", np.array(influence_values), allow_pickle=True)

    print("Saved influence scores to 'influence_test_indices.npy' and 'influence_values.npy'.")

if __name__ == "__main__":
    main()
