import torch
import torch.nn as nn
import torch.optim as optim

def hvp(loss, model, params, v):
    """
    Compute Hessian-vector product H v = (d^2 L / d theta^2) v using Pearlmutter's trick.
    """
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    grad_dot_v = sum((g * vv).sum() for g, vv in zip(grads, v))
    Hv = torch.autograd.grad(grad_dot_v, params, retain_graph=True)
    return Hv

def conjugate_gradient(loss_fn, model, params, b, cg_iters=10, cg_tol=1e-10):
    """
    Solve Hx = b using the Conjugate Gradient method.
    """
    x = [torch.zeros_like(p) for p in params]
    r = [bb.clone() for bb in b]
    p = [rr.clone() for rr in r]
    rr_dot = sum((rr * rr).sum() for rr in r)

    for i in range(cg_iters):
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

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
    def forward(self, x):
        return self.linear(x)

def compute_gradient(loss, params):
    grads = torch.autograd.grad(loss, params, create_graph=False, retain_graph=True)
    return grads

def compute_loss(model, X, y, reduction='sum'):
    logits = model(X)
    return nn.BCEWithLogitsLoss(reduction=reduction)(logits, y)

if __name__ == '__main__':
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Synthetic data
    torch.manual_seed(0)
    n_train = 100
    n_features = 10

    X_train = torch.randn(n_train, n_features).to(device)
    true_w = torch.randn(n_features).to(device)
    p = torch.sigmoid(X_train @ true_w)
    y_train = (p > 0.5).float().unsqueeze(1).to(device)

    # Train a logistic regression model
    model = LogisticRegression(n_features).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Training loop
    for _ in range(100):
        optimizer.zero_grad()
        loss = compute_loss(model, X_train, y_train)
        loss.backward()
        optimizer.step()

    # Pick a test point
    x_test = torch.randn(1, n_features).to(device)
    y_test = (torch.sigmoid(x_test @ true_w) > 0.5).float().unsqueeze(1).to(device)

    # Compute gradients for the test point
    test_loss = compute_loss(model, x_test, y_test)
    params = list(model.parameters())
    grad_test = compute_gradient(test_loss, params)

    # Approximate influence for a sampled subset of training points
    sample_size = min(50, n_train)  # Sample up to 50 points for efficiency
    sampled_indices = torch.randperm(n_train)[:sample_size]
    influences = []

    def full_train_loss():
        return compute_loss(model, X_train, y_train)

    for i in sampled_indices:
        xi = X_train[i:i+1]
        yi = y_train[i:i+1]
        loss_i = compute_loss(model, xi, yi)
        grad_i = compute_gradient(loss_i, params)

        # Solve H v = grad_i
        v = conjugate_gradient(full_train_loss, model, params, grad_i, cg_iters=10)

        # Influence = - grad_test^T v
        influence_i = -sum((gt * vv).sum().item() for gt, vv in zip(grad_test, v))
        influences.append(influence_i)

    print(f"Approximate influences for {len(sampled_indices)} sampled points:")
    print(influences)
