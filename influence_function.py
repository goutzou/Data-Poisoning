import torch
import torch.nn as nn
import torch.optim as optim

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

if __name__ == '__main__':
    # Synthetic data
    torch.manual_seed(0)
    n_train = 100
    n_features = 10

    X_train = torch.randn(n_train, n_features)
    true_w = torch.randn(n_features)
    p = torch.sigmoid(X_train @ true_w)
    y_train = (p > 0.5).float().unsqueeze(1)

    # Train a logistic regression model
    model = LogisticRegression(n_features)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Simple training loop
    for _ in range(100):
        optimizer.zero_grad()
        loss = compute_loss(model, X_train, y_train)
        loss.backward()
        optimizer.step()

    # Pick a test point
    x_test = torch.randn(1, n_features)
    y_test = (torch.sigmoid(x_test @ true_w) > 0.5).float().unsqueeze(1)

    # Compute gradients for test point
    test_loss = compute_loss(model, x_test, y_test)
    params = list(model.parameters())
    grad_test = compute_gradient(test_loss, params)

    # We'll compute influence of each training point
    influences = []

    # Define a closure for loss on training set (sum of individual losses)
    # We'll use this to get Hessian-vector products
    def full_train_loss():
        return compute_loss(model, X_train, y_train)

    # For each training point, approximate H^{-1} grad_i
    # Note: This is O(n_train * cg_iters) which can be slow. Consider just sampling points.
    for i in range(n_train):
        xi = X_train[i:i+1]
        yi = y_train[i:i+1]
        loss_i = compute_loss(model, xi, yi)
        grad_i = compute_gradient(loss_i, params)

        # Solve H v = grad_i using conjugate gradient
        # Here H is Hessian of full_train_loss
        def loss_fn():
            return full_train_loss()

        v = conjugate_gradient(loss_fn, model, params, grad_i, cg_iters=10)

        # Now compute influence = - grad_test^T v
        influence_i = 0.0
        for gt, vv in zip(grad_test, v):
            influence_i += (gt * vv).sum()
        influence_i = -influence_i.item()
        influences.append(influence_i)

    print("Approximate influences:", influences)
