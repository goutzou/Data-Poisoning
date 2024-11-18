import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import grad
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

batch_size = 64
num_epochs = 5
learning_rate = 0.01
momentum = 0.9

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=1000,
    shuffle=False
)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  
        return x

model = Net().to(device)
criterion = nn.CrossEntropyLoss()  
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

def train(model, device, train_loader, optimizer, epoch):
    model.train()  
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()  
        output = model(data)
        loss = criterion(output, target)  
        loss.backward()  
        optimizer.step()  
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} '
                  f'[{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()  
    test_loss = 0
    correct = 0
    with torch.no_grad():  
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  
            test_loss += criterion(output, target).item()  
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader) 
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.2f}%)\n')

print("Training the original model...\n")
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

def get_loss_gradient(model, data, target):
    model.eval()
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = criterion(output, target)
    grad_params = torch.autograd.grad(loss, model.parameters())
    grad_vector = torch.cat([g.view(-1) for g in grad_params])
    return grad_vector

def hessian_vector_product(model, vector, data_loader):
    damping = 0.01  
    hvp = torch.zeros_like(vector)
    model.train()
    num_batches = 10  
    for i, (data, target) in enumerate(data_loader):
        if i >= num_batches:
            break
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_vector = torch.cat([g.contiguous().view(-1) for g in grad_params])
        grad_grad = torch.autograd.grad(grad_vector, model.parameters(), grad_outputs=vector, retain_graph=True)
        hvp_sample = torch.cat([g.contiguous().view(-1) for g in grad_grad])
        hvp += hvp_sample
    hvp = hvp / num_batches + damping * vector
    return hvp


def conjugate_gradient(hvp_fn, b, data_loader, max_iter=50, tol=1e-5):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rsold = torch.dot(r, r)
    
    for i in range(max_iter):
        Avp = hvp_fn(model, p, data_loader)
        alpha = rsold / (torch.dot(p, Avp) + 1e-8)
        x += alpha * p
        r -= alpha * Avp
        rsnew = torch.dot(r, r)
        if torch.sqrt(rsnew) < tol:
            print(f'Converged after {i+1} iterations.')
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

test_sample_idx = 0  
test_data_sample = test_dataset[test_sample_idx][0].unsqueeze(0) 
test_target_sample = torch.tensor([test_dataset[test_sample_idx][1]])

print("Computing gradient at the test point...")
grad_test = get_loss_gradient(model, test_data_sample, test_target_sample)

print("Approximating H^{-1} * grad_test...")
s_test = conjugate_gradient(hessian_vector_product, grad_test, train_loader)

print("Computing influence scores for training data...")
influence_scores = []
num_train_samples = 10000 
train_subset = torch.utils.data.Subset(train_dataset, range(num_train_samples))
train_subset_loader = torch.utils.data.DataLoader(
    dataset=train_subset,
    batch_size=1,
    shuffle=False
)

for idx, (data, target) in enumerate(train_subset_loader):
    grad_train = get_loss_gradient(model, data, target)
    influence = torch.dot(s_test, grad_train).item()
    influence_scores.append((influence, idx, data, target))

influence_scores.sort(key=lambda x: x[0], reverse=True)

k = 100  
top_influential_points = influence_scores[:k]
print(f"Top {k} influential training points identified.")
epsilon = 0.5  
print("Applying adversarial perturbations to the most influential points...")

train_data_perturbed = train_dataset.data.clone().float() / 255.0
train_targets_perturbed = train_dataset.targets.clone()

for influence, idx, data, target in top_influential_points:
    data = data.to(device)
    data.requires_grad = True
    output = model(data)
    loss = criterion(output, target.to(device))
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    train_data_perturbed[idx] = perturbed_data.detach().cpu().squeeze() 
    original_label = train_targets_perturbed[idx].item()
    new_label = (original_label + 1) % 10 
    train_targets_perturbed[idx] = new_label
    print(f"Perturbed training sample {idx}: Original label {original_label}, New label {new_label}")

train_dataset_poisoned = torch.utils.data.TensorDataset(train_data_perturbed.unsqueeze(1), train_targets_perturbed)
train_loader_poisoned = torch.utils.data.DataLoader(
    dataset=train_dataset_poisoned,
    batch_size=batch_size,
    shuffle=True
)

model_poisoned = Net().to(device)
learning_rate = 0.005  
optimizer_poisoned = optim.SGD(model_poisoned.parameters(), lr=learning_rate, momentum=momentum)

print("\nRetraining the model with poisoned data...\n")
for epoch in range(1, num_epochs + 1):
    train(model_poisoned, device, train_loader_poisoned, optimizer_poisoned, epoch)
    test(model_poisoned, device, test_loader)

print("Evaluating the impact on the specific test point...\n")
model_poisoned.eval()
all_preds = []
all_targets = []
model.eval()
with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=False).cpu()
        all_preds.extend(pred.numpy())
        all_targets.extend(target.numpy())

conf_mat = confusion_matrix(all_targets, all_preds)

most_confused = conf_mat.argmax(axis=1)
for influence, idx, data, target in top_influential_points:
    original_label = train_targets_perturbed[idx].item()
    new_label = most_confused[original_label]
    train_targets_perturbed[idx] = new_label
    print(f"Perturbed training sample {idx}: Original label {original_label}, New label {new_label}")
