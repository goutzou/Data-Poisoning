import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.autograd.functional import hessian

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
batch_size = 64

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
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = Net().to(device)

criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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

def get_test_loss_gradient(model, test_data, test_target):
    model.eval()
    test_data, test_target = test_data.to(device), test_target.to(device)
    output = model(test_data)
    loss = criterion(output, test_target)
    grad_test = torch.autograd.grad(loss, model.parameters())
    grad_test = torch.cat([g.contiguous().view(-1) for g in grad_test])
    return grad_test

# Compute gradient of the loss at training point
def get_train_loss_gradient(model, train_data, train_target):
    model.eval()
    train_data, train_target = train_data.to(device), train_target.to(device)
    output = model(train_data)
    loss = criterion(output, train_target)
    grad_train = torch.autograd.grad(loss, model.parameters())
    grad_train = torch.cat([g.contiguous().view(-1) for g in grad_train])
    return grad_train

# Function to compute Hessian-vector product
def hessian_vector_product(model, vector, data_loader):
    model.train()
    damping = 0.01  # Damping factor to ensure positive-definite Hessian
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_params = torch.cat([g.contiguous().view(-1) for g in grad_params])
        hvp = torch.autograd.grad(grad_params, model.parameters(), grad_outputs=vector, retain_graph=True)
        hvp = torch.cat([h.contiguous().view(-1) for h in hvp])
        return hvp + damping * vector

def conjugate_gradient(hvp_fn, b, data_loader, max_iter=100, tol=1e-5):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rsold = torch.dot(r, r)
    
    for i in range(max_iter):
        Ap = hvp_fn(p, data_loader)
        alpha = rsold / (torch.dot(p, Ap) + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        rsnew = torch.dot(r, r)
        if torch.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

def calculate_influence_scores(model, test_data, test_target, train_loader):
    # Get gradient at test point
    grad_test = get_test_loss_gradient(model, test_data, test_target)
    
    # Approximate H^{-1} * grad_test
    hessian_vec_prod_fn = lambda v, data_loader: hessian_vector_product(model, v, data_loader)
    s_test = conjugate_gradient(hessian_vec_prod_fn, grad_test, train_loader)
    
    influence_scores = []
    for data, target in train_loader:
        # For each training point
        grad_train = get_train_loss_gradient(model, data, target)
        influence = torch.dot(s_test, grad_train)
        influence_scores.append((influence.item(), data, target))
    
    # Sort training points by influence score
    influence_scores.sort(key=lambda x: x[0], reverse=True)
    return influence_scores

          
k = 10  # Number of points to perturb
top_influential_points = influence_scores[:k]


num_epochs = 5  
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)



