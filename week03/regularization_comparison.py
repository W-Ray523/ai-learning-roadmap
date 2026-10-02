import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

generator = torch.Generator().manual_seed(42)

x = torch.randn(1000, 20)

y = (
    x[:, 0]
    + 0.8 * x[:, 1]
    - 0.6 * x[:, 2]
    > 0
).long()

dataset = TensorDataset(x, y)

train_dataset, val_dataset = random_split(
    dataset,
    [200, 800],
    generator=generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

class RegularizedClassifier(nn.Module):
    def __init__(self, dropout_p = 0.5):
        super().__init__()

        self.fc1 = nn.Linear(20, 128)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dp(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dp(x)
        x = self.fc3(x)
        return x

model = RegularizedClassifier(dropout_p = 0.5)  

def train_one_epoch(model, device, optimizer, criterion, loader):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)

        predicted = outputs.argmax(dim = 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc

def evaluate(model, device, criterion, loader):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
        
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
        
                total_loss += loss.item() * x_batch.size(0)
        
                predicted = outputs.argmax(dim = 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc

device = "cuda" if torch.cuda.is_available() else "cpu"

def run_experiment(dropout_p, weight_decay):
     torch.manual_seed(42)

     model = RegularizedClassifier(dropout_p)
     model.to(device)

     criterion = nn.CrossEntropyLoss()

     optimizer = torch.optim.Adam(
          model.parameters(),
          lr = 0.001,
          weight_decay=weight_decay
     )

     for epoch in range(50):
          train_loss, train_acc = train_one_epoch(model, device, optimizer, criterion, train_loader)
          val_loss, val_acc = evaluate(model, device, criterion, val_loader)
     return train_loss, train_acc, val_loss, val_acc

result_a = run_experiment(
     dropout_p=0.0,
     weight_decay=0.0
)

result_b = run_experiment(
     dropout_p=0.5,
     weight_decay=0.0
)

result_c = run_experiment(
     dropout_p=0.0,
     weight_decay=0.01
)

result_d = run_experiment(
     dropout_p=0.5,
     weight_decay=0.01
)

print("A No Regularization:", result_a)
print("B Dropout:", result_b)
print("C Weight Decay:", result_c)
print("D Dropout + Weight Decay:", result_d)