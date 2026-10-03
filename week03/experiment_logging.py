import csv
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

config = {
    "experiment_name": "no_dropout",
    "seed": 42,
    "lr": 0.001,
    "optimizer": "Adam",
    "batch_size": 64,
    "epochs": 50,
    "dropout": 0.0,
    "weight_decay": 0.01
}

def set_seed(seed):
    torch.manual_seed(seed)

def save_experiment(config, result, csv_file):
    record = {
    **config,
    **result
    }

    fieldnames = record.keys()
    file_exists = os.path.exists(csv_file)

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(record)

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

set_seed(config["seed"])

x = torch.randn(1000, 20)

y = (
    x[:, 0]
    + 0.8 * x[:, 1]
    - 0.6 * x[:, 2]
    > 0
).long()

dataset = TensorDataset(x, y)

generator = torch.Generator().manual_seed(config["seed"])

train_dataset, val_dataset = random_split(
    dataset,
    [200, 800],
    generator=generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=False
)

def run_experiment(config):
    set_seed(config["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RegularizedClassifier(
        dropout_p = config["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model, device, optimizer, criterion, train_loader
        )

        val_loss, val_acc = evaluate(
            model, device, criterion, val_loader
        )

    result = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    }

    return result

result = run_experiment(config)

save_experiment(
    config,
    result,
    "week03/experiments.csv"
)

print(result)