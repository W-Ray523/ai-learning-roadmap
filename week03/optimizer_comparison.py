import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

torch.manual_seed(42)

x = torch.randn(1000, 4)

y = (
    x[:, 0]
    + x[:, 1]
    - x[:, 2]
    + 0.5 * x[:, 3]
    > 0
).long()

dataset = TensorDataset(x, y)

def make_loader():
    generator = torch.Generator().manual_seed(42)

    return DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        generator=generator
    )

def train_model(model, loader, criterion, optimizer, epochs=20, scheduler = None):
    for epoch in range(epochs):
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for x_batch, y_batch in loader:
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
        accuracy = correct / total

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:02d} | "
            f"lr={current_lr:.5f} | "
            f"loss={avg_loss:.4f} | "
            f"acc={accuracy:.4f}"
        )

        if scheduler is not None:
            scheduler.step()

    return avg_loss, accuracy

criterion = nn.CrossEntropyLoss()

torch.manual_seed(42)
model_sgd = SimpleClassifier()

loader_sgd = make_loader()

optimizer_sgd = torch.optim.SGD(
    model_sgd.parameters(),
    lr = 0.1
)

scheduler_sgd = torch.optim.lr_scheduler.StepLR(
    optimizer_sgd,
    step_size = 5,
    gamma = 0.1
)

print("\n======== SGD ========")

sgd_loss, sgd_acc = train_model(
    model_sgd,
    loader_sgd,
    criterion,
    optimizer_sgd,
    epochs = 20,
    scheduler = scheduler_sgd
)

torch.manual_seed(42)
model_adam = SimpleClassifier()

loader_adam = make_loader()

optimizer_adam = torch.optim.Adam(
    model_adam.parameters(),
    lr = 0.01
)

print("\n======== Adam ========")

adam_loss, adam_acc = train_model(
    model_adam,
    loader_adam,
    criterion,
    optimizer_adam,
    epochs = 20
)

torch.manual_seed(42)
model_adamw = SimpleClassifier()

loader_adamw = make_loader()

optimizer_adamw = torch.optim.AdamW(
    model_adamw.parameters(),
    lr = 0.001,
    weight_decay=0.01
)

print("\n======== AdamW ========")

adamw_loss, adamw_acc = train_model(
    model_adamw,
    loader_adamw,
    criterion,
    optimizer_adamw,
    epochs = 20
)

print("\n======== Final Results ========")

print(
    f"SGD   | "
    f"loss = {sgd_loss:.4f} | "
    f"acc = {sgd_acc:.4f}"
)

print(
    f"Adam   | "
    f"loss = {adam_loss:.4f} | "
    f"acc = {adam_acc:.4f}"
)

print(
    f"AdamW   | "
    f"loss = {adamw_loss:.4f} | "
    f"acc = {adamw_acc:.4f}"
)