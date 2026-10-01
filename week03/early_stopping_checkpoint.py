import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

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

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [800, 200],
    generator=generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

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
    accuracy = correct / total

    return avg_loss, accuracy

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
    accuracy = correct / total

    return avg_loss, accuracy

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleClassifier().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
     model.parameters(),
     lr = 0.01
)

epochs = 50
patience = 5

best_val_loss = float("inf")
patience_counter = 0

for epoch in range(epochs):

     train_loss, train_acc = train_one_epoch(
          model,
          device,
          optimizer,
          criterion,
          train_loader
     )

     val_loss, val_acc = evaluate(
          model,
          device,
          criterion,
          val_loader
     )

     print(
          f"Epoch {epoch + 1:02d} | "
          f"train_loss={train_loss:.4f} | "
          f"train_acc={train_acc:.4f} | "
          f"val_loss={val_loss:.4f} | "
          f"val_acc={val_acc:.4f}"
     )

     if val_loss < best_val_loss:
          best_val_loss = val_loss
          patience_counter = 0

          torch.save(
               model.state_dict(),
               "best_model.pth"
          )

     else:
          patience_counter += 1

     if patience_counter >= patience:
          print(f"Early stopping at epoch {epoch + 1}")
          break

best_model = SimpleClassifier().to(device)

best_model.load_state_dict(
     torch.load("best_model.pth")
)

val_loss, val_acc = evaluate(
     best_model,
     device,
     criterion,
     val_loader
)

print(val_loss, val_acc)