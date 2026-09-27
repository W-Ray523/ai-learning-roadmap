import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class GestureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predicted = outputs.argmax(dim = 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predicted = outputs.argmax(dim = 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy

model = GestureCNN()

x = torch.randn(32, 3, 224, 224)

outputs = model(x)

print("outputs.shape:", outputs.shape)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GestureCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

x = torch.randn(8, 3, 224, 224)
y = torch.randint(0, 2, (8,))

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

train_loss, train_acc = train_one_epoch(
    model, loader, criterion, optimizer, device
)

val_loss, val_acc = evaluate(
    model, loader, criterion, device
)

print("train_loss:", train_loss)
print("train_acc:", train_acc)
print("val_loss:", val_loss)
print("val_acc:", val_acc)