import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root = ".data",
    train = True,
    download = True,
    transform = transform
)

test_dataset = datasets.FashionMNIST(
    root = ".data",
    train = False,
    download = True,
    transform = transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size = 64,
    shuffle = True
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 64,
    shuffle = False
)

model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),   #表示输入通道：1；输出通道：8，卷积核：3*3;padding:1
    nn.ReLU(),
    nn.MaxPool2d(2, 2),    #把高和宽减半

    nn.Conv2d(8, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Flatten(),

    nn.Linear(16 * 7 * 7, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.01
)

print("train size:", len(train_dataset))
print("test size:", len(test_dataset))

model.train()

for epoch in range(10):
    total_loss = 0.0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print("epoch:", epoch, "loss:", total_loss / len(train_loader))

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)

        predictions = outputs.argmax(dim = 1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

accuracy = correct / total

print("test accuracy:", accuracy)