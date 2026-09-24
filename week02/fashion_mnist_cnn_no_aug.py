import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_full_dataset = datasets.FashionMNIST(
    root=".data",
    train=True,
    download=True,
    transform=train_transform
)

eval_full_dataset = datasets.FashionMNIST(
    root=".data",
    train=True,
    download=True,
    transform=eval_transform
)

test_dataset = datasets.FashionMNIST(
    root=".data",
    train=False,
    download=True,
    transform=eval_transform
)

generator = torch.Generator().manual_seed(42)
indices = torch.randperm(
    60000, 
    generator = generator
)

train_indices = indices[:54000]
val_indices = indices[54000:]

train_dataset = Subset(
    train_full_dataset,
    train_indices
)

val_dataset = Subset(
    eval_full_dataset,
    val_indices
)

loader_generator = torch.Generator().manual_seed(42)

train_loader = DataLoader(
    train_dataset,
    batch_size = 64,
    shuffle = True,
    generator = loader_generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size = 64,
    shuffle = False
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

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.01
)

print("train size:", len(train_dataset))
print("val size:", len(val_dataset))
print("test size:", len(test_dataset))

best_val_accuracy = 0.0

for epoch in range(10):
    model.train()

    total_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predictions = outputs.argmax(dim = 1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "best_fashion_mnist_cnn.pth",
        )

    print("epoch:", epoch, 
          "train loss:", total_loss / len(train_loader),
          "val loss:", avg_val_loss,
          "val accuracy:", val_accuracy
          )

model.load_state_dict(
    torch.load(
        "best_fashion_mnist_cnn.pth",
        weights_only = True
    )
)

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim = 1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

accuracy = correct / total

print("test accuracy:", accuracy)