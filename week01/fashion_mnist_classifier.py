import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn

transforms = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root = "./data",
    train = True,
    download = True,   #表示加载训练集
    transform = transforms
)

test_dataset = datasets.FashionMNIST(
    root = "./data",
    train = False,     #表示加载测试集
    download = True,
    transform = transforms
)

print("train size:", len(train_dataset))
print("test size:", len(test_dataset))

image, label = train_dataset[0]

print("image shape:", image.shape)
print("label:", label)

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

images, labels = next(iter(train_loader))

print("batch images shape:", images.shape)
print("batch labels shape:", labels.shape)

flatten = torch.nn.Flatten()

flat_images = flatten(images)

print("befor flatten:", images.shape)
print("after flatten:", flat_images.shape)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

outputs = model(images)

print("model output shape:", outputs.shape)
print("first image output:", outputs[0])

predicted_class = outputs[0].argmax()

print("predicted class:", predicted_class.item())
print("true label:", labels[0].item())

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.01
)

for epoch in range(5):
    total_loss = 0.0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(
        "epoch:",
        epoch,
        "loss:",
        total_loss / len(train_loader)
    )

model.eval()   #表示模型进入评估模式

correct = 0
total = 0

with torch.no_grad():    #表示测试时不计算梯度
    for images, labels in test_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim = 1)    #dim=1表示对每一张图片的10个类别得分中，找最大值的位置
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total

print("test accuracy:", accuracy)