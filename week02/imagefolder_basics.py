from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    root="week02/gesture_demo",
    transform = transform
)

loader = DataLoader(
    dataset,
    batch_size = 2,
    shuffle = True
)

print("dataset size:", len(dataset))
print("class names:", dataset.classes)
print("class to idx:", dataset.class_to_idx)

image, label = dataset[0]

print("image shape:", image.shape)
print("label:", label)

images, labels = next(iter(loader))

print("batch images shape:", images.shape)
print("batch labels shape:", labels.shape)
print("batch labels:", labels)

model = nn.Sequential(
    nn.Conv2d(
        in_channels = 3, 
        out_channels = 16, 
        kernel_size = 3, 
        padding = 1
    ),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(
        in_channels = 16,
        out_channels = 32,
        kernel_size = 3,
        padding = 1
    ),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(
        in_channels = 32,
        out_channels = 64,
        kernel_size = 3,
        padding = 1
    ),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Flatten(),

    nn.Linear(64 * 28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
)

outputs = model(images)

print("model output shape:", outputs.shape)
print("outputs:", outputs)