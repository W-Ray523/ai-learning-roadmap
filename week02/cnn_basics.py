import torch
import torch.nn as nn

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

x = torch.randn(64, 1, 28, 28)

outputs = model(x)

print("output shape:", outputs.shape)

total_params = sum(p.numel() for p in model.parameters())

print("total parameters:", total_params)