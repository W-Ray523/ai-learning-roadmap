import torch
import torch.nn as nn

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = FashionCNN()

x = torch.randn(8, 1, 28, 28)

y = model(x)

print("输入 shape:", x.shape)
print("输出 shape:", y.shape)

print("\n---train模式---")

model.train()

y1 = model(x)
y2 = model(x)

print(torch.allclose(y1, y2))

print("\n---eval模式---")

model.eval()

y3 = model(x)
y4 = model(x)

print(torch.allclose(y3, y4))

bn = model.features[1]

print("\n--- BatchNorm train 模式 ---")

model.train()

before_train = bn.running_mean.clone()
_ = model(x)
after_train = bn.running_mean.clone()

print(torch.allclose(before_train, after_train))


print("\n--- BatchNorm eval 模式 ---")

model.eval()

before_eval = bn.running_mean.clone()
_ = model(x)
after_eval = bn.running_mean.clone()

print(torch.allclose(before_eval, after_eval))