import torch
import torch.nn as nn

# class SimpleNet(nn.Module):

#     def __init__(self):
#         super().__init__()

#         self.fc1 = nn.Linear(4, 8)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(8, 2)

#     def forward(self, x):
#         print("input:", x.shape)

#         x = self.fc1(x)
#         print("after fc1:", x.shape)

#         x = self.relu(x)
#         print("after relu:", x.shape)

#         x = self.fc2(x)
#         print("after fc2:", x.shape)

#         print("fc1 weight shape:", model.fc1.weight.shape)
#         print("fc1 bias shape:", model.fc1.bias.shape)

#         print("fc2 weight shape:", model.fc2.weight.shape)
#         print("fc2 bias shape:", model.fc2.bias.shape)

#         total_params = sum(p.numel() for p in model.parameters())
#         print("total parameters:", total_params)

#         for name, param in model.named_parameters():
#             print(name, param.shape)

#         return x

# model = SimpleNet()

# print(model)

# # test = torch.tensor([-2.0, -0.5, 0.0, 1.5, 3.0])

# # relu = nn.ReLU()

# # print("before:", test)
# # print("after:", relu(test))

# x = torch.rand(5, 4)

# output = model(x)

# print("input shape:", x.shape)
# print("output:", output)
# print("output shape:", output.shape)

class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 12)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(12, 4)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(4, 2)

    def forward(self, x):
        print("input shape:", x.shape)

        x = self.fc1(x)
        print("after fc1:", x.shape)

        x = self.relu1(x)
        print("after relu1:", x.shape)

        x = self.fc2(x)
        print("after fc2:", x.shape)

        x = self.relu2(x)
        print("after relu2:", x.shape)

        x = self.fc3(x)
        print("after fc3:", x.shape)

        print("output:", x)

        return x

model = Mymodel()
x = torch.rand(5, 6)
output = model(x)
print(model)
print("total parameters:", sum(p.numel() for p in model.parameters()))
