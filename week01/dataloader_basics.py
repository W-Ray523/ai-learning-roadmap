import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

model = nn.Linear(1, 1)

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.01
)

x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0],
    [7.0],
    [8.0],
    [9.0],
    [10.0]
])

y = 2 * x + 1

print("x shape:", x.shape)
print("y shape:", y.shape)

dataset = TensorDataset(x, y)

print("dataset length:", len(dataset))

print("first sample:", dataset[0])

dataloader = DataLoader(
    dataset,
    batch_size = 2,
    shuffle = False
)

for batch_x, batch_y in dataloader:
    print("batch_x:", batch_x)
    print("batch_y:", batch_y)
    print("----")

for epoch in range(100):

    for batch_x, batch_y in dataloader:

        prediction = model(batch_x)

        loss = criterion(prediction, batch_y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    if epoch % 10 == 0:
        print(
            "epoch:", epoch,
            "loss:", loss.item()
        )

print("weight:", model.weight.item())
print("bias:", model.bias.item())