import torch
import torch.nn as nn

# w = torch.tensor([1.0], requires_grad = True)

# optimizer = torch.optim.SGD([w], lr = 0.1)

# for step in range(10):

#     loss = (w - 3) ** 2

#     optimizer.zero_grad()

#     loss.backward()

#     optimizer.step()

#     print(
#         "step:", step,
#         "w:", w.item(),
#         "loss:", loss.item(),
#         "grad:", w.grad.item()
#     )

#     w.grad.zero_()

print("----------------------------------------------------------")

x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

y = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0]
])

model = nn.Linear(1, 1)

print("initial weight:", model.weight)
print("initial bias:", model.bias)

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.01
)

for epoch in range(1000):

    prediction = model(x)

    loss = criterion(prediction, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 100 == 0:
        print(
            "epoch:", epoch,
            "loss:", loss.item()
        )

print("trained weight:", model.weight)
print("trained bias:", model.bias)