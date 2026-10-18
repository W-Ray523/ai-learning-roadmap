import torch

Q = torch.tensor([2.0, 3.0])

K1 = torch.tensor([1.0, 2.0])
K2 = torch.tensor([1.0, 1.0])

V1 = torch.tensor([1.0, 0.0])
V2 = torch.tensor([0.0, 10.0])

score1 = torch.dot(Q, K1)
score2 = torch.dot(Q, K2)

scores = torch.stack([score1, score2])

weights = torch.softmax(scores, dim=0)

output = weights[0] * V1 + weights[1] * V2

print("Scores:", scores)
print("Weights:", weights)
print("Output:", output)
print("V1 Shape:", V1.shape)
print("Output Shape:", output.shape)