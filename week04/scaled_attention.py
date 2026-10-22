import torch
import math

Q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

K = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

V = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 2.0, 0.0, 2.0],
    [3.0, 3.0, 3.0, 3.0]
])

scores = torch.matmul(Q, K.T)

weights_before = torch.softmax(scores, dim = -1)

d_k = Q.shape[-1]

scaled_scores = scores / math.sqrt(d_k)

weights_after = torch.softmax(scaled_scores, dim=-1)

print("d_k:", d_k)
print("Scores:", scores)

print("Weights Before Scaling:")
print(weights_before)

print("Weights After Scaling:")
print(weights_after)

output_before = torch.matmul(weights_before, V)

output_after = torch.matmul(weights_after, V)

print("Output Before Scaling:")
print(output_before)

print("Output After Scaling:")
print(output_after)

print("Output Shape:", output_after.shape)

print("Row Sums:", weights_after.sum(dim=-1))