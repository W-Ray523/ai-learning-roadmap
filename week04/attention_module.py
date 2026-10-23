import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):

        d_k = Q.shape[-1]

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        ) / math.sqrt(d_k)

        weights = torch.softmax(scores, dim=-1)

        output = torch.matmul(weights, V)

        return output, weights, scores

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
    [1.0, 0.0],
    [0.0, 2.0],
    [3.0, 3.0]
])

attention = ScaledDotProductAttention()

output, weights, scores = attention(Q, K, V)

print("Scores Shape:", scores.shape)
print("Weights Shape:", weights.shape)
print("Output Shape:", output.shape)

print("Scores:")
print(scores)

print("Weights:")
print(weights)

print("Output:")
print(output)

Q_new = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0]
])

K_new = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0]
])

V_new = torch.tensor([
    [10.0, 0.0],
    [0.0, 10.0]
])

output_new, weights_new, scores_new = attention(
    Q_new, K_new, V_new
)

print("New Scores Shape:", scores_new.shape)
print("New Weights Shape:", weights_new.shape)
print("New Output Shape:", output_new.shape)

print("New Output:")
print(output_new)