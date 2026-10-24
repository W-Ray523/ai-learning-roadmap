import torch
import torch.nn as nn
import math

def scaled_attention(Q, K, V):
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    output = weights @ V
    return output, weights

torch.manual_seed(27)

X = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],
])

q1 = nn.Linear(4, 2, bias=False)
k1 = nn.Linear(4, 2, bias=False)
v1 = nn.Linear(4, 2, bias=False)

q2 = nn.Linear(4, 2, bias=False)
k2 = nn.Linear(4, 2, bias=False)
v2 = nn.Linear(4, 2, bias=False)

Q1, K1, V1 = q1(X), k1(X), v1(X)
Q2, K2, V2 = q2(X), k2(X), v2(X)

output1, weights1 = scaled_attention(Q1, K1, V1)
output2, weights2 = scaled_attention(Q2, K2, V2)

combined = torch.cat([output1, output2], dim=-1)

print("X shape:", X.shape)
print("Head 1 Q/K/V shapes:", Q1.shape, K1.shape, V1.shape)
print("Head 2 Q/K/V shapes:", Q2.shape, K2.shape, V2.shape)
print("Head 1 weights shape:", weights1.shape)
print("Head 2 weights shape:", weights2.shape)
print("Head 1 output shape:", output1.shape)
print("Head 2 output shape:", output2.shape)
print("Combined shape:", combined.shape)
print("\nHead 1 weights:\n", weights1.detach())
print("\nHead 2 weights:\n", weights2.detach())

out_proj = nn.Linear(4, 4, bias=False)
final_output = out_proj(combined)

print("\nFinal output shape:", final_output.shape)
print("Final output:\n", final_output.detach())