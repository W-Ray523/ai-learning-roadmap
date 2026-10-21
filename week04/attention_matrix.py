import torch

Q = torch.tensor([
    [2.0, 0.0],
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

scores = torch.matmul(Q, K.T)

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("K.T shape:", K.T.shape)
print("Scores shape:", scores.shape)
print("Scores:")
print(scores)

attention_weights = torch.softmax(scores, dim=1)

print("Attention Weights:")
print(attention_weights)

print("Weights Shape:", attention_weights.shape)

row_sums = attention_weights.sum(dim=1)

print("Row Sums:", row_sums)

output = torch.matmul(attention_weights, V)

print("V Shape:", V.shape)
print("Output Shape:", output.shape)

print("Attention Output:")
print(output)