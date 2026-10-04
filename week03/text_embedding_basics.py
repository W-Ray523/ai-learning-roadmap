import torch
import torch.nn as nn

vocab = {
    "<pad>": 0,
    "I": 1,
    "love": 2,
    "AI": 3,
    "and": 4,
    "PyTorch": 5
}

tokens = ["I", "love", "AI"]

token_ids = torch.tensor(
    [vocab[token] for token in tokens]
)

embedding = nn.Embedding(
    num_embeddings=len(vocab),
    embedding_dim=4,
    padding_idx=0
)

embedded = embedding(token_ids)

sentences = [
    ["I", "love", "AI"],
    ["I", "love", "AI", "and", "PyTorch"]
]

max_len = max(len(sentence) for sentence in sentences)
print(max_len)

padded_ids = []
attention_masks = []

for sentence in sentences:
    ids = [vocab[token] for token in sentence]

    pad_len = max_len - len(ids)

    ids = ids + [vocab["<pad>"]] * pad_len

    mask = [1] * len(sentence) + [0] * pad_len

    padded_ids.append(ids)
    attention_masks.append(mask)

padded_ids = torch.tensor(padded_ids)
attention_masks = torch.tensor(attention_masks)

print(padded_ids)
print(padded_ids.shape)

print(attention_masks)
print(attention_masks.shape)

embedded = embedding(padded_ids)

print(embedded.shape)
print(embedding.weight[0])