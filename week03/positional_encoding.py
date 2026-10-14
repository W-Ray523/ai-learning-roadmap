import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embedding_dim):
        super().__init__()

        self.position_encoding = torch.zeros(
            max_len,
            embedding_dim
        )

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2)
            *
            (-torch.log(torch.tensor(10000.0))
             / embedding_dim)
        )

        self.position_encoding[:, 0::2] = torch.sin(
            position * div_term
        )

        self.position_encoding[:, 1::2] = torch.cos(
            position * div_term
        )

max_len = 10
embedding_dim = 16

pe = PositionalEncoding(
    max_len,
    embedding_dim
)

print(pe.position_encoding)
print(pe.position_encoding.shape)

token_ids = torch.tensor([
    [1, 2, 3, 0, 0],
    [1, 2, 4, 3, 0]
])

embedding = nn.Embedding(
    num_embeddings=6,
    embedding_dim=embedding_dim,
    padding_idx=0
)

token_embedding = embedding(token_ids)

position_encoding = pe.position_encoding[:token_ids.shape[1]]

x = token_embedding + position_encoding

print("Token IDs:", token_ids.shape)
print("Token Embedding:", token_embedding.shape)
print("Position Encoding:", position_encoding.shape)
print("Final Output:", x.shape)