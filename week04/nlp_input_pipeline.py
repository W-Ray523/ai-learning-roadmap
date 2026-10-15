import torch
import torch.nn as nn

vocab = {
    "<pad>": 0,
    "<unk>": 1,
    "I": 2,
    "love": 3,
    "AI": 4,
    "PyTorch": 5,
    "and": 6
}

sentences = [
    ["I", "love", "AI"],
    ["I", "love", "PyTorch", "and", "AI"]
]

def encode_tokens(tokens, vocab):
    token_ids = [
        vocab.get(token, vocab["<unk>"])
        for token in tokens
    ]
    return token_ids

encoded_sentences = []

for sentence in sentences:
    ids = encode_tokens(sentence, vocab)
    encoded_sentences.append(ids)

print(encoded_sentences)

def pad_and_mask(encoded_sentences, pad_id):
    max_len = max(len(ids) for ids in encoded_sentences)

    padded_sentences = []
    attention_masks = []

    for ids in encoded_sentences:
        pad_len = max_len - len(ids)

        padded_ids = ids + [pad_id] * pad_len
        mask = [1] * len(ids) + [0] * pad_len

        padded_sentences.append(padded_ids)
        attention_masks.append(mask)

    return padded_sentences, attention_masks

padded_sentences, attention_masks = pad_and_mask(
    encoded_sentences,
    vocab["<pad>"]
)

print("Padded:", padded_sentences)
print("Mask:", attention_masks)

padded_tensor = torch.tensor(
    padded_sentences,
    dtype=torch.long
)

mask_tensor = torch.tensor(
    attention_masks,
    dtype=torch.long
)

embedding_dim = 16

embedding = nn.Embedding(
    num_embeddings = len(vocab),
    embedding_dim = embedding_dim,
    padding_idx = vocab["<pad>"]
)

token_embedding = embedding = embedding(padded_tensor)

print("Token IDs:", padded_tensor.shape)
print("Mask:", mask_tensor.shape)
print("Embedding:", token_embedding.shape)

seq_len = padded_tensor.shape[1]

position_encoding = torch.zeros(
    seq_len,
    embedding_dim
)

position = torch.arange(seq_len).unsqueeze(1)

div_term = torch.exp(
    torch.arange(0, embedding_dim, 2)
    * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
)

position_encoding[:, 0::2] = torch.sin(
    position * div_term
)

position_encoding[:, 1::2] = torch.cos(
    position * div_term
)

final_embedding = token_embedding + position_encoding

print("Position Encoding:", position_encoding.shape)
print("Final Embedding:", final_embedding.shape)
