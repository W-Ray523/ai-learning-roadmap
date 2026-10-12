import torch
import torch.nn as nn

vocab = {
    "<pad>": 0,
    "<unk>": 1,
    "I": 2,
    "love": 3,
    "AI": 4,
    "PyTorch": 5
}

def encode_tokens(tokens, vocab):
    token_ids = [
        vocab.get(token, vocab["<unk>"])
        for token in tokens
    ]

    return token_ids

tokens = ["I", "love", "ChatGPT"]

token_ids = encode_tokens(tokens, vocab)

print(token_ids)

sentences = [
    ["I", "love", "AI"],
    ["I", "love", "ChatGPT", "PyTorch"]
]

encoded_sentences = []

for tokens in sentences:
    ids = encode_tokens(tokens, vocab)
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
    dtype = torch.long
)

mask_tensor = torch.tensor(
    attention_masks,
    dtype = torch.long
)

print("Token IDs shape:", padded_tensor.shape)
print("Mask shape:", mask_tensor.shape)

embedding = nn.Embedding(
    num_embeddings=len(vocab),
    embedding_dim=4,
    padding_idx=vocab["<pad>"]
)

embedded = embedding(padded_tensor)

print("Embedding shape:", embedded.shape)