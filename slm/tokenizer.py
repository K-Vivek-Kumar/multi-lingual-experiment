import torch
from torch.utils.data import Dataset


class TokenizedTextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_length, step_size):
        self.input_sequences = []
        self.target_sequences = []
        tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        for start in range(0, len(tokens) - seq_length, step_size):
            self.input_sequences.append(
                torch.tensor(tokens[start : start + seq_length])
            )
            self.target_sequences.append(
                torch.tensor(tokens[start + 1 : start + seq_length + 1])
            )

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, index):
        return self.input_sequences[index], self.target_sequences[index]


def encode_text(text, tokenizer):
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)


def decode_tokens(tokens, tokenizer):
    return tokenizer.decode(tokens.squeeze(0).tolist())
