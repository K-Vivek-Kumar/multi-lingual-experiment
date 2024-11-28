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


class MyTokenizer:
    def __init__(self, vocab_size, max_token_size=3):
        self.vocab_size = vocab_size
        self.max_token_size = max_token_size
        self.word2idx = []
        self.idx2word = {}

    def generate_tokens(self, input_path):
        freq_map = {}
        with open(input_path, "r", encoding="utf-8") as w:
            data = w.read()
        for i in range(1, self.max_token_size + 1):
            for j in range(len(data) - i + 1):
                token = data[j : j + i]
                if token in freq_map:
                    freq_map[token] += 1
                else:
                    freq_map[token] = 1
        most_freq = dict(
            sorted(freq_map.items(), key=lambda item: item[1], reverse=True)
        )
        self.word2idx = [key for key, value in most_freq.items()][: self.vocab_size]
        self.idx2word = {i: word for i, word in enumerate(self.word2idx)}

    def encode(self, sentence):
        i = 0
        encoded_sentence = []
        while i < len(sentence):
            current_size_check = self.max_token_size
            token_found = False
            while current_size_check > 0:
                token = sentence[i : i + current_size_check]
                if token in self.word2idx:
                    encoded_sentence.append(self.word2idx.index(token))
                    i += current_size_check
                    token_found = True
                    break
                current_size_check -= 1
            if not token_found:
                i += 1
        return encoded_sentence

    def decode(self, encoded_sentence):
        decoded_sentence = ""
        for i in encoded_sentence:
            decoded_sentence += self.idx2word.get(i, "")
        return decoded_sentence
