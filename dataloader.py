import tiktoken
from torch.utils.data import DataLoader

from config import MODEL_TOKENS
from slm.tokenizer import TokenizedTextDataset


def create_dataloader(
    text, batch_size, seq_length, step_size, shuffle_data, drop_extra, num_workers
):
    tokenizer = tiktoken.get_encoding(MODEL_TOKENS)
    dataset = TokenizedTextDataset(text, tokenizer, seq_length, step_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_data,
        drop_last=drop_extra,
        num_workers=num_workers,
    )
