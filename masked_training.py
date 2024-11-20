import tiktoken
import torch
import random
from torch.utils.data import Dataset, DataLoader
from slm.model import GPTModel


class MaskedGPTDataset(Dataset):
    def __init__(self, text, tokenizer, context_length, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.mask_prob = mask_prob

        lines = text.split("\n")
        self.data = [tokenizer.encode(line) for line in lines if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_ids = self.data[idx][: self.context_length]
        input_ids = token_ids.copy()
        target_ids = token_ids.copy()

        for i in range(len(input_ids)):
            if random.random() < self.mask_prob:
                input_ids[i] = self.tokenizer.encode("<mask>")[0]

        input_ids = self._pad_sequence(input_ids)
        target_ids = self._pad_sequence(target_ids)

        return torch.tensor(input_ids), torch.tensor(target_ids)

    def _pad_sequence(self, sequence):
        return sequence + [0] * (self.context_length - len(sequence))


def create_masked_dataloader(
    text, tokenizer, context_length, batch_size, mask_prob, shuffle=True
):
    dataset = MaskedGPTDataset(
        text=text,
        tokenizer=tokenizer,
        context_length=context_length,
        mask_prob=mask_prob,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def masked_training_step(model, input_batch, target_batch, optimizer, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_with_masking(model, data_loader, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for input_batch, target_batch in data_loader:
            loss = masked_training_step(
                model, input_batch, target_batch, optimizer, device
            )
            total_loss += loss

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


def main_with_masking(pretrained_path, text_file, gpt_config, settings):
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(text_file, "r", encoding="utf-8") as file:
        text_data = file.read()

    tokenizer = tiktoken.get_encoding("gpt2")

    model = GPTModel(gpt_config)
    model.load_state_dict(torch.load(pretrained_path))
    model.to(device)

    data_loader = create_masked_dataloader(
        text=text_data,
        tokenizer=tokenizer,
        context_length=gpt_config["context_length"],
        batch_size=settings["batch_size"],
        mask_prob=0.15,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )

    train_with_masking(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=settings["num_epochs"],
    )

    torch.save(model.state_dict(), "model_masked_finetuned.pth")


if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 5,
        "batch_size": 4,
        "weight_decay": 0.1,
    }

    pretrained_model_path = "trained_transformer_model.pth"
    text_file_path = "the-verdict.txt"

    main_with_masking(
        pretrained_model_path, text_file_path, GPT_CONFIG_124M, OTHER_SETTINGS
    )
