import argparse
import os
import tiktoken
import torch
from torch.utils.data import DataLoader

from config import MODEL_PARAMETERS, MODEL_TOKENS, TRAINING_PARAMETERS
from slm.model import LanguageModel
from slm.tokenizer import TokenizedTextDataset


def parse_arguments():
    parser = argparse.ArgumentParser(description="Continue Training")
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Path to the input text file for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of additional epochs to train.",
    )
    parser.add_argument(
        "--saved_model",
        type=str,
        required=True,
        help="Path to the saved model file to continue training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/",
        help="Directory to save models after each epoch.",
    )
    return parser.parse_args()


def create_dataloader(text, batch_size, seq_length, step_size):
    tokenizer = tiktoken.get_encoding(MODEL_TOKENS)
    dataset = TokenizedTextDataset(text, tokenizer, seq_length, step_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def compute_loss(inputs, targets, model, device):
    inputs, targets = inputs.to(device), targets.to(device)
    all_logits = model(inputs)
    final_loss = torch.nn.functional.cross_entropy(
        all_logits[-1].view(-1, all_logits[-1].size(-1)), targets.view(-1)
    )
    return final_loss


def train_model(model, train_loader, optimizer, device, output_dir, epochs):
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = compute_loss(inputs, targets, model, device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        if epoch % 5 == 0:
            save_path = os.path.join(output_dir, f"model_{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")


def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.input_data):
        raise FileNotFoundError(f"Input file '{args.input_data}' not found.")
    with open(args.input_data, "r", encoding="utf-8") as file:
        text_data = file.read()

    model = LanguageModel(MODEL_PARAMETERS).to(device)
    if not os.path.exists(args.saved_model):
        raise FileNotFoundError(f"Saved model file '{args.saved_model}' not found.")
    model.load_state_dict(torch.load(args.saved_model, weights_only=True))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_PARAMETERS["learning_rate"],
        weight_decay=TRAINING_PARAMETERS["weight_decay"],
    )

    train_loader = create_dataloader(
        text_data,
        batch_size=TRAINING_PARAMETERS["batch_size"],
        seq_length=MODEL_PARAMETERS["context_length"],
        step_size=MODEL_PARAMETERS["context_length"],
    )

    train_model(model, train_loader, optimizer, device, args.output_dir, args.epochs)


if __name__ == "__main__":
    main()
