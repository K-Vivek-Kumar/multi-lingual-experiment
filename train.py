import argparse
import matplotlib.pyplot as plt
import os
import torch
import tiktoken

from config import MODEL_PARAMETERS, MODEL_TOKENS, TRAINING_PARAMETERS
from dataloader import create_dataloader
from slm.model import LanguageModel
from slm.tokenizer import TokenizedTextDataset, decode_tokens, encode_text
from slm.generation import generate_text


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Path to the input text file for training the language model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        help="Number of Epochs",
    )
    return parser.parse_args()


def compute_loss(inputs, targets, model, device):
    inputs, targets = inputs.to(device), targets.to(device)
    all_logits = model(inputs)
    final_loss = torch.nn.functional.cross_entropy(
        all_logits[-1].flatten(0, 1), targets.flatten()
    )
    return final_loss


def evaluate_loss(data_loader, model, device, max_batches=None):
    total_loss = 0.0
    max_batches = max_batches or len(data_loader)
    for batch_index, (inputs, targets) in enumerate(data_loader):
        if batch_index >= max_batches:
            break
        total_loss += compute_loss(inputs, targets, model, device).item()
    return total_loss / max_batches


def evaluate_model(model, train_loader, val_loader, device, max_batches):
    model.eval()
    with torch.no_grad():
        train_loss = evaluate_loss(train_loader, model, device, max_batches)
        val_loss = evaluate_loss(val_loader, model, device, max_batches)
    model.train()
    return train_loss, val_loss


def generate_sample_text(model, tokenizer, device, initial_text):
    model.eval()
    context_limit = model.pos_emb.weight.shape[0]
    context_indices = encode_text(initial_text, tokenizer).to(device)
    with torch.no_grad():
        generated_tokens = generate_text(
            model,
            context_indices,
            max_new_tokens=50,
            context_size=context_limit,
            tokenizer=tokenizer,
        )
        print(decode_tokens(generated_tokens, tokenizer).replace("\n", " "))
    model.train()


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs,
    eval_interval,
    max_batches,
    initial_text,
    tokenizer,
):
    train_losses, val_losses, tokens_processed = [], [], []
    total_tokens, steps = 0, 0

    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = compute_loss(inputs, targets, model, device)
            loss.backward()
            optimizer.step()
            total_tokens += inputs.numel()
            steps += 1
            if steps % eval_interval == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, max_batches
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                tokens_processed.append(total_tokens)
                print(
                    f"Epoch {epoch + 1}"
                    f"Train Loss {train_loss:.3f}, Val Loss {val_loss:.3f}"
                )
        generate_sample_text(model, tokenizer, device, initial_text)

    return train_losses, val_losses, tokens_processed


def plot_training_results(epochs, tokens, train_losses, val_losses):
    fig, ax1 = plt.subplots()
    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, linestyle="-.", label="Val Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twiny()
    ax2.plot(tokens, train_losses, alpha=0)
    ax2.set_xlabel("Tokens Processed")
    fig.tight_layout()


def main(config, params, input_path):
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file '{input_path}' not found. Please provide a valid file path."
        )

    with open(input_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    model = LanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    train_loader = create_dataloader(
        text_data[: int(0.9 * len(text_data))],
        batch_size=params["batch_size"],
        seq_length=config["context_length"],
        step_size=config["context_length"],
        shuffle_data=True,
        drop_extra=True,
        num_workers=0,
    )
    val_loader = create_dataloader(
        text_data[int(0.9 * len(text_data)) :],
        batch_size=params["batch_size"],
        seq_length=config["context_length"],
        step_size=config["context_length"],
        shuffle_data=False,
        drop_extra=False,
        num_workers=0,
    )

    tokenizer = tiktoken.get_encoding(MODEL_TOKENS)
    train_losses, val_losses, tokens_processed = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs=params["num_epochs"],
        eval_interval=5,
        max_batches=1,
        initial_text="The Selfish Giant",
        tokenizer=tokenizer,
    )
    return train_losses, val_losses, tokens_processed, model


if __name__ == "__main__":
    args = parse_arguments()
    if args.epochs:
        TRAINING_PARAMETERS["num_epochs"] = args.epochs
    train_losses, val_losses, tokens_processed, model = main(
        MODEL_PARAMETERS, TRAINING_PARAMETERS, args.input_data
    )
    epochs = torch.linspace(0, TRAINING_PARAMETERS["num_epochs"], len(train_losses))
    plot_training_results(epochs, tokens_processed, train_losses, val_losses)
    plt.savefig("loss.png")
    torch.save(model.state_dict(), "model.pth")
