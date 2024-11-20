import torch
from slm.model import GPTModel
import tiktoken
import matplotlib.pyplot as plt

from train import create_dataloader_v1, plot_losses, train_model_simple


def continue_training(
    saved_model_path,
    gpt_config,
    new_settings,
    text_data,
    additional_epochs,
    start_context,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPTModel(gpt_config)
    model.load_state_dict(torch.load(saved_model_path, weights_only=True))
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=new_settings["learning_rate"],
        weight_decay=new_settings["weight_decay"],
    )

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=new_settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=new_settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=additional_epochs,
        eval_freq=5,
        eval_iter=1,
        start_context=start_context,
        tokenizer=tokenizer,
    )

    return train_losses, val_losses, tokens_seen, model


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

    CONTINUE_SETTINGS = {
        "learning_rate": 1e-4,
        "batch_size": 4,
        "weight_decay": 0.05,
    }

    file_path = "the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    train_losses, val_losses, tokens_seen, model = continue_training(
        saved_model_path="trained_transformer_model.pth",
        gpt_config=GPT_CONFIG_124M,
        new_settings=CONTINUE_SETTINGS,
        text_data=text_data,
        additional_epochs=5,
        start_context="Turtle and Rabbit Story",
    )

    torch.save(model.state_dict(), "trained_transformer_model.pth")

    epochs_tensor = torch.linspace(0, len(train_losses), len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("continued_loss.pdf")
