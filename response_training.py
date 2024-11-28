import argparse
import os
import torch
import tiktoken
from torch.utils.data import DataLoader

from config import MODEL_PARAMETERS, MODEL_TOKENS, TRAINING_PARAMETERS
from slm.model import LanguageModel
from slm.tokenizer import TokenizedTextDataset, decode_tokens, encode_text
from slm.generation import generate_text


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune an existing model")
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Path to the input text file for fine-tuning the language model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=5,
        help="Number of epochs for fine-tuning",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pre-trained model checkpoint",
    )
    parser.add_argument(
        "--end_token",
        type=str,
        required=True,
        help="End token to indicate when to end the generated responses",
    )
    return parser.parse_args()


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


def compute_loss(inputs, targets, model, device):
    inputs, targets = inputs.to(device), targets.to(device)
    all_logits = model(inputs)
    final_loss = torch.nn.functional.cross_entropy(
        all_logits[-1].flatten(0, 1), targets.flatten()
    )
    return final_loss


def fine_tune_model(
    model, train_loader, optimizer, device, epochs, end_token, tokenizer
):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = compute_loss(inputs, targets, model, device)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}, Loss: {loss.item():.3f}")

        generate_sample_text(model, tokenizer, device, end_token)


def generate_sample_text(model, tokenizer, device, end_token):
    model.eval()
    context_limit = model.pos_emb.weight.shape[0]

    initial_text = "Response: "
    context_indices = encode_text(initial_text, tokenizer).to(device)

    with torch.no_grad():
        generated_tokens = generate_text(
            model,
            context_indices,
            max_new_tokens=100,
            context_size=context_limit,
            tokenizer=tokenizer,
        )

    generated_text = decode_tokens(generated_tokens, tokenizer)
    if end_token in generated_text:
        print(f"Generated text (ending with {end_token}):")
        print(generated_text.split(end_token)[0])
    else:
        print("End token not found in generated response.")

    model.train()


def main(config, params, input_path, model_path, end_token):
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file '{input_path}' not found. Please provide a valid file path."
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint '{model_path}' not found. Please provide a valid model path."
        )

    with open(input_path, "r", encoding="utf-8") as file:
        text_data = file.readlines()

    model = LanguageModel(config).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    tokenizer = tiktoken.get_encoding(MODEL_TOKENS)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    text_data = "\n".join(text_data)

    train_loader = create_dataloader(
        text_data,
        batch_size=params["batch_size"],
        seq_length=config["context_length"],
        step_size=config["context_length"],
        shuffle_data=True,
        drop_extra=True,
        num_workers=0,
    )

    fine_tune_model(
        model,
        train_loader,
        optimizer,
        device,
        epochs=params["num_epochs"],
        end_token=end_token,
        tokenizer=tokenizer,
    )

    torch.save(model.state_dict(), "fine_tuned_model.pth")


if __name__ == "__main__":
    args = parse_arguments()
    main(
        MODEL_PARAMETERS,
        TRAINING_PARAMETERS,
        args.input_data,
        args.model_path,
        args.end_token,
    )
