import argparse
import torch
import tiktoken

from config import MODEL_PARAMETERS, MODEL_TOKENS
from slm.generation import generate_text
from slm.model import LanguageModel
from slm.tokenizer import encode_text, decode_tokens


tokenizer = tiktoken.get_encoding(MODEL_TOKENS)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Interactive Text Generation")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model file.",
    )
    return parser.parse_args()


def generate(model, context, max_new_tokens=100, context_size=256):
    token_ids = generate_text(
        model,
        context,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        tokenizer=tokenizer,
        logits_file="logits.txt",
    )
    return decode_tokens(token_ids, tokenizer)


def main():
    args = parse_arguments()

    tokenizer = tiktoken.get_encoding(MODEL_TOKENS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LanguageModel(MODEL_PARAMETERS)
    model.to(device)

    if not args.model_path:
        raise FileNotFoundError(
            f"Model file '{args.model_path}' not found. Please provide a valid path."
        )
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model.eval()

    while True:
        user_input = input("Enter your prompt (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Exiting the program.")
            break

        encoded_context = encode_text(user_input, tokenizer).to(device)

        generated_response = generate(
            model,
            encoded_context,
            max_new_tokens=100,
            context_size=MODEL_PARAMETERS["context_length"],
        )

        truncated_response = generated_response.split("\n")[0]

        print("\nGenerated Text:\n")
        print(truncated_response)
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
