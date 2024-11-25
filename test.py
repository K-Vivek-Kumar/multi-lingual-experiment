import torch
import tiktoken


from config import MODEL_PARAMETERS, MODEL_TOKENS
from slm.generation import generate_text
from slm.model import LanguageModel
from slm.tokenizer import encode_text, decode_tokens


tokenizer = tiktoken.get_encoding(MODEL_TOKENS)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LanguageModel(MODEL_PARAMETERS)
model.to(device)


model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()


def generate(model, context, max_new_tokens=100, context_size=256):
    token_ids = generate_text(
        model, context, max_new_tokens=max_new_tokens, context_size=context_size
    )
    return decode_tokens(token_ids, tokenizer)


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

    print("\nGenerated Text:\n")
    print(generated_response)
    print("\n" + "=" * 50 + "\n")
