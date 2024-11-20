import torch
import tiktoken
from slm.model import GPTModel
from train import generate_text_simple, text_to_token_ids, token_ids_to_text


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


tokenizer = tiktoken.get_encoding("gpt2")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(GPT_CONFIG_124M)
model.to(device)


model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()


start_context = "How he laughed?"


encoded_context = text_to_token_ids(start_context, tokenizer).to(device)


def generate_text(model, context, max_new_tokens=100, context_size=256):

    token_ids = generate_text_simple(
        model, context, max_new_tokens=max_new_tokens, context_size=context_size
    )

    return token_ids_to_text(token_ids, tokenizer)


generated_text = generate_text(
    model,
    encoded_context,
    max_new_tokens=100,
    context_size=GPT_CONFIG_124M["context_length"],
)


print(generated_text)
