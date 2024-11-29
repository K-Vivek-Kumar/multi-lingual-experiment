import torch


def generate_text(
    model, idx, max_new_tokens, context_size, tokenizer, logits_file=None
):
    file = None
    if logits_file:
        file = open(logits_file, "w", encoding="utf-8")

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits_list = model(idx_cond)

        final_logits = logits_list[-1][:, -1, :]

        probas = torch.softmax(final_logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)

        decoded_token = tokenizer.decode([idx_next.item()])

        if file:
            file.write(f"Final Logits: {final_logits.squeeze().tolist()}\n")
            file.write(f"Generated token: {decoded_token}\n")

    if file:
        file.close()

    return idx
