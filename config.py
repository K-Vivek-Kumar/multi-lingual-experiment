MODEL_PARAMETERS = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


TRAINING_PARAMETERS = {
    "learning_rate": 5e-4,
    "num_epochs": 10,
    "batch_size": 2,
    "weight_decay": 0.1,
}


MODEL_TOKENS = "gpt2"
