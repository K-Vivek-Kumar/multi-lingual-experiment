import torch
import torch.nn as nn
from torch.nn import Module as NeuralNet

from slm.blocks import LayerNorm
from slm.decoder import TransformerBlock


class LanguageModel(NeuralNet):
    def __init__(self, cfg):
        NeuralNet.__init__(self)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "transformer": TransformerBlock(cfg),
                        "norm": LayerNorm(cfg["emb_dim"]),
                        "head": nn.Linear(
                            cfg["emb_dim"], cfg["vocab_size"], bias=False
                        ),
                    }
                )
                for _ in range(cfg["n_layers"])
            ]
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        intermediate_logits = []

        for block in self.blocks:
            x = block["transformer"](x)
            normed_x = block["norm"](x)
            logits = block["head"](normed_x)
            intermediate_logits.append(logits)

        return intermediate_logits
