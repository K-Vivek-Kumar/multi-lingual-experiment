import torch
import torch.nn as nn
from torch.nn import Module as NeuralNet


class GELU(NeuralNet):
    def __init__(self):
        NeuralNet.__init__(self)

    def forward(self, input_tensor):
        tanh_multiplier = torch.sqrt(torch.tensor(2.0 / torch.pi))
        cubic_component = 0.044715 * torch.pow(input_tensor, 3)
        activation = (
            0.5
            * input_tensor
            * (1 + torch.tanh(tanh_multiplier * (input_tensor + cubic_component)))
        )
        return activation


class LayerNorm(NeuralNet):
    def __init__(self, emb_dim):
        NeuralNet.__init__(self)
        self.epsilon = 1e-5
        self.scale_param = nn.Parameter(torch.ones(emb_dim))
        self.shift_param = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, input_tensor):
        mean_value = input_tensor.mean(dim=-1, keepdim=True)
        variance_value = input_tensor.var(dim=-1, keepdim=True, unbiased=False)
        normalized_tensor = (input_tensor - mean_value) / torch.sqrt(
            variance_value + self.epsilon
        )
        return self.scale_param * normalized_tensor + self.shift_param


class FeedForward(NeuralNet):
    def __init__(self, cfg):
        NeuralNet.__init__(self)
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
