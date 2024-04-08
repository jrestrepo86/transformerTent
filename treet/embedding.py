import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, input_dim, model_dim, max_len=5000, embd_dropout=0.1):
        super(Embedding, self).__init__()

        # positional embedding
        self.PositionalEmbedding(model_dim, max_len)

        # value embedding
        self.value_embedding = nn.Linear(input_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(p=embd_dropout)

    def PositionalEmbedding(self, model_dim, max_len, n=10000.0):
        assert (
            model_dim % 2 == 0
        ), f"Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={model_dim})"
        pe = torch.zeros(max_len, model_dim, dtype=torch.float, requires_grad=False)
        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 10000^(2i/model_dim), i is the index of embedding
        denominators = torch.pow(n, 2 * torch.arange(0, model_dim // 2) / model_dim)
        # sin(pos/10000^(2i/model_dim))
        pe[:, 0::2] = torch.sin(positions / denominators)
        # cos(pos/10000^(2i/model_dim))
        pe[:, 1::2] = torch.cos(positions / denominators)
        # pe shape (1, seq_len, model_dim)
        self.position_embedding = pe.unsqueeze(0)

    def forward(self, x):
        """
        x shape (samples, seq_len, input dim)
        output shape (samples, seq_len, model dim)
        """
        x = self.value_embedding(x) + self.position_embedding[:, : x.size(1)]
        return self.dropout(x)
