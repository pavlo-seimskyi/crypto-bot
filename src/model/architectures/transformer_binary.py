import numpy as np
import torch


class TransformerBinaryClassifier(torch.nn.Module):
    def __init__(
        self,
        n_features: int,
        n_embedding_dims: int,
        n_head: int,
        n_layers: int,
        n_out: int = 1,
        dropout: float = 0.1,
    ):
        super(TransformerBinaryClassifier, self).__init__()
        self.n_embedding_dims = n_embedding_dims
        self.embedding = torch.nn.Linear(n_features, n_embedding_dims)
        self.pos_encoder = PositionalEncoding(n_embedding_dims, dropout)
        self.layer_norm = torch.nn.LayerNorm(n_embedding_dims)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            n_embedding_dims, n_head
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, n_layers
        )
        self.output_layer = torch.nn.Linear(n_embedding_dims, n_out)
        self.batch_norm = torch.nn.BatchNorm1d(n_out)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.embedding(x)
        out = self.pos_encoder(out)
        out = self.layer_norm(out)
        # PyTorch's TransformerEncoder expects input shape
        # (sequence_length, batch_size, n_embedding_dims), hence permutation twice
        out = out.permute(1, 0, 2)
        out = self.transformer_encoder(out)
        out = out.permute(1, 0, 2)
        out = out[:, -1, :]  # output of the last sequence step
        # out = out.mean(dim=1)  # mean of all sequence steps
        out = self.output_layer(out)
        out = self.batch_norm(out)
        out = self.sigmoid(out)
        return out.squeeze(dim=1)


class PositionalEncoding(torch.nn.Module):
    def __init__(
        self, n_embedding_dims: int, dropout: float = 0.1, max_len: int = 5000
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, n_embedding_dims)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n_embedding_dims, 2).float()
            * (-np.log(10000.0) / n_embedding_dims)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
