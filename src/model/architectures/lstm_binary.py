import torch


class LSTMBinaryClassifier(torch.nn.Module):
    def __init__(
        self, n_features: int, n_hidden: int, n_layers: int, n_out: int = 1
    ):
        super(LSTMBinaryClassifier, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        # The LSTM layer
        self.lstm = torch.nn.LSTM(
            n_features, n_hidden, n_layers, batch_first=True
        )
        # The output layer
        self.fc = torch.nn.Linear(n_hidden, n_out)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden).to(x.device)
        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Only take the output from the final timetep
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out.squeeze(dim=1)
