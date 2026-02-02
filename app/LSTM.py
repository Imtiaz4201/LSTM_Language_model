import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        # LSTM with dropout between layers (only works if num_layers > 1)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Dropout on LSTM outputs
        self.dropout = nn.Dropout(dropout)

        # Final classifier
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x: (B, T)
        x = self.embedding(x)  # (B, T, E)
        out, _ = self.lstm(x)  # (B, T, H)
        out = self.dropout(out)  # regularization
        out = self.fc(out)  # (B, T, V)
        return out
