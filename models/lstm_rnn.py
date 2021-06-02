from torch.nn.utils.rnn import pack_padded_sequence

import torch.nn as nn
import torch


class RNNClassifier(nn.Module):
    def __init__(self, embedding, embed_dim=300, hidden_dim=300, num_labels=2, num_layers=2, dropout=0.2, max_len=37,
                 features_dim=6):
        super(RNNClassifier, self).__init__()
        self.features = True if features_dim > 0 else False
        self.embedding = embedding
        self.n_layers = num_layers
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim+features_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_labels)
        )

    def init_hidden(self, batch_size, device):
        self.hidden = (torch.zeros(self.n_layers*2, batch_size, self.hidden_dim).to(device),
                       torch.zeros(self.n_layers*2, batch_size, self.hidden_dim).to(device))

    def forward(self, x, lengths, features=None):
        e = self.embedding(x)

        packed_e = pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.encoder(packed_e)
        hidden = hidden[-1, :, :]
        if self.features:
            inputs = torch.cat((hidden, features), dim=1)
        else:
            inputs = hidden
        return self.decoder(inputs)
