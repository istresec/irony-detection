from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.nn.functional as F
import torch.nn as nn
import torch


class RNNClassifier(nn.Module):
    def __init__(self, embedding, embed_dim=300, hidden_dim=300, num_labels=2):
        super(RNNClassifier, self).__init__()
        self.embedding = embedding
        self.encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=0.3
        )
        self.decoder = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, x, lengths):
        e = self.embedding(x)
        h_pack = pack_padded_sequence(e,
                                      lengths,
                                      enforce_sorted=False,
                                      batch_first=True)

        _, h = self.encoder(h_pack) # [2L x B x H]
        # Concat last state of left and right directions

        h = torch.cat([h[-1], h[-2]], dim=-1) # [B x 2H]

        return self.decoder(h)
