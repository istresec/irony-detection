from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.nn.functional as F
import torch.nn as nn
import torch


class SimpleClassifier(nn.Module):
    def __init__(self, embedding, embed_dim=300, features_dim=6, hidden_dim=300, num_labels=2):
        super(SimpleClassifier, self).__init__()
        self.features = True if features_dim > 0 else False
        self.embedding = embedding
        self.fc_1 = nn.Linear(in_features=embed_dim + features_dim, out_features=hidden_dim)
        self.fc_2 = nn.Linear(in_features=hidden_dim, out_features=num_labels)

    def forward(self, x, lengths, features=None):
        e = self.embedding(x)
        e = torch.mean(e, dim=1)

        e_and_f = torch.cat((e, features), dim=1) if self.features else e

        s_1 = self.fc_1(e_and_f)
        h_1 = torch.relu(s_1)

        logits = self.fc_2(h_1)

        return logits
