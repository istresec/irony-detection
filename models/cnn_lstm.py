from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.nn as nn
import torch


class CnnRnnClassifier(nn.Module):
    def __init__(self, embedding, embedding_dim=50, conv1_filters=64, conv2_filters=128, dropout_rate=0.2,
                 lstm_hidden_size=20, fc_neurons=128, num_labels=2, conv1_kernel=5, conv1_padding=2,
                 conv2_kernel=3, conv2_padding=1):

        super().__init__()

        self.embedding = embedding

        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=conv1_filters,
                               kernel_size=conv1_kernel, padding=conv1_padding)
        self.conv2 = nn.Conv1d(in_channels=conv1_filters, out_channels=conv2_filters,
                               kernel_size=conv2_kernel, padding=conv2_padding)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.lstm = nn.LSTM(input_size=conv2_filters, hidden_size=lstm_hidden_size, num_layers=2,
                            bidirectional=True, batch_first=True)

        self.fc1 = nn.Linear(in_features=2*lstm_hidden_size, out_features=fc_neurons)
        self.fc2 = nn.Linear(in_features=fc_neurons, out_features=num_labels)

    def forward(self, x, lengths):
        e = self.embedding(x)
        e = e.transpose(1, 2)
        e = self.conv1(e)
        e = torch.relu(e)
        e = self.conv2(e)
        e = torch.relu(e)
        e = self.dropout(e)

        h_pack = pack_padded_sequence(e,
                                      lengths,
                                      enforce_sorted=False,
                                      batch_first=True)
        e = e.transpose(1, 2)
        _, (h, c) = self.lstm(e) # [2L x B x H]
        # Concat last state of left and right directions

        h = torch.cat([h[-1], h[-2]], dim=-1)  # [B x 2H]

        h = self.fc1(h)
        h = torch.relu(h)
        h = self.fc2(h)
        return h
