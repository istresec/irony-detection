from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.nn as nn
import torch


class CnnRnnClassifier(nn.Module):
    def __init__(self, embedding, embedding_dim=50, conv1_filters=64, conv2_filters=128, dropout_rate=0.2,
                 lstm_hidden_size=20, fc_neurons=128, num_labels=2, conv1_kernel=3, conv1_padding=0,
                 conv2_kernel=3, conv2_padding=1, max_length=37, features_dim=6):
        super().__init__()

        self.embedding = embedding
        self.features = True if features_dim > 0 else False
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv1_filters,
                               kernel_size=(conv1_kernel, embedding_dim), padding=(conv1_padding, 0))
        self.conv2 = nn.Conv1d(in_channels=conv1_filters, out_channels=conv2_filters,
                               kernel_size=conv2_kernel, padding=conv2_padding)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.lstm = nn.LSTM(input_size=conv2_filters, hidden_size=lstm_hidden_size, num_layers=2,
                            bidirectional=False, batch_first=True)

        self.fc1 = nn.Linear(in_features=lstm_hidden_size * (max_length - 2) + features_dim, out_features=num_labels)

    def forward(self, x, lengths, features=None):
        e = self.embedding(x)

        # Reshape for 2D convolution
        e = torch.reshape(e, shape=(e.shape[0], 1, e.shape[1], e.shape[2]))  # [B x 1 x L x D]

        e = self.conv1(e)
        e = torch.sigmoid(e)

        # Reshape for 1D convolution
        e = torch.reshape(e, shape=(e.shape[0], e.shape[1], e.shape[2]))

        e = self.conv2(e)
        e = torch.sigmoid(e)

        e = self.dropout(e)

        e = e.transpose(1, 2)
        o, (h, c) = self.lstm(e)  # [2L x B x H]

        h = torch.flatten(o, start_dim=1)

        # Concat features if they are used
        h_and_f = torch.cat((h, features), dim=1) if self.features else h

        h = self.fc1(h_and_f)
        return h
