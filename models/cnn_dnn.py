import torch.nn as nn
import torch


class CnnDnnClassifier(nn.Module):
    def __init__(self, embedding, embedding_dim=50, conv1_filters=64, conv1_kernel=5, conv1_padding=2,
                 conv2_filters=128, conv2_kernel=3, conv2_padding=1, dropout_rate=0.2,
                 fc_neurons=128, num_labels=2, max_length=37, features_dim=6):
        super().__init__()
        self.features = True if features_dim > 0 else False
        self.embedding = embedding

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv1_filters,
                               kernel_size=(conv1_kernel, embedding_dim), padding=(conv1_padding, 0))
        self.conv2 = nn.Conv1d(in_channels=conv1_filters, out_channels=conv2_filters,
                               kernel_size=conv2_kernel, padding=conv2_padding)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(in_features=conv2_filters * (max_length - 2) + features_dim, out_features=fc_neurons)
        self.fc2 = nn.Linear(in_features=fc_neurons, out_features=num_labels)

    def forward(self, x, lengths, features=None):
        e = self.embedding(x)

        # Reshape for 2D convolution
        e = torch.reshape(e, shape=(e.shape[0], 1, e.shape[1], e.shape[2]))  # [B x 1 x L x D]

        e = self.conv1(e)
        e = torch.relu(e)

        # Reshape for 1D convolution
        e = torch.reshape(e, shape=(e.shape[0], e.shape[1], e.shape[2]))

        e = self.conv2(e)
        e = torch.relu(e)

        e = self.dropout(e)

        e = e.flatten(start_dim=1)

        e_and_f = torch.cat((e, features), dim=1) if self.features else e

        e = self.fc1(e_and_f)
        e = torch.relu(e)
        e = self.dropout(e)
        e = self.fc2(e)
        return e
