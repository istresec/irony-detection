import torch.nn as nn
import torch


class CnnDnnClassifier(nn.Module):
    def __init__(self, embedding, embedding_dim=50, conv1_filters=64, conv1_kernel=5, conv1_padding=2,
                 conv2_filters=128, conv2_kernel=3, conv2_padding=1, dropout_rate=0.2,
                 fc_neurons=128, num_labels=2, max_length=37, features_dim=6):

        super().__init__()
        self.features = True if features_dim > 0 else False
        self.embedding = embedding

        self.conv1 = nn.Conv1d(in_channels=embedding_dim+features_dim, out_channels=conv1_filters,
                               kernel_size=conv1_kernel, padding=conv1_padding)
        self.conv2 = nn.Conv1d(in_channels=conv1_filters, out_channels=conv2_filters,
                               kernel_size=conv2_kernel, padding=conv2_padding)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(in_features=max_length*conv2_filters, out_features=fc_neurons)
        self.fc2 = nn.Linear(in_features=fc_neurons, out_features=num_labels)

    def forward(self, x, lengths, features=None):
        e = self.embedding(x)

        # Get embedding average using lengths
        e = torch.sum(e, dim=1)
        e = torch.div(e, torch.tensor(lengths).to(device=e.device)[:, None])

        e_and_f = torch.cat((e, features), dim=1) if self.features else e

        e = self.conv1(e_and_f)
        e = torch.relu(e)
        e = self.dropout(e)
        e = self.conv2(e)
        e = torch.relu(e)
        e = self.dropout(e)
        e = e.flatten(start_dim=1)
        e = self.fc1(e)
        e = torch.relu(e)
        e = self.dropout(e)
        e = self.fc2(e)
        return e
