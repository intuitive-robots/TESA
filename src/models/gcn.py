import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    # EDGE ATTRIBUTES ARE NOT USED
    def __init__(self, num_features, num_hidden, num_output, learning_rate=0.01):
        super().__init__()
        self.conv1 = GCNConv(num_features, num_hidden)
        self.conv2 = GCNConv(num_hidden, num_hidden)
        self.conv3 = GCNConv(num_hidden, num_hidden)
        self.conv4 = GCNConv(num_hidden, num_hidden)
        # linear layer
        self.fc = torch.nn.Linear(num_hidden, num_output)
        self.learning_rate = learning_rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch

        x = self.conv1(x, edge_index)  # possible improvement: hier fehlt edge_weight.
        # x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.fc(x)
        x = global_mean_pool(x, batch)

        return x  # F.log_softmax(x, dim=1) # check shape == 1,512 (check in non-toy!)


class MixedN(torch.nn.Module):
    def __init__(self, num_features, num_hidden, num_output, learning_rate=0.01):
        super().__init__()
        self.conv1 = GCNConv(num_hidden, num_hidden)
        self.conv2 = GCNConv(num_hidden, num_hidden)
        # linear layer
        self.fc = torch.nn.Linear(num_hidden, num_output)
        self.fc1 = torch.nn.Linear(num_features, num_hidden)
        self.fc2 = torch.nn.Linear(num_hidden, num_hidden)
        self.learning_rate = learning_rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch
        # fc1 relu conv1 relu d fc2 relu conv2 relu dropout fc mean
        x = self.fc1(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc(x)  # possible alternative: vor oder nach global_mean_pool?
        x = global_mean_pool(x, batch)

        return x  # F.log_softmax(x, dim=1) # check shape == 1,512 (check in non-toy!)


class trivial_base(torch.nn.Module):
    r"""
    Only one linear layer, average pooling"""

    def __init__(self, num_features, num_output):
        super().__init__()
        # linear layer
        self.fc = torch.nn.Linear(num_features, num_output)

    def forward(self, data):
        return global_mean_pool(self.fc(data.x), data.batch)
