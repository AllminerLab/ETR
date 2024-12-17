import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim=256, num_layers=2, dropout=0.6):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, dim))
        self.convs.append(GATConv(dim, out_channels))

    def forward(self, x, edge_index):
        # x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers - 1):
            x = F.relu(self.convs[i](x, edge_index))
            # x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)

        return x
