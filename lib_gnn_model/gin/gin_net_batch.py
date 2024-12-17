import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GINConv


class GINNet(torch.nn.Module):

    def __init__(self, input_dim, out_dim, class_num):
        super(GINNet, self).__init__()
    
        self.num_layers = 2
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(Linear(input_dim, out_dim, bias=False)))
        self.convs.append(GINConv(Linear(out_dim, class_num, bias=False)))
    
    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = F.relu(self.convs[i](x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
    
        x = self.convs[-1](x, edge_index)
    
        return x
