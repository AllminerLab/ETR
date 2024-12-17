import logging

import torch
from sklearn.metrics import f1_score

torch.cuda.empty_cache()
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
import torch.nn as nn
from lib_gnn_model.gat.gat_net_batch import GATNet
from lib_gnn_model.gin.gin_net_batch import GINNet
from lib_gnn_model.gcn.gcn_net_batch import GCNNet
from lib_gnn_model.graphsage.graphsage_net_batch import SAGENet
from lib_gnn_model.gnn_base import GNNBase
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader, GraphSAINTRandomWalkSampler
import copy

class NodeClassifier(GNNBase):
    def __init__(self, num_feats, num_classes, args, data=None):
        super(NodeClassifier, self).__init__()

        self.args = args
        self.logger = logging.getLogger('node_classifier')
        self.target_model = args['target_model']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        self.model = self.determine_model(num_feats, num_classes).to(self.device)
        self.data = data

    def determine_model(self, num_feats, num_classes):
        # self.logger.info('target model: %s' % (self.args['target_model'],))
        if self.target_model == 'SAGE':
            return SAGENet(num_feats, num_classes, self.args['dim'])
        elif self.target_model == 'GAT':
            return GATNet(num_feats, num_classes, self.args['dim'])
        elif self.target_model == 'GCN':
            return GCNNet(num_feats, num_classes, self.args['dim'])
        elif self.target_model == 'GIN':
            return GINNet(num_feats, num_classes)
        else:
            raise Exception('unsupported target model')
    