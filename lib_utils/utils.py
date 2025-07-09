import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import yaml
from yaml import SafeLoader


def filter_edge_index(edge_index, node_indices, reindex=True):
    assert np.all(np.diff(node_indices) >= 0), 'node_indices must be sorted'
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu()

    node_index = np.isin(edge_index, node_indices)
    col_index = np.nonzero(np.logical_and(node_index[0], node_index[1]))[0]
    edge_index = edge_index[:, col_index]

    if reindex:
        return np.searchsorted(node_indices, edge_index)
    else:
        return edge_index
    

def get_dataset_train(data):

    train_indices = np.nonzero(data.train_mask.cpu().numpy())[0]
    edge_index = filter_edge_index(data.edge_index, train_indices, reindex=False)
    if edge_index.shape[1] == 0:
        edge_index = torch.tensor([[1, 2], [2, 1]])

    dataset_train = Data(x=data.x, edge_index=edge_index, y=data.y, train_mask=data.train_mask,
                         test_mask=data.test_mask, train_indices=data.train_indices, 
                         test_indices=data.test_indices)

    return dataset_train


def get_influence_nodes(args, unlearn_nodes, edge_index, hops=2):
    influenced_nodes = unlearn_nodes
    for _ in range(hops):
        target_nodes_location = np.isin(edge_index[0], influenced_nodes)
        neighbor_nodes = edge_index[1, target_nodes_location]
        influenced_nodes = np.append(influenced_nodes, neighbor_nodes)
        influenced_nodes = np.unique(influenced_nodes)
    if args['unlearn_task'] == 'node':
        neighbor_nodes = np.setdiff1d(influenced_nodes, unlearn_nodes)
    else:
        neighbor_nodes = influenced_nodes
    return neighbor_nodes


def get_subgraph(node_id, data, hops=2):
    node_id = node_id[np.isin(node_id, data.edge_index.cpu().numpy())]
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(torch.tensor(node_id), hops, data.edge_index, relabel_nodes=True)
    subgraph = Data(x=data.x[subset], edge_index=edge_index, y=data.y[subset], batch_size=len(node_id), mapping=mapping)
    return subgraph


def get_dataset_unlearn(args, data, unlearning_id, delete_edge_index=None):
    if args['unlearn_task'] == 'feature':
        x = data.x
        x[unlearning_id] = 0
        dataset_unlearn = Data(x=x, edge_index=data.edge_index, y=data.y, train_mask=data.train_mask, test_mask=data.test_mask, train_indices=data.train_indices, test_indices=data.test_indices)
    else:
        if args['unlearn_task'] == 'node':
            edge_index_unlearn = update_edge_index_unlearn(args, data.edge_index.cpu().numpy(), unlearning_id)
        elif args['unlearn_task'] == 'edge':
            edge_index_unlearn = update_edge_index_unlearn(args, data.edge_index.cpu().numpy(), unlearning_id, delete_edge_index)
        dataset_unlearn = Data(x=data.x, edge_index=edge_index_unlearn, y=data.y, train_mask=data.train_mask, test_mask=data.test_mask, train_indices=data.train_indices, test_indices=data.test_indices)

    return dataset_unlearn


def update_edge_index_unlearn(args, edge_index, delete_nodes, delete_edge_index=None):
    unique_indices = np.where(edge_index[0] < edge_index[1])[0]
    unique_indices_not = np.where(edge_index[0] > edge_index[1])[0]

    if args['unlearn_task'] == 'edge':
        remain_indices = np.setdiff1d(unique_indices, delete_edge_index)
    else:
        unique_edge_index = edge_index[:, unique_indices]
        delete_edge_indices = np.logical_or(np.isin(unique_edge_index[0], delete_nodes),
                                            np.isin(unique_edge_index[1], delete_nodes))
        remain_indices = np.logical_not(delete_edge_indices)
        remain_indices = np.where(remain_indices == True)

    remain_encode = edge_index[0, remain_indices] * edge_index.shape[1] * 2 + edge_index[1, remain_indices]
    unique_encode_not = edge_index[1, unique_indices_not] * edge_index.shape[1] * 2 + edge_index[0, unique_indices_not]
    sort_indices = np.argsort(unique_encode_not)
    remain_indices_not = unique_indices_not[sort_indices[np.searchsorted(unique_encode_not, remain_encode, sorter=sort_indices)]]
    remain_indices = np.union1d(remain_indices, remain_indices_not)

    return torch.from_numpy(edge_index[:, remain_indices])


def get_parameter(args):
    config = yaml.load(open(args['para_config']), Loader=SafeLoader)
    args['lr'] = float(config[args['target_model']][args['dataset']]['lr'])
    args['wd'] = float(config[args['target_model']][args['dataset']]['wd'])
    args['erase_ratio'] = float(config[args['target_model']][args['dataset']]['erase_ratio'])
    args['l'] = float(config[args['target_model']][args['dataset']]['l'])
    return args
