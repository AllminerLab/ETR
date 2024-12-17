import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

torch.cuda.empty_cache()
from torch_geometric.nn import CorrectAndSmooth
import numpy as np
from exp.exp import Exp
from lib_gnn_model.node_classifier import NodeClassifier
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import copy
from lib_utils.utils import get_dataset_train, get_influence_nodes, get_subgraph, get_dataset_unlearn
from typing import Dict


class ExpETR(Exp):
    def __init__(self, args):
        super(ExpETR, self).__init__(args)

        self.logger = logging.getLogger('ExpETR')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.load_data()
        self.num_feats = self.data.num_features
        self.train_test_split()
        self.unlearning_request()

        self.target_model_name = self.args['target_model']

        # self.get_edge_indeces()
        self.determine_target_model()

        self.num_layers = 2

        run_training_time, _ = self._train_model()

        # unlearning with ETR
        unlearning_time, f1_score_unlearning = self.etr()

        print("Unlearning: test_acc: {:4f}, time: {:4f}".format(f1_score_unlearning, unlearning_time))

        # unlearning with Retrain
        retraining_time, f1_score_retraining = self.retrain()
        diff = self.get_diff()

        with open('./result/ETR.txt', 'a') as f:
            f.write('Dataset: %s, Seed: %d, GNN: %s, Task: %s, Unlearning f1: %.6f, Unlearning time: %.6f, Diff: %.6f, Unlearning f1: %.6f, Unlearning time: %.6f\n'%(self.args['dataset'], self.args['seed'], self.args['target_model'], self.args['unlearn_task'], f1_score_unlearning, unlearning_time, diff, f1_score_retraining, retraining_time))

        
    def retrain(self):
        retrain_start = time.time()
        criterion = nn.CrossEntropyLoss(label_smoothing=self.args['label_smoothing']).to(self.device)
        optimizer = torch.optim.Adam(self.retrain_model.parameters(), lr=self.args['lr'], weight_decay=self.args['wd'])
        for epoch in range(1, self.args['epochs'] + 1):
            self.retrain_model.train()

            train_true_num = 0
            loss = 0
            num = 0
            for batch in self.dataset_train_unlearned_loader:
                optimizer.zero_grad()
                output = self.retrain_model(batch.x.to(self.device), batch.edge_index.to(self.device))
                preds = torch.argmax(output[:batch.batch_size], dim=1)
                train_true_num += torch.sum(preds == batch.y[:batch.batch_size].to(self.device))
                loss = criterion(output[:batch.batch_size], batch.y[:batch.batch_size].to(self.device))
                loss.backward()
                optimizer.step()
            train_acc = train_true_num / self.dataset_train_unlearned.train_mask.sum()
        retrain_end = time.time()

        self.retrain_model.eval()
        test_true_num = 0
        with torch.no_grad():
            for batch in self.dataset_unlearned_test_loader:
                output = self.retrain_model(batch.x.to(self.device), batch.edge_index.to(self.device))
                preds = torch.argmax(output[:batch.batch_size], dim=1)
                test_true_num += torch.sum(preds == batch.y[:batch.batch_size].to(self.device)).float()
            test_acc = test_true_num / self.dataset_unlearned.test_mask.sum()

        print("Retraining: train_acc: {:4f}, test_acc: {:4f}, time: {:4f}".format(train_acc, test_acc, retrain_end - retrain_start))

        retraining_test_acc = test_acc

        return retrain_end - retrain_start, retraining_test_acc

    
    def get_diff(self):
        diff = 0
        para_num = 0
        for (u_n, u_p), (r_n, r_p) in zip(self.target_model.model.named_parameters(), self.retrain_model.named_parameters()):
            diff += (u_p - r_p).pow(2).sum()
            para_num += u_p.flatten().shape[-1]
        diff /= para_num

        return diff


    def load_data(self):
        self.data = self.data_store.load_raw_data()


    def train_test_split(self):

        if self.args['dataset'] in ['ogbn-arxiv', 'ogbn-products']:
            self.train_indices, self.test_indices = self.data.train_indices.numpy(), self.data.test_indices.numpy()
        else:
            self.train_indices, self.test_indices = train_test_split(np.arange(self.data.num_nodes),
                                                                        test_size=self.args['test_ratio'],
                                                                        random_state=100)

        self.data_store.save_train_test_split(self.train_indices, self.test_indices)

        self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))
        self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))
        self.data.train_indices = self.train_indices
        self.data.test_indices = self.test_indices


    def unlearning_request(self):

        self.dataset_train = get_dataset_train(self.data)
        if self.args['unlearn_task'] == 'edge':
            self.unique_indices = np.where(self.data.edge_index[0] < self.data.edge_index[1])[0]
            self.remove_indices = np.random.choice(self.unique_indices, int(self.unique_indices.shape[0] * self.args['unlearn_ratio']), replace=False)
            self.remove_edges = self.data.edge_index[:, self.remove_indices]
            self.unique_nodes = np.unique(self.remove_edges)
            self.influenced_nodes = get_influence_nodes(self.args, self.unique_nodes, self.dataset_train.edge_index, 1)
            self.dataset_unlearned = get_dataset_unlearn(self.args, self.data, self.unique_nodes, self.remove_indices)
            self.dataset_train_unlearned = get_dataset_train(self.dataset_unlearned)
        else:
            if self.args['dataset'] in ['ogbn-mag', 'ogbn-products']:
                self.unlearning_id = np.random.choice(self.data.train_indices, 100, replace=False)
            else:
                self.unlearning_id = np.random.choice(self.data.train_indices, int(len(self.data.train_indices) * self.args['unlearn_ratio']), replace=False)
            self.influenced_nodes = get_influence_nodes(self.args, self.unlearning_id, self.dataset_train.edge_index)
            self.dataset_train_unlearned = get_dataset_unlearn(self.args, self.dataset_train, self.unlearning_id)
            self.dataset_unlearned = get_dataset_unlearn(self.args, self.data, self.unlearning_id)
            if self.args['unlearn_task'] == 'node':
                self.dataset_train_unlearned.train_mask[self.unlearning_id] = False
                self.unlearn_subgraph = get_subgraph(self.unlearning_id, self.dataset_train)
        
        self.influenced_subgraph = get_subgraph(self.influenced_nodes, self.dataset_train)
        self.influenced_subgraph_unlearned = get_subgraph(self.influenced_nodes, self.dataset_train_unlearned)
        
        self.dataset_train_loader = NeighborLoader(self.dataset_train, num_neighbors=[5] * self.args['hops'], input_nodes=self.dataset_train.train_mask, batch_size=256, shuffle=True)
        self.data_test_loader = NeighborLoader(self.data, num_neighbors=[5] * self.args['hops'], input_nodes=self.data.test_mask, batch_size=256, shuffle=False)
        self.dataset_train_unlearned_loader = NeighborLoader(self.dataset_train_unlearned.contiguous(), num_neighbors=[5] * self.args['hops'], input_nodes=self.dataset_train_unlearned.train_mask, batch_size=256, shuffle=True)
        if self.args['dataset'] in ['ogbn-mag', 'ogbn-products']:
            self.dataset_unlearned_test_loader = NeighborLoader(self.dataset_unlearned.contiguous(), num_neighbors=[5] * self.args['hops'], input_nodes=self.dataset_unlearned.test_mask, batch_size=256, shuffle=False)
        else:
            self.dataset_unlearned_test_loader = NeighborLoader(self.dataset_unlearned.contiguous(), num_neighbors=[-1] * self.args['hops'], input_nodes=self.dataset_unlearned.test_mask, batch_size=256, shuffle=False)
        if self.args['dataset'] in ['ogbn-mag', 'ogbn-products']:
            self.unlearn_subgraph_loader = NeighborLoader(self.unlearn_subgraph, num_neighbors=[5] * self.args['hops'], input_nodes=self.unlearn_subgraph.mapping, batch_size=len(self.unlearn_subgraph.mapping), shuffle=True)
            self.influenced_subgraph_loader = NeighborLoader(self.influenced_subgraph, num_neighbors=[5] * self.args['hops'], input_nodes=self.influenced_subgraph.mapping, batch_size=len(self.influenced_subgraph.mapping), shuffle=True)
            self.influenced_subgraph_unlearned_loader = NeighborLoader(self.influenced_subgraph_unlearned, num_neighbors=[5] * self.args['hops'], input_nodes=self.influenced_subgraph_unlearned.mapping, batch_size=len(self.influenced_subgraph_unlearned.mapping), shuffle=True)
        

    def determine_target_model(self):
        # self.logger.info('target model: %s' % (self.args['target_model'],))
        num_classes = self.data.num_classes

        self.target_model = NodeClassifier(self.num_feats, num_classes, self.args)


    def _train_model(self):

        start_time = time.time()
        self.target_model.data = self.data
        res = self.train_model()
        train_time = time.time() - start_time

        # self.data_store.save_target_model(run, self.target_model)
        self.logger.info(f"Model training time: {train_time:.4f}")

        return train_time, res


    def train_model(self):
        # self.logger.info("training model")
        self.target_model.model.train()
        self.retrain_model = copy.deepcopy(self.target_model.model)
        self.target_model.model, self.data = self.target_model.model.to(self.device), self.data.to(self.device)
        self.data.y = self.data.y.squeeze().to(self.device)

        optimizer = torch.optim.Adam(self.target_model.model.parameters(), lr=self.args['lr'], weight_decay=self.args['wd'])

        criterion = nn.CrossEntropyLoss(label_smoothing=self.args['label_smoothing']).to(self.device)
        for epoch in range(1, self.args['epochs'] + 1):
            self.target_model.model.train()
            loss_all = 0
            train_true_num = 0
            num = 0
            for batch in self.dataset_train_loader:
                optimizer.zero_grad()
                output = self.target_model.model(batch.x.to(self.device), batch.edge_index.to(self.device))
                preds = torch.argmax(output[:batch.batch_size], dim=1)
                train_true_num += torch.sum(preds.cpu() == batch.y[:batch.batch_size])
                loss = criterion(output[:batch.batch_size], batch.y[:batch.batch_size].to(self.device))
                loss.backward()
                optimizer.step()
            train_acc = train_true_num / self.dataset_train.train_mask.sum()

        self.target_model.model.eval()
        test_true_num = 0
        with torch.no_grad():
            for batch in self.data_test_loader:
                batch = batch
                output = self.target_model.model(batch.x.to(self.device), batch.edge_index.to(self.device))
                preds = torch.argmax(output[:batch.batch_size], dim=1)
                test_true_num += torch.sum(preds == batch.y[:batch.batch_size])
            test_acc = test_true_num / self.data.test_mask.sum()

        print("Training: train_acc: {:4f}, test_acc: {:4f}".format(train_acc, test_acc))
    
        training_test_acc = test_acc
        return training_test_acc
    

    def etr(self):
        criterion = nn.CrossEntropyLoss(label_smoothing=self.args['label_smoothing']).to(self.device)
        self.model_unlearn = copy.deepcopy(self.target_model.model)
        optimizer_u = torch.optim.Adam(self.model_unlearn.parameters(), lr=self.args['lr'], weight_decay=self.args['wd'])

        if self.args['dataset'] in ['ogbn-mag', 'ogbn-products']:
            grad_ori_all, parameter_importance_all = self.get_parameter_grad_and_information_minibatch(self.model_unlearn, self.dataset_train_loader, optimizer_u, criterion)
        else:
            grad_ori_all, parameter_importance_all = self.get_parameter_grad_and_information(self.model_unlearn, self.dataset_train, optimizer_u, criterion)

        start = time.time()

        if self.args['unlearn_task'] == 'node':
            if self.args['dataset'] in ['ogbn-mag', 'ogbn-products']:    
                grad_ori_f, parameter_importance_unlearn = self.get_parameter_grad_and_information_minibatch(self.model_unlearn, self.unlearn_subgraph_loader, optimizer_u, criterion)        
            else:
                grad_ori_f, parameter_importance_unlearn = self.get_parameter_grad_and_information(self.model_unlearn, self.unlearn_subgraph, optimizer_u, criterion, batch=True)  
        if self.args['dataset'] in ['ogbn-mag', 'ogbn-products']:
            grad_ori_k, parameter_importance_influnence = self.get_parameter_grad_and_information_minibatch(self.model_unlearn, self.influenced_subgraph_loader, optimizer_u, criterion)
        else:
            grad_ori_k, parameter_importance_influnence = self.get_parameter_grad_and_information(self.model_unlearn, self.influenced_subgraph, optimizer_u, criterion, batch=True)

        if self.args['unlearn_task'] == 'node':
            self.modify_weight(self.model_unlearn, parameter_importance_all, parameter_importance_unlearn, 
                               parameter_importance_influnence, self.args['erase_ratio'])
        else:
            self.modify_weight_ef(self.model_unlearn, parameter_importance_all, parameter_importance_influnence, self.args['erase_ratio'])

        if self.args['dataset'] in ['ogbn-mag', 'ogbn-products']:
            grad_u, _ = self.get_parameter_grad_and_information_minibatch(self.model_unlearn, self.influenced_subgraph_unlearned_loader, optimizer_u, criterion)
        else:
            grad_u, _ = self.get_parameter_grad_and_information(self.model_unlearn, self.influenced_subgraph_unlearned, optimizer_u, criterion, batch=True)

        if self.args['unlearn_task'] == 'node':     
            self.modify_weight2(self.model_unlearn, grad_ori_all, grad_ori_f, grad_ori_k, grad_u, self.args['l'], len(self.unlearning_id), len(self.influenced_nodes), self.dataset_train.train_mask.sum())
        else:
            self.modify_weight2_ef(self.model_unlearn, grad_ori_all, grad_ori_k, grad_u, self.args['l'], len(self.influenced_nodes), self.dataset_train.train_mask.sum())

        end = time.time()

        with torch.no_grad():
            test_true_num = 0
            for batch in self.dataset_unlearned_test_loader:
                output = self.model_unlearn(batch.x.to(self.device), batch.edge_index.to(self.device))
                preds = torch.argmax(output[:batch.batch_size], dim=1)
                test_true_num += torch.sum(preds == batch.y[:batch.batch_size].to(self.device))
            test_acc = test_true_num / self.dataset_unlearned.test_mask.sum()
        unlearn_test_acc = test_acc
        
        return end - start, unlearn_test_acc
    

    def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]:
        return dict(
            [
                (k, torch.zeros_like(p, device=p.device))
                for k, p in model.named_parameters()
            ]
        )


    def get_parameter_information_minibatch(self, model, data_loader, optimizer, criterion):
        model.train()
        parameter_importance = self.zerolike_params_dict(model)
        loss = 0
        num = 0

        for batch in data_loader:
            optimizer.zero_grad()
            output = model(batch.x, batch.edge_index)
            loss = criterion(output[:batch.batch_size], batch.y[:batch.batch_size])
            loss.backward()

            for (k1, p), (k2, imp) in zip(model.named_parameters(), parameter_importance.items()):
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        for _, imp in parameter_importance.items():
            imp.data /= float(len(data_loader))

        return parameter_importance


    def get_parameter_information(self, model, data, optimizer, criterion, batch=False):
        model.train()
        parameter_importance = self.zerolike_params_dict(model)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        if batch:
            loss = criterion(output[:data.batch_size], data.y[:data.batch_size])
        else:
            loss = criterion(output[data.train_mask], data.y[data.train_mask])
        loss.backward()

        for (k1, p), (k2, imp) in zip(model.named_parameters(), parameter_importance.items()):
            if p.grad is not None:
                imp.data += p.grad.data.clone().pow(2)

        return parameter_importance


    def get_parameter_grad_and_information_minibatch(self, model, data_loader, optimizer, criterion):
        model.train()
        parameter_grad = self.zerolike_params_dict(model)
        parameter_importance = self.zerolike_params_dict(model)
        loss = 0
        num = 0
        for batch in data_loader:
            optimizer.zero_grad()
            output = model(batch.x, batch.edge_index)
            loss = criterion(output[:batch.batch_size], batch.y[:batch.batch_size])
            loss.backward()

            for (k1, p), (k2, imp), (k3, imp2) in zip(model.named_parameters(), parameter_grad.items(), parameter_importance.items()):
                if p.grad is not None:
                    imp.data += p.grad.data.clone()
                    imp2.data += p.grad.data.clone().pow(2)

        for _, imp in parameter_grad.items():
            imp.data /= float(len(data_loader))

        for _, imp in parameter_importance.items():
            imp.data /= float(len(data_loader))

        return parameter_grad, parameter_importance


    def get_parameter_grad_and_information(self, model, data, optimizer, criterion, batch=False):
        model.train()
        parameter_grad = self.zerolike_params_dict(model)
        parameter_importance = self.zerolike_params_dict(model)
        optimizer.zero_grad()
        output = model(data.x.to(self.device), data.edge_index.to(self.device))
        if batch:
            loss = criterion(output[:data.batch_size], data.y[:data.batch_size].to(self.device))
        else:
            loss = criterion(output[data.train_mask], data.y[data.train_mask].to(self.device))
        loss.backward()

        for (k1, p), (k2, imp), (k3, imp2) in zip(model.named_parameters(), parameter_grad.items(), parameter_importance.items()):
            if p.grad is not None:
                imp.data += p.grad.data.clone()
                imp2.data += p.grad.data.clone().pow(2)

        return parameter_grad, parameter_importance


    def modify_weight_ef(self, model, parameter_importance_all, parameter_importance_before, k):
        with torch.no_grad():
            for (n, p), (bimp_n, aimp), (aimp_n, bimp) in zip(model.named_parameters(), parameter_importance_all.items(),
                                                              parameter_importance_before.items()):
                b = torch.min(torch.nan_to_num(aimp / bimp, nan=1), torch.ones_like(bimp))
                locations = (b <= torch.quantile(b, k))
                update = b[locations] / torch.quantile(b, k)
                p[locations] = p[locations].mul(update)


    def modify_weight(self, model, parameter_importance_all, parameter_importance_unlearn, 
                      parameter_importance_influnence, k):
        with torch.no_grad():
            for (n, p), (oimp_n, oimp), (uimp_n, uimp), (nimp_n, nimp) in zip(model.named_parameters(),
                                                                              parameter_importance_all.items(),
                                                                              parameter_importance_unlearn.items(),
                                                                              parameter_importance_influnence.items()):
                b1 = torch.min(torch.nan_to_num(oimp / uimp, nan=1), torch.ones_like(oimp))
                b2 = torch.min(torch.nan_to_num(oimp.pow(2) / (uimp * nimp), nan=1), torch.ones_like(oimp))
                locations1 = (b1 <= torch.quantile(b1, k))
                locations2 = (b2 <= torch.quantile(b2, k))
                locations2[locations1] = False

                update1 = b1[locations1] / torch.quantile(b1, k)
                update2 = b2[locations2] / torch.quantile(b2, k)
                p[locations1] = p[locations1].mul(update1)
                p[locations2] = p[locations2].mul(update2)


    def modify_weight2(self, model, grad_ori_all, grad_ori_f, grad_ori_k, grad_u, l, u_num, i_num, num):
        with torch.no_grad():
            m = num - u_num
            for (n, p), (a_n, ap), (f_n, fp), (k_n, kp), (u_n, up) in zip(model.named_parameters(), 
                                                                          grad_ori_all.items(), 
                                                                          grad_ori_f.items(), 
                                                                          grad_ori_k.items(), 
                                                                          grad_u.items()):
                p.sub_(l * (ap * num / m - fp * u_num / m - kp * i_num / m + up * i_num / m))


    def modify_weight2_ef(self, model, grad_ori_all, grad_ori_k, grad_u, l, i_num, num):
        with torch.no_grad():
            for (n, p), (a_n, ap), (k_n, kp), (u_n, up) in zip(model.named_parameters(), grad_ori_all.items(), 
                                                               grad_ori_k.items(), grad_u.items()):
                p.sub_(l * (ap - kp * i_num / num + up * i_num / num))
