import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

ADG_THR = {
    'dist': 0.31,
    'corr0': 0.45,
    'plv0': 0.42,
    'coh0': 0.21,
    'pdc0': 0.47,
    'corr0_overlap': 0.44,
    'plv0_overlap': 0.42,
    'coh0_overlap': 0.21,
    'pdc0_overlap': 0.47,
}


def get_data_paths(dataset, signal_type, conn_method, padding='zeroed'):
    assert signal_type in (
        'raw_preprocessed',
        'extract_feature',
        'extract_feature_de'
    ), "Value of signal_type must be 'raw_preprocessed' or 'extract_feature' or 'extract_feature_de'."
    assert conn_method in (
        'dist',
        'corr0',
        'plv0',
        'coh0',
        'pdc0',
        'corr0_overlap',
        'plv0_overlap',
        'coh0_overlap',
        'pdc0_overlap',
    ), "Value of adj_method must be one of the ('dist', 'corr', 'coh', 'plv')."

    if padding == 'last':
        x_path = f'/app/data/{dataset}/{signal_type}/X_with_padding_last.npy'
        y_path = f'/app/data/{dataset}/{signal_type}/y_last.npy'
    elif padding == 'zeroed':
        x_path = f'/app/data/{dataset}/{signal_type}/X_with_padding_zeroed.npy'
        y_path = f'/app/data/{dataset}/{signal_type}/y_zeroed.npy'
    else:
        # padding = -1e6 чтобы можно было отрезать
        x_path = f'/app/data/{dataset}/{signal_type}/X_with_padding.npy'
        y_path = f'/app/data/{dataset}/{signal_type}/y.npy'

    adj_path = f'/app/data/{dataset}/{conn_method}_adjacencies_temporal/'
    adj_thr = ADG_THR[conn_method]

    return x_path, y_path, adj_path, adj_thr


class GraphDataset(Dataset):
    @staticmethod
    def _get_edges(adjacencies, thr, t):
        edges = []
        for adj in adjacencies:
            adj = adj[t]
            edges_ = []
            for j in range(len(adj)):
                for i in range(j):
                    if adj[i][j] >= thr:
                        edges_.extend(([i, j], [j, i]))

            edges.append(np.array(edges_).T)
        return edges

    @staticmethod
    def _get_edge_weights(adjacencies, thr, t):
        edge_weights = []
        for adj in adjacencies:
            adj = adj[t]
            edge_weights_ = []
            for j in range(len(adj)):
                for i in range(j):
                    if adj[i][j] >= thr:
                        edge_weights_.extend((adj[i][j], adj[i][j]))

            edge_weights.append(np.array(edge_weights_))
        return edge_weights

    @staticmethod
    def _get_targets_and_features(x, y, drop_inx):
        target_range = np.sort(np.unique(y))
        targets = [np.where(i == target_range)[0] for i in y]
        if drop_inx is not None:
            for i in drop_inx:
                x[:, i, :, :] = 0
        features = [x[i].swapaxes(1, 2) for i in range(len(x))]
        return targets, features

    def __init__(self, x_path, y_path, adj_path, adj_thr, shuff, t, drop_inx=None):
        super().__init__()

        self.t_cut = 250

        x = np.load(x_path)
        y = np.load(y_path)
        x = x[:, :, :self.t_cut, :]

        if shuff:
            x, y = shuffle(x, y, random_state=0)

        num_matrix = len(os.listdir(adj_path))
        if num_matrix == 1:
            adjs = [np.load(adj_path+'matrix.npy')[:self.t_cut]]*len(x)
        else:
            adjs = [np.load(adj_path+f'matrix{i}.npy')[:self.t_cut] for i in range(len(x))]

        self.edges = self._get_edges(adjs, adj_thr, t)
        self.edge_weights = self._get_edge_weights(adjs, adj_thr, t)
        self.target, self.feature = self._get_targets_and_features(x, y, drop_inx)

    def len(self):
        return len(self.feature)

    def get(self, idx):
        data = Data(
            x=torch.Tensor(self.feature[idx]),
            edge_index=torch.LongTensor(self.edges[idx]),
            edge_attr=torch.Tensor(self.edge_weights[idx]),
            y=torch.LongTensor(self.target[idx])
        )
        return data


def get_loaders(
        data_type,
        signal_type,
        conn_method,
        padding,
        batch_size,
        train_ratio=0.85,
        shuffle=False,
        t=0,
):
    data_paths = get_data_paths(data_type, signal_type, conn_method, padding)
    dataset = GraphDataset(*data_paths, shuffle, t)

    train_size = int(len(dataset) * train_ratio)
    train_dataset, test_dataset = dataset[:train_size], dataset[train_size:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=30)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=30)
    return train_loader, test_loader


def cross_val_loaders(
    data_type,
    signal_type,
    conn_method,
    padding,
    batch_size,
    shuffle=False,
    k_fold=5
):
    data_paths = get_data_paths(data_type, signal_type, conn_method, padding)
    dataset = GraphDataset(*data_paths, shuffle)
    indices = np.arange(len(dataset))

    loaders = []
    kf = KFold(n_splits=k_fold)
    for train_index, test_index in kf.split(indices):
        train_dataset = dataset[train_index]
        test_dataset = dataset[test_index]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=30)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=30)
        loaders.append((train_loader, test_loader))
    return loaders


def drop_loaders(
    data_type,
    signal_type,
    conn_method,
    padding,
    batch_size,
    train_ratio=0.85,
    shuffle=False,
    drop_max=1,
):
    adj = np.load(f'data/{data_type}/{conn_method}_adjacencies/matrix.npy')
    adj = np.where(adj > ADG_THR[conn_method], 1, 0)
    sort_degree_ind = np.argsort(adj.sum(axis=1))
    loaders = []
    data_paths = get_data_paths(data_type, signal_type, conn_method, padding)
    for i in [1, 2, 4, 8, 16, 32, 38, 42]:  # range(drop_max+1):
        if i == 0:
            drop_inx = None
        else:
            drop_inx = sort_degree_ind[:i]
        dataset = GraphDataset(*data_paths, shuffle, drop_inx)

        train_size = int(len(dataset) * train_ratio)
        train_dataset, test_dataset = dataset[:train_size], dataset[train_size:]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=30)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=30)
        loaders.append((train_loader, test_loader))
    return loaders