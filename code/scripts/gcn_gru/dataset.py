import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

ADG_THR = 0.1
PAD = -1e6


def get_data_paths(dataset, signal_type, conn_method, padding=False):
    assert signal_type in (
        'raw_preprocessed',
        'extract_feature'
    ), "Value of signal_type must be 'raw_preprocessed' or 'extract_feature'."
    assert conn_method in (
        'dist',
        'corr',
        'coh',
        'plv'
    ), "Value of adj_method must be one of the ('dist', 'corr', 'coh', 'plv')."

    if padding:
        x_path = f'data/{dataset}/{signal_type}/X_with_padding_zeroed.npy'
        y_path = f'data/{dataset}/{signal_type}/y_zeroed.npy'
    else:
        x_path = f'data/{dataset}/{signal_type}/X_with_padding.npy'
        y_path = f'data/{dataset}/{signal_type}/y.npy'

    adj_path = f'data/{dataset}/{conn_method}_adjacencies/'

    return x_path, y_path, adj_path


class GraphDataset(Dataset):
    @staticmethod
    def _get_edges(adjacencies):
        edges = []
        for adj in adjacencies:
            edges_ = []
            for j in range(len(adj)):
                for i in range(j):
                    if adj[i][j] >= ADG_THR:
                        edges_.extend(([i, j], [j, i]))

            edges.append(np.array(edges_).T)
        return edges

    @staticmethod
    def _get_edge_weights(adjacencies):
        edge_weights = []
        for adj in adjacencies:
            edge_weights_ = []
            for j in range(len(adj)):
                for i in range(j):
                    if adj[i][j] >= ADG_THR:
                        edge_weights_.extend((adj[i][j], adj[i][j]))

            edge_weights.append(np.array(edge_weights_))
        return edge_weights

    @staticmethod
    def _get_targets_and_features(x, y):
        target_range = np.sort(np.unique(y))
        targets = [np.where(i == target_range)[0] for i in y]
        features = [x[i].swapaxes(1, 2) for i in range(len(x))]
        features = [x[:, :, x[0, 0] != PAD] for x in features]
        return targets, features

    def __init__(self, x_path, y_path, adj_path):
        super().__init__()

        x = np.load(x_path)
        y = np.load(y_path)

        matrix_names = os.listdir(adj_path)

        adjs = []
        for name in matrix_names:
            adjs.append(np.load(adj_path+name))

        la, lx = len(adjs), len(x)
        if la < lx:
            if la == 1:
                adjs = adjs * lx
            else:
                raise ValueError(f'Insufficient number of connection matrices: get {la} should {lx}.')

        self.edges = self._get_edges(adjs)
        self.edge_weights = self._get_edge_weights(adjs)
        self.target, self.feature = self._get_targets_and_features(x, y)

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


def get_loaders(dataset, signal_type, conn_method, padding, batch_size, train_ratio=0.9, shuffle=False):
    data_paths = get_data_paths(dataset, signal_type, conn_method, padding)
    dataset = GraphDataset(*data_paths)
    if shuffle:
        dataset = dataset.shuffle()

    train_size = int(len(dataset) * train_ratio)
    train_dataset, test_dataset = dataset[:train_size], dataset[train_size:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=30)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=30)
    return train_loader, test_loader