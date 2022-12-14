import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, TopKPooling

N_CH = 62
EPS = 1e-6


def _get_batch_attrs(adj, x):
    bs = x.shape[0] // N_CH
    edge_index = torch.LongTensor([(i, j) for i in range(len(adj)) for j in range(len(adj))]).repeat(bs, 1).T
    edge_weight = adj.ravel().repeat(bs)
    return edge_index.to(x.device), edge_weight.to(x.device)


class GCN_LSTM(torch.nn.Module):
    def __init__(self, node_features, seq_len, output_dim, dataset='seed', adj_init='dist'):
        super().__init__()
        self.conv1 = ChebConv(
            in_channels=node_features,
            out_channels=node_features,
            K=2,
            normalization=None,#"sym",
            bias=True
        )
        self.conv2 = ChebConv(
            in_channels=node_features,
            out_channels=node_features,
            K=2,
            normalization=None,#"sym",
            bias=True
        )
        self.lstm1 = nn.LSTM(
            N_CH * node_features,
            128,
            bidirectional=False,
            num_layers=1,
            batch_first=True
        )

        self.dropout0 = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.25)
        self.linear1 = nn.Linear(250 * 128, 128)
        self.linear4 = nn.Linear(128, output_dim)

        #adj_init_matrix = np.load(f'data/{dataset}/{adj_init}_adjacencies/matrix.npy')
        #adj_init_matrix = torch.ones((N_CH, N_CH)) - torch.eye(N_CH)
        adj_init_matrix = torch.rand(N_CH, N_CH)
        adj_init_matrix.fill_diagonal_(0)
        self.adj_ = nn.Parameter(adj_init_matrix, requires_grad=True)


    # -> x.shape = N*KxFxT
    def forward(self, x, edge_index_, edge_weight_, batch):
        self.adj = F.relu(0.5 * (self.adj_.T + self.adj_))
        self.adj = self.adj / (self.adj.sum(axis=1)[..., None] + EPS)
        #with torch.no_grad():
        edge_index, edge_weight = _get_batch_attrs(self.adj, x)
        #edge_index = edge_index.detach()
        #edge_weight = edge_weight.detach()
        x_conv = []
        for t in range(x.shape[-1]):
            x_t = self.conv1(x[..., t], edge_index, edge_weight, batch)
            x_t = F.leaky_relu(x_t)
            x_t = self.conv2(x_t, edge_index, edge_weight, batch)
            x_conv.append(x_t.unsqueeze(-1))

        # -> x_conv.shape = N*KxFxT
        x_conv = torch.cat(x_conv, -1)
        x_conv = x_conv.reshape(-1, N_CH, x_conv.shape[1], x_conv.shape[2])
        #x_conv = x_conv.transpose(1, 2).transpose(2, 3).transpose(1, 2)
        x_conv = x_conv.transpose(2, 3).transpose(1, 2)
        #x_conv = x_conv.reshape(x_conv.shape[0], -1, N_CH)
        x_conv = x_conv.reshape(x_conv.shape[0], x_conv.shape[1], -1)
        #x_conv = self.dropout0(x_conv)

        # -> x_conv.shape = NxF*TxK
        x_lstm, _ = self.lstm1(x_conv)
        x_lstm = self.dropout1(x_lstm)

        x_lin = torch.flatten(x_lstm, 1)
        x_lin = F.relu(self.linear1(x_lin))
        x_lin = self.dropout1(x_lin)
        #x_lin = F.relu(self.linear2(x_lin))
        #x_lin = self.dropout3(x_lin)
        #x_lin = F.relu(self.linear3(x_lin))
        x_lin = self.linear4(x_lin)
        return x_lin