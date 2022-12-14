import torch
import torch.nn as nn
from torch_geometric.nn.conv import ChebConv

N_CH = 62


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, hidden_dim, gn_layers, ks, dropout, output_dim):
        super().__init__()
        self.gru = nn.GRU(
            node_features*N_CH,
            hidden_dim,
            bidirectional=False,
            num_layers=gn_layers,
            dropout=dropout,
            batch_first=True
        )
        self.gconv1 = ChebConv(
            in_channels=node_features,
            out_channels=node_features,
            K=ks[0],
            normalization="sym",
            bias=True,
        )
        self.gconv2 = ChebConv(
            in_channels=node_features,
            out_channels=node_features,
            K=ks[1],
            normalization="sym",
            bias=True,
        )
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU()

    # x.shape = N*KxFxT
    def forward(self, x, edge_index, edge_weight, batch):
        x_conv = []
        for t in range(x.shape[-1]):
            x_t = self.leaky_relu(self.gconv1(x[..., t], edge_index, edge_weight, batch))
            x_t = self.gconv2(x_t, edge_index, edge_weight, batch)
            x_conv.append(x_t.unsqueeze(-1))

        # x_conv.shape = N*KxHxT
        x_conv = torch.cat(x_conv, -1)
        x_conv = x_conv.reshape(-1, N_CH*x_conv.shape[1], x_conv.shape[2])
        x_conv = x_conv.transpose(1, 2)

        # x_conv.shape = NxTxK*H
        _, hidden = self.gru(x_conv)
        hidden = hidden[0, :, :]
        hidden = self.dropout(hidden)
        output = self.linear(hidden)
        return output