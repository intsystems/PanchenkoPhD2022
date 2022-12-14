import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, TopKPooling, GCNConv

N_CH = 62


class GCN_LSTM(torch.nn.Module):
    def __init__(self, node_features, seq_len, output_dim):
        super().__init__()
        self.conv1 = ChebConv(
            in_channels=node_features,
            out_channels=node_features,
            K=2,
            normalization="sym",
            bias=True
        )
        self.conv2 = ChebConv(
            in_channels=node_features,
            out_channels=node_features,
            K=2,
            normalization="sym",
            bias=True
        )
        self.conv3 = ChebConv(
            in_channels=node_features,
            out_channels=node_features,
            K=2,
            normalization="sym",
            bias=True
        )
        self.pool_ratio = 0.8
        self.pool1 = TopKPooling(node_features, ratio=self.pool_ratio)
        #self.pool2 = TopKPooling(node_features, ratio=self.pool_ratio)
        #self.new_n_ch = int(np.ceil(np.ceil(N_CH * self.pool_ratio) * self.pool_ratio))
        self.new_n_ch = N_CH#int(np.ceil(N_CH * self.pool_ratio))
        self.lstm1 = nn.LSTM(
            self.new_n_ch,
            64,
            bidirectional=False,
            num_layers=1,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            64,
            128,
            bidirectional=False,
            num_layers=1,
            batch_first=True
        )
        self.lstm3 = nn.LSTM(
            128,
            256,
            bidirectional=False,
            num_layers=1,
            batch_first=True
        )

        #self.pool_size = 256

        self.dropout0 = nn.Dropout(0.05)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.35)
        #self.linear1 = nn.Linear(250 * 128, 128)
        self.linear1 = nn.Linear(seq_len * 256, 256)
        #self.linear1 = nn.Linear(self.pool_size * 256, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, output_dim)

    # -> x.shape = N*KxFxT
    def forward(self, x, edge_index, edge_weight, batch):
        x_conv = []
        for t in range(x.shape[-1]):
            x_t = self.conv1(x[..., t], edge_index[t], edge_weight[t], batch[t])
            x_t = F.leaky_relu(x_t)
            #x_t, edge_index_, edge_weight_, batch_, _, _ = self.pool1(x_t, edge_index, edge_weight, batch)
            x_t = self.conv2(x_t, edge_index[t], edge_weight[t], batch[t])
            #x_t, edge_index_, edge_weight_, batch_, _, _ = self.pool2(x_t, edge_index_, edge_weight_, batch_)
            #x_t = self.conv3(x_t, edge_index_, edge_weight_, batch_)
            x_conv.append(x_t.unsqueeze(-1))

        # -> x_conv.shape = N*KxFxT
        x_conv = torch.cat(x_conv, -1)
        x_conv = x_conv.reshape(-1, self.new_n_ch, x_conv.shape[1], x_conv.shape[2])
        x_conv = x_conv.transpose(1, 2).transpose(2, 3).transpose(1, 2)
        #x_conv = x_conv.transpose(2, 3).transpose(1, 2)
        x_conv = x_conv.reshape(x_conv.shape[0], -1, self.new_n_ch)
        #x_conv = x_conv.reshape(x_conv.shape[0], x_conv.shape[1], -1)
        #x_conv = self.dropout0(x_conv)

        # -> x_conv.shape = NxF*TxK
        x_lstm, _ = self.lstm1(x_conv)
        x_lstm = self.dropout1(x_lstm)
        x_lstm, _ = self.lstm2(x_lstm)
        x_lstm = self.dropout2(x_lstm)
        x_lstm, _ = self.lstm3(x_lstm)

        #x_lstm = x_lstm.transpose(1, 2)
        #x_lstm = F.adaptive_max_pool1d(x_lstm, output_size=self.pool_size).transpose(1, 2)
        x_lstm = self.dropout3(x_lstm)

        x_lin = torch.flatten(x_lstm, 1)
        x_lin = F.relu(self.linear1(x_lin))
        x_lin = self.dropout1(x_lin)
        x_lin = F.relu(self.linear2(x_lin))
        x_lin = self.dropout3(x_lin)
        x_lin = F.relu(self.linear3(x_lin))
        x_lin = self.linear4(x_lin)
        return x_lin