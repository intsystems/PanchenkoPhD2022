import torch
import torch.nn as nn
import torch.nn.functional as F

N_CH = 62


class LSTM(torch.nn.Module):
    def __init__(self, seq_len, output_dim):
        super().__init__()
        self.lstm1 = nn.LSTM(
            N_CH,
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

        self.pool_size = 256

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.35)
        #self.linear1 = nn.Linear(256, 256)
        #self.linear1 = nn.Linear(250 * 128, 128)
        self.linear1 = nn.Linear(seq_len * 256, 256)
        #self.linear1 = nn.Linear(self.pool_size * 256, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, output_dim)

    # x.shape = NxF*TxK
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)

        #x = x.transpose(1, 2)
        #x = F.adaptive_max_pool1d(x, output_size=self.pool_size).transpose(1, 2)
        x = self.dropout3(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)
        x = F.relu(self.linear2(x))
        x = self.dropout3(x)
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x