import torch
import numpy as np
from torch_geometric.nn import ChebConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

N_ch = 62


class GConvGRU(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: str = "sym",
        bias: bool = True,
    ):
        super(GConvGRU, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization
        self.bias = bias
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):

        self.conv_x_z = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_z = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_reset_gate_parameters_and_layers(self):

        self.conv_x_r = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_r = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_candidate_state_parameters_and_layers(self):

        self.conv_x_h = ChebConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

        self.conv_h_h = ChebConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            K=self.K,
            normalization=self.normalization,
            bias=self.bias,
        )

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H, init='zero'):
        if H is None:
            if init == 'uniform':
                k = - np.sqrt(1 / self.out_channels)
                H = torch.nn.init.uniform_(torch.empty(X.shape[0], self.out_channels), k, -k).to(X.device)
            else:
                H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, batch, H, lambda_max):
        Z = self.conv_x_z(X, edge_index, edge_weight, batch, lambda_max=lambda_max)
        Z = Z + self.conv_h_z(H, edge_index, edge_weight, batch, lambda_max=lambda_max)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, batch, H, lambda_max):
        R = self.conv_x_r(X, edge_index, edge_weight, batch, lambda_max=lambda_max)
        R = R + self.conv_h_r(H, edge_index, edge_weight, batch, lambda_max=lambda_max)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, batch, H, R, lambda_max):
        H_tilde = self.conv_x_h(X, edge_index, edge_weight, batch, lambda_max=lambda_max)
        H_tilde = H_tilde + self.conv_h_h(H * R, edge_index, edge_weight, batch, lambda_max=lambda_max)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        batch: torch.LongTensor = None,
        H: torch.FloatTensor = None,
        lambda_max: torch.Tensor = None,
    ) -> torch.FloatTensor:
        """
        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, batch, H, lambda_max)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, batch, H, lambda_max)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, batch, H, R, lambda_max)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, h_dim, k, n_output):
        super(RecurrentGCN, self).__init__()
        self.h_dim = h_dim
        self.recurrent = GConvGRU(node_features, h_dim, k)
        self.linear = torch.nn.Linear(h_dim * N_ch, n_output)

    # x.shape = N*KxFxT
    def forward(self, x, edge_index, edge_weight, batch):
        # hidden.shape = N*Kxh_dim
        hidden = None
        for t in range(x.shape[-1]):
            hidden = self.recurrent(x[..., t], edge_index, edge_weight, batch, hidden)

        hidden = hidden.reshape(-1, N_ch, self.h_dim)
        # hidden = F.relu(hidden)
        hidden = torch.flatten(hidden, start_dim=1)
        # hidden = global_mean_pool(hidden, batch)
        output = self.linear(hidden)
        return output