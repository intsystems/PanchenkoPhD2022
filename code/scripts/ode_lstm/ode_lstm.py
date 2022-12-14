# Code based on the implementation of the ODE-LSTM Authors Mathias Lechner ad Ramin Hasani

import torch
import torch.nn as nn
from torchdyn.models import NeuralODE


class OdeLstmCell(nn.Module):
    def __init__(self, input_size, hidden_size, solver_type="dopri5"):
        super(OdeLstmCell, self).__init__()
        self.solver_type = solver_type
        self.fixed_step_solver = solver_type.startswith("fixed_")
        self.lstm = nn.LSTMCell(input_size, hidden_size)

        self.f_node = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        if not self.fixed_step_solver:
            self.node = NeuralODE(self.f_node, solver=solver_type)
        else:
            options = {
                "fixed_euler": self.euler,
                "fixed_heun": self.heun,
                "fixed_rk4": self.rk4,
            }
            if solver_type not in options.keys():
                raise ValueError("Unknown solver type '{:}'".format(solver_type))
            self.node = options[self.solver_type]

    def forward(self, input, hx, ts):
        new_h, new_c = self.lstm(input, hx)
        if self.fixed_step_solver:
            new_h = self.solve_fixed(new_h, ts)
        else:
            indices = torch.argsort(ts)
            batch_size = ts.size(0)
            device = input.device
            s_sort = ts[indices]
            s_sort = s_sort + torch.linspace(0, 1e-4, batch_size, device=device)
            trajectory = self.node.trajectory(new_h, s_sort)
            new_h = trajectory[indices, torch.arange(batch_size, device=device)]

        return (new_h, new_c)

    def solve_fixed(self, x, ts):
        ts = ts.view(-1, 1)
        for _ in range(3):  # 3 unfolds
            x = self.node(x, ts * (1.0 / 3))
        return x

    def euler(self, y, delta_t):
        dy = self.f_node(y)
        return y + delta_t * dy

    def heun(self, y, delta_t):
        k1 = self.f_node(y)
        k2 = self.f_node(y + delta_t * k1)
        return y + delta_t * 0.5 * (k1 + k2)

    def rk4(self, y, delta_t):
        k1 = self.f_node(y)
        k2 = self.f_node(y + k1 * delta_t * 0.5)
        k3 = self.f_node(y + k2 * delta_t * 0.5)
        k4 = self.f_node(y + k3 * delta_t)

        return y + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


class OdeLstm(nn.Module):
    """ODE LSTM Neural Network.

    Based on the paper
    "Learning Long-Term Dependencies in Irregularly-Sampled Time Series"
    https://arxiv.org/abs/2006.04418
    """

    def __init__(
        self,
        in_features,
        hidden_size,
        out_feature,
        return_sequences=True,
        solver_type="dopri5",
    ):
        """
        Args:
            in_features: number of channels
            hidden_size: hidden dimension
            out_feature: number of instances to predict (num of classes)
            return_sequences: whether to get predictions for each input (if True),
                or only one predictions per sequens(if False)
            solver_type="dopri5": method for solving ODE

        """
        super(OdeLstm, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_feature = out_feature
        self.return_sequences = return_sequences

        self.rnn_cell = OdeLstmCell(in_features, hidden_size, solver_type=solver_type)
        self.fc = nn.Linear(self.hidden_size, self.out_feature)

    def forward(self, x, timespans):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        hidden_state = [
            torch.zeros((batch_size, self.hidden_size), device=device),
            torch.zeros((batch_size, self.hidden_size), device=device),
        ]
        outputs = []
        last_output = torch.zeros((batch_size, self.out_feature), device=device)
        for t in range(seq_len):
            inputs = x[:, t]
            ts = timespans[:, t].squeeze()
            hidden_state = self.rnn_cell.forward(inputs, hidden_state, ts)
            current_output = self.fc(hidden_state[0])
            outputs.append(current_output)
            last_output = current_output
        if self.return_sequences:
            outputs = torch.stack(outputs, dim=1)
        else:
            outputs = last_output
        return outputs