# **********************************
# 	Author: Suraj Subramanian
# 	2nd January 2020
# **********************************

import torch
import torch.nn as nn


class GRUDCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, mask_dim, delta_dim, x_mean=None):
        super(GRUDCell, self).__init__()

        self.device = try_gpu()
        self.hidden_dim = hidden_dim

        # Set empirical mean if first GRU-D layer. Subsequent stacked layers don't need the mean.
        if x_mean is not None:
            self.x_mean = x_mean
            self.first_layer = True
        else:
            self.first_layer = False

        self.R_lin = nn.Linear(input_dim + hidden_dim + mask_dim, hidden_dim)  # RESET
        self.Z_lin = nn.Linear(input_dim + hidden_dim + mask_dim, hidden_dim)  # UPDATE
        self.tilde_lin = nn.Linear(input_dim + hidden_dim + mask_dim, hidden_dim)  # CANDIDATE STATE
        self.gamma_h_lin = nn.Linear(delta_dim, hidden_dim)  # HIDDEN STATE DECAY PARAMETERS
        if self.first_layer:
            self.gamma_x_lin = nn.Linear(delta_dim, delta_dim)  # INPUT DECAY PARAMETERS

        # XAVIER INIT FOR FASTER CONVERGENCE
        nn.init.xavier_normal_(self.R_lin.weight)
        nn.init.xavier_normal_(self.Z_lin.weight)
        nn.init.xavier_normal_(self.tilde_lin.weight)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim, device=self.device, dtype=torch.float)

    def forward(self, x, obs_mask, delta, x_tm1, h):  # inputs = (batch_size x type x dim)
        # DECAY OPERATIONS
        if self.first_layer:
            # Input is decayed only in the first GRU-D layer
            gamma_x = torch.exp(-torch.max(torch.zeros(delta.size(), device=self.device), self.gamma_x_lin(delta)))
            x = (obs_mask * x) + (1-obs_mask)*((gamma_x*x_tm1) + (1-gamma_x)*self.x_mean)
        gamma_h = torch.exp(-torch.max(torch.zeros(h.size(), device=self.device), self.gamma_h_lin(delta)))
        h = torch.squeeze(gamma_h*h)

        # GRU OPERATIONS
        gate_in = torch.cat((x, h, obs_mask), -1)
        z = torch.sigmoid(self.Z_lin(gate_in))
        r = torch.sigmoid(self.R_lin(gate_in))
        tilde_in = torch.cat((x, r*h, obs_mask), -1)
        tilde = torch.tanh(self.tilde_lin(tilde_in))
        h = (1-z)*h + z*tilde

        return h


class StackedGRUDClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, x_mean, aux_op_dims=[], op_act=None):
        """
        Example classifier with 2 GRUD layers, {aux_op_dims} auxiliary target classifiers, and 1 primary target classifier
        """
        super(StackedGRUDClassifier, self).__init__()
        self.device = try_gpu()

        # Assign input and hidden dim
        self.hidden_dim = input_dim*6
        self.output_dim = output_dim
        self.x_mean = torch.tensor(x_mean, device=self.device, dtype=float)

        # Activation function
        self.op_act = op_act or nn.LeakyReLU()

        # GRU layers
        self.gru1 = GRUDCell(input_dim, self.hidden_dim, input_dim, input_dim, self.x_mean)
        self.gru2 = GRUDCell(self.hidden_dim, self.hidden_dim, input_dim, input_dim)

        # 4 FC Layers with dropout for each Aux output
        self.aux_fc_layers = nn.ModuleList()
        for aux in aux_op_dims:
            self.aux_fc_layers.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim//3),
                    nn.Dropout(0.3),
                    self.op_act,
                    nn.Linear(self.hidden_dim//3, self.hidden_dim//9),
                    nn.Dropout(0.3),
                    self.op_act,
                    nn.Linear(self.hidden_dim//9, self.hidden_dim//27),
                    nn.Dropout(0.3),
                    self.op_act,
                    nn.Linear(self.hidden_dim//27, aux)
                )
            )
            nn.init.xavier_normal_(self.aux_fc_layers[-1][0].weight, 0.1)
            nn.init.xavier_normal_(self.aux_fc_layers[-1][3].weight, 0.1)
            nn.init.xavier_normal_(self.aux_fc_layers[-1][6].weight, 0.1)
            nn.init.xavier_normal_(self.aux_fc_layers[-1][9].weight, 0.1)

        # 4 FC Layers with dropout for primary output
        self.fc_op = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//3),
            nn.Dropout(0.3),
            self.op_act,
            nn.Linear(self.hidden_dim//3, self.hidden_dim//9),
            nn.Dropout(0.3),
            self.op_act,
            nn.Linear(self.hidden_dim//9, self.hidden_dim//27),
            nn.Dropout(0.3),
            self.op_act,
            nn.Linear(self.hidden_dim//27, output_dim)
        )
        nn.init.xavier_normal_(self.aux_fc_layers[-1][0].weight, 0.1)
        nn.init.xavier_normal_(self.aux_fc_layers[-1][3].weight, 0.1)
        nn.init.xavier_normal_(self.aux_fc_layers[-1][6].weight, 0.1)
        nn.init.xavier_normal_(self.aux_fc_layers[-1][9].weight, 0.1)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim, device=self.device, dtype=torch.float)

    def forward(self, inputs, h1, h2):
        inputs = inputs.float()
        batch_size = inputs.size(0)
        step_size = inputs.size(2)
        outputs = None
        tr_out = torch.zeros(batch_size, step_size, self.output_dim)

        # TR prediction
        for t in range(step_size):
            h1, h2 = self.step(inputs[:, :, t:t+1, :], h1, h2)
            tr_out[:, t:t+1, :] = torch.unsqueeze(self.fc_op(h2), 1)

        # OP prediction
        outputs = self.fc_op(h2)

        # Aux prediction
        aux_out = []
        for aux_layer in self.aux_fc_layers:
            a_o = aux_layer(h2)
            aux_out.append(a_o)

        return outputs, tr_out, aux_out, h2

    def predict(self, inputs, h1, h2):
        """
        Returns: pred_op, h2
        """
        step_size = inputs.size(2)
        outputs = None

        for t in range(step_size):
            h1, h2 = self.step(inputs[:, :, t:t+1, :], h1, h2)

        outputs = self.fc_op(h2)
        return outputs, h2

    def step(self, inputs, h1, h2):  # inputs = (batch_size x type x dim)
        inputs = inputs.float()
        x, obs_mask, delta, x_tm1 = torch.squeeze(inputs[:, 0, :, :]), \
            torch.squeeze(inputs[:, 1, :, :]), \
            torch.squeeze(inputs[:, 2, :, :]), \
            torch.squeeze(inputs[:, 3, :, :])

        h1 = self.gru1(x, obs_mask, delta, x_tm1, h1)
        h1 = nn.Dropout()(h1)
        h2 = self.gru2(h1, obs_mask, delta, x_tm1, h2)
        h2 = nn.Dropout()(h2)

        return h1, h2


def try_gpu():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
