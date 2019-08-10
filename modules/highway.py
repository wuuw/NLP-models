import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHighway(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearHighway, self).__init__()
        # transform features: (batch_size, in_features) → （batch_size, out_features)
        self.pre_linear = nn.Linear(
            in_features=in_features,
            out_features=out_features
        )
        # out for gate
        self.gate_linear = nn.Linear(
            in_features=out_features,
            out_features=out_features
        )
        # highway transform gata
        self.transform_gate = nn.Linear(
            in_features=out_features,
            out_features=out_features
        )

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_normal_(self.pre_linear.weight.data)
        nn.init.xavier_normal_(self.gate_linear.weight.data)
        nn.init.xavier_normal_(self.transform_gate.weight.data)

    def forward(self, x):
        # features transform layer
        highway_input = F.relu(self.pre_linear(x))
        # highway layer
        for_gate = F.relu(self.gate_linear(highway_input))

        transform_gate = F.sigmoid(self.transform_gate(highway_input))
        carry_gate = 1 - transform_gate

        out = torch.mul(for_gate, transform_gate) + torch.mul(highway_input, carry_gate)

        return out
