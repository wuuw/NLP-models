import torch.nn as nn


class DropLinear(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super(DropLinear, self).__init__()
        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_normal_(self.linear.weight.data)

    def forward(self, x):
        return self.linear(self.dropout(x))