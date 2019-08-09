import torch.nn as nn

"""
带有 Dropout 功能的全连接层

Parameters:
@ in_features: 输入特征个数
@ out_features: 输出特征个数
@ dropout_rate: Dropout rate

Return:
全连接层的输出
"""


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