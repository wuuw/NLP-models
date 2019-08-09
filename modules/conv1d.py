import torch.nn as nn

class Conv1dList(nn.Module):
    def __init__(self, in_channels, out_channels, filter_sizes):
        super(Conv1dList, self).__init__()
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=filter_size)
            for filter_size in filter_sizes
        ])

        self.init_parameters()

    def init_parameters(self):
        for conv in self.conv1d_list:
            nn.init.xavier_normal_(conv.weight.data)

    def forward(self, x):
        return [conv1d(x) for conv1d in self.conv1d_list]
