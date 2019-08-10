import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.highway import LinearHighway
from modules.conv1d import Conv1dList


class TextCNNwithHighway(nn.Module):
    def __init__(self, pre_trained, in_channels, out_channels, filter_sizes):
        super(TextCNNwithHighway, self).__init__()

        self.embed = nn.Embedding.from_pretrained(pre_trained, freeze=False)

        self.conv1d_list = Conv1dList(in_channels, out_channels, filter_sizes)

        self.highway = nn.ModuleList([
            LinearHighway(out_channels * len(filter_sizes), 256),
            LinearHighway(256, 64),
            LinearHighway(64, 16),
            LinearHighway(16, 2)
        ])

    def forward(self, x):
        embed = self.embed(x)
        out = embed.permute([0, 2, 1])

        outs = self.conv1d_list(out)

        outs = [F.relu(out) for out in outs]

        outs = [F.adaptive_max_pool1d(out, 1) for out in outs]

        out = torch.cat(tuple(outs), dim=1).squeeze(2)

        for layer in self.highway:
            out = layer(out)

        return out
