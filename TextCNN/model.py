import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.conv1d import Conv1dList
from modules.drop_linear import DropLinear
"""
TextCNN 模型
"""


class TextCNN(nn.Module):

    def __init__(self, pre_trained, in_channels, out_channels, filter_sizes):
        super(TextCNN, self).__init__()
        self.embed = nn.Embedding.from_pretrained(pre_trained, freeze=False)

        self.conv1d_list = Conv1dList(in_channels=in_channels,
                                      out_channels=out_channels,
                                      filter_sizes=filter_sizes)

        self.drop_linear = DropLinear(out_channels * len(filter_sizes), 2, 0.7)

    def forward(self, x):
        out = self.embed(x)
        # (bs, sentence_len, embedding_dim) → (bs, embedding_dim, sentence_len)
        out = out.permute([0, 2, 1])

        outs = self.conv1d_list(out)
        # (bs, 64, sentence_len - filter_size + 1)
        outs = [F.relu(out) for out in outs]
        # 在第 3 维度上取最大值 (bs, 64, 1)
        outs = [F.adaptive_max_pool1d(out, 1) for out in outs]

        cat = torch.cat(tuple(outs), dim=1).squeeze(2)

        out = self.drop_linear(cat)

        return out
