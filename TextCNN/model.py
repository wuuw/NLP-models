import torch
import torch.nn as nn

"""
TextCNN 模型
"""


def create_embedding_layer(embedding_matrix, trainable=True):
    vocab_size, embed_dim = embedding_matrix.size()
    embed_layer = nn.Embedding(vocab_size, embed_dim)
    embed_layer.load_state_dict({'weight': embedding_matrix})
    embed_layer.weight.requires_grad = trainable
    return embed_layer, vocab_size, embed_dim


class TextCNN(nn.Module):
    def __init__(self, embedding_matrix):
        super(TextCNN, self).__init__()
        self.embed, self.vocab_size, self.embed_dim = create_embedding_layer(embedding_matrix, trainable=True)

        self.conv1d_1 = nn.Conv1d(200, 100, 2)  # 输入的 channels 为 embedding_dim
        self.conv1d_2 = nn.Conv1d(200, 100, 3)
        self.conv1d_3 = nn.Conv1d(200, 100, 4)

        self.relu = nn.ReLU()

        self.maxpool_1 = nn.AdaptiveMaxPool1d(1)
        self.maxpool_2 = nn.AdaptiveMaxPool1d(1)
        self.maxpool_3 = nn.AdaptiveMaxPool1d(1)

        self.dropout_1 = nn.Dropout(0.7)
        self.fc_1 = nn.Linear(100 * 3, 32)
        self.dropout_2 = nn.Dropout(0.7)
        self.fc_2 = nn.Linear(32, 2)

    def forward(self, x):
        out = self.embed(x)
        # (bs, sentence_len, embedding_dim) → (bs, embedding_dim, sentence_len)
        out = out.permute([0, 2, 1])

        out_1 = self.conv1d_1(out)
        out_2 = self.conv1d_2(out)
        out_3 = self.conv1d_3(out)
        # (bs, 64, sentence_len - filter_size + 1)

        out_1 = self.relu(out_1)
        out_2 = self.relu(out_2)
        out_3 = self.relu(out_3)

        # 在第 3 维度上取最大值 (bs, 64, 1)
        out_1 = self.maxpool_1(out_1).squeeze(2)
        out_2 = self.maxpool_2(out_2).squeeze(2)
        out_3 = self.maxpool_3(out_3).squeeze(2)

        cat = torch.cat((out_1, out_2, out_3), dim=1)

        out = self.dropout_1(cat)
        out = self.fc_1(out)
        out = self.dropout_2(out)
        out = self.fc_2(out)
        return out
