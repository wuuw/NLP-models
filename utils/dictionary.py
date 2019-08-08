"""
词典类，记录数据集中数显的词汇及其索引和数量
"""


class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word2count = {}

    def add_word(self, word):
        # build word - count
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1

    '''
    仅使用训练集中出现次数 >= 阈值的词汇构建词表
    '''

    def cut_dict(self, cut=2):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        word2count = {}
        idx = 2

        for word in self.word2count.keys():
            if self.word2count[word] >= cut:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
                word2count[word] = self.word2count[word]

        self.word2count = word2count
