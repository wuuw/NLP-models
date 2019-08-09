import torch
import numpy as np
import pandas as pd
import jieba
from gensim.models import KeyedVectors

# self defined utils
from utils.dictionary import Dictionary

jieba.load_userdict('../word2vec/500000-dict.txt')


'''################################################
 构建停止词表（funNLP 仓库提供的中文停止词表的子集）
################################################'''
print('1. 正在构建停止词表...\n')

stop_words = []
with open('../word2vec/stopwords.txt') as f:
    for line in f:
        stop_words.append(line.strip())


'''###################
对评论进行中文分词
###################'''
print('2. 正在进行分词...\n')


def seg(data_frame):
    segs = jieba.cut(data_frame['review'], cut_all=False)
    row = []
    # 去除停止词表中的词汇
    for seg in segs:
        if seg not in stop_words and seg != ' ':
            row.append(seg)
    return row


df = pd.read_csv('../datasets/online_shopping_10_cats.csv')
df = df[['label', 'review']].dropna().copy()  # 删除具有缺省的行
df['review'] = df.apply(seg, axis=1)


'''
简单的分词结果数据统计
'''
length_list = [len(review) for review in df['review']]
print('数据集样本最小长度: {}\n'
      '数据集样本最大长度: {}\n'
      '数据集样本平均长度: {:.1f}'
      .format(min(length_list), max(length_list), sum(length_list) / len(length_list)))

print('正样本数量: {}\n'
      '负样本数量: {}\n'
      .format(len(df[df['label'] == 1]), len(df[df['label'] == 0])))


'''##################
加载腾讯预训练词向量
##################'''
print('3. 正在加载腾讯的预训练词向量（子集，50 万词）...\n')

vector = KeyedVectors.load_word2vec_format('../word2vec/500000-small.txt')


'''#########################
按照词频截断，获取训练集字典
#########################'''
print('4. 正在获取训练集字典...\n')

dictionary = Dictionary()

for review in df['review']:
    for i, word in enumerate(review):
        dictionary.add_word(word)

# 选择出现次数至少为 2 的词构建词典
dictionary.cut_dict(2)


'''#####################
对训数集中的词汇进行统计
#####################'''
print('5. 正在对分词后数据集的词汇、词量进行统计...\n')

em_word = 0
for word in dictionary.word2idx.keys():
    if word in vector:
        em_word += 1

print('词汇总量为: {}'
      '\n具有词向量的词汇量为: {}'
      '\n占比为: {:.2f}%\n'
      .format(len(dictionary.word2idx), em_word, em_word/len(dictionary.word2idx)*100))

'''
对训练集中的词汇数进行统计
'''
# 按照词汇出现数量进行排序，大→小
dict_list = [(key, value) for key, value in dictionary.word2count.items()]
dict_list.sort(key=lambda x:x[1], reverse=True)

# 统计词数总量和具有词向量的词汇量
total_count = 0
em_count = 0
for d in dict_list:
    total_count += d[1]
    if d[0] in vector:
        em_count += d[1]
print('词汇总数为: {}'
      '\n具有词向量的词汇量为: {}'
      '\n占比为: {:.2f}%\n'
      .format(total_count, em_count, em_count/total_count*100))


'''######################
构建直接用于模型的数据集
######################'''
print('6. 正在构建可直接用于模型的 Tensor 数据...\n')

EMBEDDING_DIM = 200

# 词嵌入矩阵, 首先全部用 0 向量初始化
embedding_matrix = np.zeros((len(dictionary.word2idx), EMBEDDING_DIM), dtype=np.float32)

# 依据 word2idx 的映射，为 embedding_matrix 填充词向量
# 并将 DataFrame 中的文本数据序列化

seqs = []  # 用来装载每一条训练序列
labels = []  # 用来装载分类标签 0/1

for review in df['review']:
    # 一条训练数据
    seq = []
    for i, word in enumerate(review):
        # 如果在字典中没有找到该词汇，设置为 1 即 <UNK>
        embedding_index = dictionary.word2idx.get(word, 1)
        seq.append(embedding_index)

        # 如果该词有预训练词向量，用其初始化词向量
        # 否则保持使用 0 向量初始化
        if word in vector:
            embedding_matrix[embedding_index, :] = vector[word]

    seqs.append(seq)

for label in df['label']:
    labels.append(label)

# 构建词嵌入 Tensor 并保存
embedding_matrix = torch.Tensor(embedding_matrix)
torch.save(embedding_matrix, '../middle_ware/embedding_matrix.mat')
print('7. embedding_matrix 已保存至 middle_ware/文件夹，将用于训练模型\n')


'''#############################################
对数据集进行 padding/truncation 操作, pre / post 控制从前/后 truncate
#############################################'''
FIXED_LENGTH = 25


def pad_truncate(dataset, length=50, type='pre'):
    fixed_texts = []

    for text in dataset:
        fixed_text = []

        if len(text) < length:
            fixed_text = text[:] + [0] * (length - len(text))
        else:
            if type == 'pre':  # 截取前面 length 个词
                fixed_text = text[:length]
            elif type == 'post':  # 截取后面 length 个词
                fixed_text = text[-length:]

        fixed_texts.append(fixed_text)

    return fixed_texts


texts = pad_truncate(seqs, length=FIXED_LENGTH, type='pre')


'''#####################
划分训练集、验证集(5000)
#####################'''
np.random.seed(1)
data_size = len(texts)
indices = np.arange(data_size)
np.random.shuffle(indices)

# 训练集
train_texts = np.array(texts)[indices[:-5000]]
train_labels = np.array(labels)[indices[:-5000]]
# 验证集
val_texts = np.array(texts)[indices[-5000:]]
val_labels = np.array(labels)[indices[-5000:]]

train_x = torch.Tensor(train_texts).long()
train_y = torch.Tensor(train_labels).long()
val_x = torch.Tensor(val_texts).long()
val_y = torch.Tensor(val_labels).long()

# 将 Tensor 数据保存为文件
torch.save(train_x, '../middle_ware/train_x.mat')
torch.save(train_y, '../middle_ware/train_y.mat')
torch.save(val_x, '../middle_ware/val_x.mat')
torch.save(val_y, '../middle_ware/val_y.mat')
print('8. Tensor 数据已保存至 middle_ware/文件夹,将用于训练模型\n(所有数据均处理为统一长度：25)')