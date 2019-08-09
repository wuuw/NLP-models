# NLP-models

### Introduction

这个仓库实现了笔者认为值得实现的 NLP 深度学习模型，并在实现各模型时对实践和模型进行了总结。总结内容暂未 push。

### Content

- [TextCNN](./TextCNN/) 

  参考文献：[Convolutional Neural Networks for Sentence Classification](<https://www.aclweb.org/anthology/D14-1181>) (TextCNN origin paper)

  参考文献：[A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional
  Neural Networks for Sentence Classification](<https://arxiv.org/pdf/1510.03820.pdf>) (对 TextCNN 模型的分析及调参建议)


### Datasets

- [商品评论中文数据集](./datasets/)，包含十种商品的 Positive/Negative 分类；针对情感分类，可分为 2 类；数据包含 0/1 两类各约 30000 条。

### Pre-trained Word Embedding

- 使用 [腾讯开源的中文词向量](<https://cloud.tencent.com/developer/article/1356164>) （精简版）；

- 原词向量包含 800 万词表大小的词向量，解压后约 16GB，实验环境下可使用下表精简版词向量，对于大部分数据集也能覆盖绝大部分词汇。例如，使用以上商品评论中文数据集，使用包含 50 万词汇的词向量（约 900 MB）能够覆盖超过 90% 的词汇；

- [腾讯预训练词向量下载地址](https://pan.baidu.com/s/1TvTlHONTagk1nWKyV5SVJQ)（密码: lj7c），包括完整词向量和精简词向量。`word_embeddings`文件夹为词向量文件夹，`word_dictionaries`文件夹为各词向量对应词典文件；

- **如链接失效，加我微信（WeChat ID: fivesheeps）重新给你生成 : D**

| Vocab    | 5 千 | 4.5 万 |  7 万  | 10 万  | 50 万  | 100 万  | 200 万  | 800 万 (Full) |
| -------- | :--: | :----: | :----: | :----: | :----: | :-----: | :-----: | :-----------: |
| **Size** | 9 MB | 82 MB  | 127 MB | 182 MB | 909 MB | 1.77 GB | 3.55 GB |     16 GB     |

