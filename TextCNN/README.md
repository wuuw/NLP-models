# TextCNN

### Steps

1. 使用首页 `REAM.ME` 中的链接下载腾讯中文预训练词向量，并存放在 `/word2vec/` 目录中；

2. 运行 [`/utils/preprocess_pipeline.py`](../utils/preprocess_pipeline.py) 对文本进行预处理，会自动处理 [`/datasets/online_shopping_10_cats.csv`](../datasets/online_shopping_10_cats.csv) 中的文本数据，并自动生成 `torch.Tensor` 类型的词嵌入矩阵（用于模型的 Embedding Layer）和四个 `torch.Tensor` 类型的训练集、验证集（train_x.mat / train_y.mat / val_x.mat / val_y.mat）；以上 5 个文件均会存放在 `/middle_ware/` 文件夹中；

3. 运行 `/TextCNN/` 下的 `main.py` 训练模型。