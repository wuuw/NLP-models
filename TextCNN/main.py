from TextCNN.model import TextCNN
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from torchsummary import summary

# hyperparameters
embedding_dim = 200
feature_maps = 100
batch_size = 64
epochs = 10
learning_rate = 0.0003
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
从文件加载训练数据，并使用 TensorDataset 和 DataLoader 存放数据集
'''
train_x, train_y, val_x, val_y = map(
    torch.load, ('../middle_ware/train_x.mat', '../middle_ware/train_y.mat',
                 '../middle_ware/val_x.mat', '../middle_ware/val_y.mat')
)

embedding_matrix = torch.load('../middle_ware/embedding_matrix.mat')


# 加载为 TensorDataset 对象
train_ds = TensorDataset(train_x, train_y)
val_ds = TensorDataset(val_x, val_y)

# 加载为 DataLoader 对象
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)

# initialize Model
model = TextCNN(embedding_matrix, embedding_dim, feature_maps, [2, 3, 4]).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# fit
for epoch in range(epochs):
    model.train()

    with tqdm(enumerate(train_dl)) as t:
        for i, (x, y) in t:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = criterion(out, y)

            # backprop and optimizer
            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            if (i + 1) % 50 == 0:
                print('Epoch: [{}/{}], 正在训练：[{}/{}] 个样本'
                      .format(epoch + 1, epochs, (i + 1) * batch_size, len(train_x)))

                with torch.no_grad():
                    out_ = model(x)
                    correct = torch.sum(torch.argmax(out_, dim=1) == y).item()
                    print('Val: {:.3f}%'.format(correct / batch_size * 100))
t.close()


print('###################################\n')
'''
训练集 Accuracy
'''
with torch.no_grad():
    correct = 0

    for _, (x_, y_) in enumerate(train_dl):
        out_ = model(x_.to(device))
        correct += torch.sum(torch.argmax(out_, dim=1) == y_.to(device)).item()

    print('训练集 Acc: {:.3f}%'.format(correct / len(train_x) * 100))

'''
验证集 Accuracy
'''
with torch.no_grad():
    correct = 0

    for _, (x_, y_) in enumerate(val_dl):
        out_ = model(x_.to(device))
        correct += torch.sum(torch.argmax(out_, dim=1) == y_.to(device)).item()

    print('验证集 Acc: {:.3f}%'.format(correct / len(val_x) * 100))