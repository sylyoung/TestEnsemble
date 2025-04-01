import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

class AutoEncoder(nn.Module):
    def __init__(self, input_size, category_size,source_wise_template):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Linear(input_size, category_size)
        self.decoder = nn.Linear(category_size, input_size)
        self.source_wise_template = source_wise_template

    def forward(self, x):
        y = torch.softmax(self.encoder(x), dim=1)
        x_reconstr = self.decoder(y)
        exp_x_reconstr_tmp = torch.exp(x_reconstr)
        x_reconstr = exp_x_reconstr_tmp / (exp_x_reconstr_tmp @ self.source_wise_template)
        return y, x_reconstr

def LAA_net(preds, preds_one_hot, num_labels):

    n_classifier, n_samples = preds.shape
    votes_mat = np.zeros((num_labels, n_samples))
    for i in range(n_classifier):
        for j in range(n_samples):
            class_id = preds[i, j]
            votes_mat[class_id, j] += 1
    voted_preds = []
    for i in range(n_samples):
        pred = np.random.choice(np.flatnonzero(votes_mat[:, i] == votes_mat[:, i].max()))
        voted_preds.append(pred)
    voted_preds = np.array(voted_preds)

    user_labels = preds_one_hot
    majority_y = voted_preds
    category_size = num_labels
    source_num = preds.shape[0]
    n_samples = preds.shape[1]
    input_size = source_num * category_size
    batch_size = n_samples

    source_wise_template = torch.zeros((input_size, input_size))
    for i in range(input_size):
        source_wise_template[i * category_size:(i + 1) * category_size,
        i * category_size:(i + 1) * category_size] = 1

    # 初始化模型、损失函数和优化器
    model = AutoEncoder(input_size, category_size,source_wise_template)
    criterion_cross_entropy = nn.CrossEntropyLoss(reduction='mean')
    optimizer_classifier_x_y = optim.Adam(model.parameters(), lr=0.005)

    # 训练过程
    epochs = 50
    for epoch in range(epochs):
        total_hit = 0

        for i in range(n_samples // batch_size):
            # 获取当前批次数据
            batch_x = torch.FloatTensor(user_labels)  # 这里需要根据你的数据进行调整
            batch_majority_y = torch.FloatTensor(majority_y)
            labels = batch_majority_y.squeeze().long()
            batch_majority_y = F.one_hot(labels, num_classes=category_size).to(torch.float64)


            # 梯度清零
            optimizer_classifier_x_y.zero_grad()

            # 前向传播
            y_classifier, x_reconstr = model(batch_x)

            # 计算损失
            loss_classifier_x_y = criterion_cross_entropy(y_classifier, batch_majority_y)  # 分类交叉熵损失

            # 反向传播
            loss_classifier_x_y.backward()
            optimizer_classifier_x_y.step()

    # print("Initialize x -> y ...")
    # 继续训练整体网络
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    epochs = 100
    for epoch in range(epochs):
        total_hit = 0

        for i in range(n_samples // batch_size):
            batch_x = torch.FloatTensor(user_labels)
            batch_majority_y = torch.FloatTensor(majority_y)
            labels = batch_majority_y.squeeze().long()
            batch_majority_y = F.one_hot(labels, num_classes=category_size).to(torch.float64)

            # 获取y_prob从分类器x -> y
            with torch.no_grad():
                y_classifier, x_reconstr = model(batch_x)

                # 梯度清零
            optimizer.zero_grad()

            # 计算整体损失
            loss_cross = criterion_cross_entropy(batch_x, x_reconstr)
            loss_y_kl = nn.KLDivLoss(reduction='batchmean')(torch.log(y_classifier + 1e-10), batch_majority_y)
            loss_w_classifier_l1 = torch.sum(torch.abs(model.encoder.weight))
            loss =loss_cross \
            + 0.0001 * loss_y_kl \
            + 0.005 / (source_num * category_size * category_size) * loss_w_classifier_l1

            # 反向传播
            loss.backward()
            optimizer.step()

        y_classifier, x_reconstr = model(batch_x)
        _, predicted = torch.max(y_classifier.data, 1)
        predicted = predicted.to('cpu').detach().numpy()
        return predicted