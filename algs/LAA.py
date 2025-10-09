# -*- coding: utf-8 -*-
# @Time    : 2025/7/18
# @Author  : Chenhao Liu
# @File    : LAA.py
# Aggregating crowd wisdoms with label-aware autoencoders
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

# Encoding and decoding neural networks for LAA
# Construct Classifier
class Classifier(nn.Module):
    """x -> y """
    def __init__(self, input_size, category_size):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(input_size, category_size))
        self.biases = nn.Parameter(torch.empty(category_size))
        # Initialize weights and biases
        nn.init.trunc_normal_(self.weights, mean=0.0, std=0.01)
        nn.init.zeros_(self.biases)

    def forward(self, x):
        return F.softmax(x @ self.weights + self.biases, dim=1)

# Construct Decoder
class Decoder(nn.Module):

    def __init__(self, category_size, input_size, source_wise_template):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(category_size, input_size))
        self.biases = nn.Parameter(torch.empty(input_size))
        self.template = source_wise_template
        # Initialize weights and biases
        nn.init.trunc_normal_(self.weights, mean=0.0, std=0.01)
        nn.init.zeros_(self.biases)

    def forward(self, y):
        x_reconstr_tmp = y @ self.weights + self.biases
        exp_x = torch.exp(x_reconstr_tmp)
        return exp_x / (exp_x @ self.template)


def LAA_net(preds, preds_one_hot, num_labels, voted_preds):
    user_labels = preds_one_hot
    majority_y = voted_preds
    category_size = num_labels
    source_num = preds.shape[0]
    n_samples = preds.shape[1]
    input_size = source_num * category_size
    batch_size = n_samples

    source_wise_template = torch.zeros((input_size, input_size))
    for i in range(source_num):
        source_wise_template[i * category_size:(i + 1) * category_size,
        i * category_size:(i + 1) * category_size] = 1

    classifier = Classifier(input_size, category_size)
    decoder = Decoder(category_size, input_size, source_wise_template)
    criterion_cross_entropy = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(
        list(classifier.parameters()) + list(decoder.parameters()),
        lr=0.005
    )

    # train classifier first
    epochs = 50
    for epoch in range(epochs):
        total_hit = 0

        for i in range(n_samples // batch_size):
            # Get all data
            batch_x = torch.FloatTensor(user_labels)
            batch_majority_y = torch.FloatTensor(majority_y)
            labels = batch_majority_y.squeeze().long()
            batch_majority_y = F.one_hot(labels, num_classes=category_size).to(torch.float32)

            optimizer.zero_grad()

            y_classifier = classifier(batch_x)

            # loss classifier
            loss_classifier_x_y = criterion_cross_entropy(y_classifier, batch_majority_y)

            loss_classifier_x_y.backward()
            optimizer.step()

    # train decoder
    epochs = 100
    for epoch in range(epochs):
        total_hit = 0

        for i in range(n_samples // batch_size):
            batch_x = torch.FloatTensor(user_labels)
            batch_majority_y = torch.FloatTensor(majority_y)
            labels = batch_majority_y.squeeze().long()
            batch_majority_y = F.one_hot(labels, num_classes=category_size).to(torch.float64)

            optimizer.zero_grad()

            y_classifier = classifier(batch_x)
            x_reconstr = decoder(y_classifier)

            loss_cross = torch.mean(torch.sum(-batch_x * torch.log(x_reconstr + 1e-10), dim=1))
            loss_y_kl = nn.KLDivLoss(reduction='batchmean')(torch.log(y_classifier + 1e-10), batch_majority_y)
            loss_w_classifier_l1 = torch.sum(torch.abs(classifier.weights))
            loss = loss_cross \
                   + 0.0001 * loss_y_kl \
                   + 0.005 / (source_num * category_size * category_size) * loss_w_classifier_l1


            loss.backward()
            optimizer.step()

    with torch.no_grad():
        y_classifier = classifier(batch_x)
        _, predicted = torch.max(y_classifier.data, 1)
        predicted = predicted.to('cpu').detach().numpy()
        return predicted
