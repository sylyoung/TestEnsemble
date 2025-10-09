# -*- coding: utf-8 -*-
# @Time    : 2025/8/15
# @Author  : Chenhao Liu and Siyang Li
# @File    : StackingNet_classification.py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import sys


# Stacking network for classification
# Take one-hot predictions as inputs, which are converted from class predictions
# Optimize the aggregation weights as trainable parameters
class Net(nn.Module):
    def __init__(self, in_features, n_class, weight_init, use_dropout=False, dropout_rate=0.1, use_softmax=False, use_clamping=False):
        super(Net, self).__init__()
        # Initialize M weights, one for each base classifier
        self.workers = in_features // n_class
        self.classes = n_class

        if use_softmax:
            self.use_softmax = True
            weight_init = backconvert_weight_init(weight_init)
        else:
            self.use_softmax = False

        self.weights = nn.Parameter(weight_init)

        if use_dropout:
            self.use_dropout = True
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.use_dropout = False

        self.use_clamping = use_clamping

    def forward(self, x):
        # x should be of shape (batch_size, M, K)
        # Broadcast weights to match the M dimension of x
        # Weights shape after unsqueeze: (1, M, 1) for broadcasting
        batch_size = x.shape[0]
        x = x.reshape((batch_size, self.workers, self.classes))
        if self.use_softmax:
            weights = F.softmax(self.weights, dim=0).unsqueeze(0)  # Adjust for batch dimension
        else:
            weights = self.weights.unsqueeze(0)  # Adjust for batch dimension

        # clamping to be nonnegative
        if self.use_clamping:
            weights = torch.clamp(self.weights, min=0.0)

        # Perform weighted sum across the M dimension
        weighted_sum = torch.mul(x, weights)  # Element-wise multiplication
        if self.use_dropout:
            weighted_sum = self.dropout(weighted_sum)

        result = weighted_sum.sum(dim=1)  # Summing over the M dimension
        return result


# applies a reverse softmax to weights initialization, such that they keep their original values after softmax
def backconvert_weight_init(weight_init: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (weight_init.clamp_min(eps) / weight_init.clamp_min(eps).sum()).log()


def train(net, num_epochs, optimizer, train_loader, target, loss, args):
    # Using Adam optimization algorithm
    for _ in range(num_epochs):
        for data in train_loader:
            X, y = data
            y = y.long()
            outputs_source = net(X)
            outputs_target = net(target).to(torch.float32)
            outputs_source = outputs_source.to(torch.float32)
            task_loss = loss(outputs_source, y)

            if args.loss == 'PM':
                weights = []
                batch_size = target.shape[0]
                target = target.reshape((batch_size, args.workers, args.classes))

                max_indices = torch.argmax(outputs_target, dim=1)

                # Construct one-hot encoding
                outputs_target = torch.zeros_like(outputs_target)
                outputs_target.scatter_(1, max_indices.unsqueeze(1), 1)  # Setting 1 at the maximum position
                for i in range(args.workers):
                    dif = (target[:, i, :] != outputs_target).any(dim=1).float().sum()
                    weight = dif * net.weights[i]
                    weights.append(weight)
                pm = args.unsupervised_weight * sum(weights)
                if args.sigma:
                    weights_sum = l1_regularization(net, 1)
                    sigma = (1 - weights_sum) ** 2  # Regularization
                    loss_combined = task_loss + pm + args.regularization_weight * sigma
                else:
                    loss_combined = task_loss + pm
            elif args.loss == 'theta':
                weights_sum = net.weights.sum()
                # reg_loss = 100*(weights_sum - 1) ** 2
                std = torch.std(outputs_target)
                theta = 10 * (std - torch.std(outputs_source)) ** 2
                loss_combined = task_loss + theta
            else:
                loss_combined = task_loss

            optimizer.zero_grad()
            loss_combined.backward()
            optimizer.step()
            if args.net == '10->1':
                with torch.no_grad():
                    for param in net.parameters():
                        param.data.clamp_(min=0)  # Make sure that the weight parameter is not less than 0


def train_unsupervised(net, num_epochs, optimizer, target, args):
    # Using Adam optimization algorithm
    for _ in range(num_epochs):
        outputs_target = net(target).to(torch.float32)
        if args.loss == 'PM':
            weights = []
            batch_size = target.shape[0]
            target = target.reshape((batch_size, args.workers, args.classes))
            max_indices = torch.argmax(outputs_target, dim=1)

            # Construct one-hot encoding
            outputs_target = torch.zeros_like(outputs_target)
            outputs_target.scatter_(1, max_indices.unsqueeze(1), 1)  # Padding 1 at the maximum position

            # Indicator loss: Minimize the difference between the model prediction and the ensemble prediction for each large model
            for i in range(args.workers):
                dif = (target[:, i, :] != outputs_target).any(dim=1).float().sum()
                weight = dif * net.weights[i]
                weights.append(weight)

            pm = args.unsupervised_weight * sum(weights)
            if args.sigma:
                weights_sum = l1_regularization(net, 1)
                # weights_sum = elementwise_exp_norm(net)
                sigma = (1 - weights_sum) ** 2  # Regularization
                loss_combined = pm + args.regularization_weight * sigma
            else:
                loss_combined = pm

            optimizer.zero_grad()
            loss_combined.backward()
            optimizer.step()
            if args.net == '10->1':
                with torch.no_grad():
                    for param in net.parameters():
                        param.data.clamp_(min=0)  # Make sure that the weight parameter is not less than 0
        else:
            print('untrainable without an loss objective!')
            sys.exit(0)


def l1_regularization(model, lambda_l1):
    l1_norm = torch.norm(model.weights, 1)
    return lambda_l1 * l1_norm


def Stacking_Classification(args, n_classes, preds_test, weight_init=None, preds_golden=None, labels_golden=None, return_model=False):
    # Categories can be unbalanced
    if args.golden_num != 0:
        unique_classes, counts = np.unique(labels_golden, return_counts=True)
        total_count = len(labels_golden)
        batch_size = total_count
        ratios = counts / total_count
        weight = np.zeros(n_classes)
        for class_label, ratio in zip(unique_classes, ratios):
            weight[class_label] = 1 / ratio
        class_weights = torch.tensor(weight, dtype=torch.float32, device=args.device)
        loss = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss = nn.CrossEntropyLoss()
        batch_size = 0

    # Convert the preds to one-hot encoding
    preds_one_hot_test = []
    for i in range(len(preds_test)):
        max_indices = preds_test[i]
        encoded_arr = np.zeros((preds_test.shape[1], n_classes), dtype=int)
        encoded_arr[np.arange(preds_test.shape[1]), max_indices] = 1
        preds_one_hot_test.append(encoded_arr)
    preds_one_hot_test = np.concatenate(preds_one_hot_test, axis=1)

    if preds_golden is not None:
        preds_one_hot_golden = []
        for i in range(len(preds_golden)):
            max_indices = preds_golden[i]
            encoded_arr = np.zeros((preds_golden.shape[1], n_classes), dtype=int)
            encoded_arr[np.arange(preds_golden.shape[1]), max_indices] = 1
            preds_one_hot_golden.append(encoded_arr)
        preds_one_hot_golden = np.concatenate(preds_one_hot_golden, axis=1)

    in_features = preds_one_hot_test.shape[1]
    weight_init = torch.tensor(weight_init).reshape(in_features // n_classes, 1)

    # print('weight_init': np.array2string(weight_init.numpy(), precision=4, suppress_small=True))

    net = Net(in_features, n_classes, weight_init, args.use_dropout, args.dropout_rate, args.regularization_relu)

    net.to(args.device)

    preds_one_hot_test_tensor = torch.tensor(preds_one_hot_test, dtype=torch.float32, device=args.device)

    if args.golden_num != 0:
        golden_task = torch.tensor(preds_one_hot_golden, dtype=torch.float32, device=args.device)  # Golden data
        golden_answer = torch.tensor(labels_golden, device=args.device)
        num_epochs, lr, weight_decay = args.epoch, args.lr, 0
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        train_dataset = TensorDataset(golden_task, golden_answer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)

        train(net, num_epochs, optimizer, train_loader, preds_one_hot_test_tensor, loss, args)

    if args.golden_num == 0:
        num_epochs, lr, weight_decay = args.epoch, args.lr, 0
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        train_unsupervised(net, num_epochs, optimizer, preds_one_hot_test_tensor, args)

    net.eval()
    print('Trained StackingNet Weights:')
    for name, param in net.named_parameters():
        print(f"{name}:\n{param.data.cpu().numpy()}")

    outputs = net(preds_one_hot_test_tensor)

    _, predicted = torch.max(outputs.data, dim=1)
    predicted = predicted.to('cpu').detach().numpy()

    if return_model:
        return predicted, net
    return predicted
