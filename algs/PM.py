# -*- coding: utf-8 -*-
# @Time    : 2025/7/18
# @Author  : Chenhao Liu
# @File    : PM.py
# Crowdsourcing for multiple-choice question answering
# and Resolving conflicts in heterogeneous data by truth discovery and source reliability estimation
import math,random
import numpy as np
from sklearn.metrics import balanced_accuracy_score

def PM(preds, n_classes, iterr = 3):
    # Initial voting matrix construction
    n_classifier, n_samples = preds.shape
    votes_mat = np.zeros((n_classes, n_samples))
    for i in range(n_classifier):
        for j in range(n_samples):
            class_id = preds[i, j]
            votes_mat[class_id, j] += 1

    # Random selection in case of ties
    votes_pred = []
    for i in range(n_samples):
        pred = np.random.choice(np.flatnonzero(votes_mat[:, i] == votes_mat[:, i].max()))
        votes_pred.append(pred)
    votes_pred = np.array(votes_pred)

    # Initialize truth and weights
    truth = votes_pred
    weight = np.zeros(preds.shape[0])
    worker_num = preds.shape[0]
    
    preds_one_hot = []
    for i in range(len(preds)):
        max_indices = preds[i]
        encoded_arr = np.zeros((preds.shape[1], n_classes), dtype=int)
        encoded_arr = encoded_arr - 1
        encoded_arr[np.arange(preds.shape[1]), max_indices] = 1
        preds_one_hot.append(encoded_arr)
    preds_one_hot = np.stack(preds_one_hot)

    weight_max = 0.0

    # Iterative refinement of weights and truth values
    while iterr > 0:
        # Calculate initial weights based on prediction differences
        for worker in range(worker_num):
            dif = 0.0
            dif = np.sum(preds[worker,:]!=truth)
            if dif == 0.0:
                dif = 0.00000001

            weight[worker] = dif
            if weight[worker] > weight_max:
                weight_max = weight[worker]

        # Normalize weights
        for worker in range(worker_num):
            weight[worker] = weight[worker] / weight_max

        for worker in range(worker_num):
            weight[worker] = - math.log(weight[worker] + 0.0000001) + 0.0000001

        # Recompute truth using weighted combination of predictions
        predictions = np.einsum('a,abc->bc', weight, preds_one_hot)
        truth = np.argmax(predictions, axis=1)
        iterr -= 1

    return truth, weight