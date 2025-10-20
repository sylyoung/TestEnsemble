import math
import numpy as np


def PM(preds, n_classes, voted_pred, iterr = 3):
    truth = voted_pred
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

    while iterr > 0:
        for worker in range(worker_num):
            dif = 0.0
            dif = np.sum(preds[worker,:]!=truth)
            if dif == 0.0:
                dif = 0.00000001

            weight[worker] = dif
            if weight[worker] > weight_max:
                weight_max = weight[worker]

        for worker in range(worker_num):
            weight[worker] = weight[worker] / weight_max

        for worker in range(worker_num):
            weight[worker] = - math.log(weight[worker] + 0.0000001) + 0.0000001

        predictions = np.einsum('a,abc->bc', weight, preds_one_hot)
        truth = np.argmax(predictions, axis=1)
        iterr -= 1

    return truth, weight