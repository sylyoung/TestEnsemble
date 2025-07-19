# -*- coding: utf-8 -*-
# @Time    : 2025/7/18
# @Author  : Siyang Li
# @File    : ensemble.py
# black-box test-time ensemble using multiple base classifiers' predictions
import time, sys, argparse, random, os
import numpy as np
import torch
import pandas as pd

from crowdkit.aggregation import DawidSkene, Wawa, MMSR, MACE, GLAD
from sklearn.metrics import balanced_accuracy_score, accuracy_score

from algs.PM import PM
from algs.ZC import ZC
from algs.LA_twopass import one_pass, two_pass
from algs.LAA import LAA_net
from algs.EBCC import ebcc_vb
import warnings
warnings.simplefilter("ignore", UserWarning)


def write_to_file(data, path):
    data = np.array(data)
    data = data.reshape(-1,)
    f = open(path, 'w')
    np.savetxt(f, data.astype(int))
    f.close()


def write_to_file_online(data, data_offline, label, path):
    data = np.array(data).reshape(-1,)
    f = open(path + 'online.csv', 'a')
    np.savetxt(f, data.astype(int))
    f.close()

    data_offline = np.array(data_offline).reshape(-1,)
    f = open(path + 'offline.csv', 'a')
    np.savetxt(f, data_offline.astype(int))
    f.close()

    label = np.array(label).reshape(-1,)
    f = open(path + 'label.csv', 'a')
    np.savetxt(f, label.astype(int))
    f.close()


def SML(preds, labels, args, write=True):
    # SML for binary classification
    start_time = time.time()

    # {-1, 1}
    pred = np.ones((preds.shape[0], preds.shape[1])) * -1
    for j in range(len(preds)):
        for n in range(len(preds[1])):
            if preds[j, n] == 1:
                pred[j, n] = 1
            else:
                pred[j, n] = -1

    mu = np.mean(pred, axis=1)
    deviations = pred - mu[:, np.newaxis]
    # Calculate the covariance matrix
    Q = np.dot(deviations, deviations.T) / (pred.shape[1] - 1)
    # Principal eigenvector
    v = np.linalg.eig(Q)[1][:, 0]

    if np.any(v < 0):
        v = np.abs(v)

    predictions = np.einsum('a,ab->b', v, pred)  # use v, not weights
    predict = np.where(predictions >= 0, 1, 0)
    score = np.round(balanced_accuracy_score(labels, predict), 5)
    acc_score = np.round(accuracy_score(labels, predict), 5)
    end_time = time.time()
    print('ensemble weights:')
    print(v)
    print('SML: {:.2f}'.format(score * 100))
    if not os.path.exists('./results/' + args.dataset_name + '/sml.csv'):
        os.mkdir('./results/' + args.dataset_name)
    if write:
        write_to_file(path='./results/' + args.dataset_name + '/sml.csv', data=predict)
    return score * 100, acc_score * 100, v, predict


def SML_OVR(preds, labels, n_classes, args, write=True, return_weights_all=False):
    # SML-onevsrest, as illustrated in our paper "Black-Box Test-Time Ensemble"
    start_time = time.perf_counter()

    preds_one_hot = []
    for i in range(len(preds)):
        max_indices = preds[i]
        encoded_arr = np.zeros((preds.shape[1], n_classes), dtype=int)
        encoded_arr[np.arange(preds.shape[1]), max_indices] = 1
        preds_one_hot.append(encoded_arr)
    preds_one_hot = np.stack(preds_one_hot)

    preds = preds_one_hot

    weights_all = []
    class_num = preds.shape[-1]
    for i in range(class_num):
        # {-1, 1}
        pred = np.ones((preds.shape[0], preds.shape[1])) * -1
        argmax_inds = np.argmax(preds, axis=-1)
        for j in range(len(preds)):
            for n in range(len(preds[1])):
                if argmax_inds[j, n] == i:
                    pred[j, n] = 1
                else:
                    pred[j, n] = -1
        mu = np.mean(pred, axis=1)
        deviations = pred - mu[:, np.newaxis]
        # Calculate the covariance matrix
        Q = np.dot(deviations, deviations.T) / (pred.shape[1] - 1)
        # Principal eigenvector
        v = np.linalg.eig(Q)[1][:, 0]
        if v[0] <= 0:
            v = -v
        weights = v / np.sum(v)  # ensemble weights
        weights_all.append(weights)

    weights_final = np.sum(np.array(weights_all), axis=0)

    predictions = np.einsum('a,abc->bc', weights_final, preds_one_hot)
    predict = np.argmax(predictions, axis=1)
    score = np.round(balanced_accuracy_score(labels, predict), 5)
    acc_score = np.round(accuracy_score(labels, predict), 5)
    end_time = time.perf_counter()
    # print('ensemble weights:')
    # print(weights_final / class_num)
    # print('SML-OVR time', end_time - start_time)
    # print('SML-OVR: {:.2f}'.format(score * 100))
    if not os.path.isdir('./results/' + args.dataset_name):
        os.mkdir('./results/' + args.dataset_name)
    if write:
        write_to_file(path='./results/' + args.dataset_name + '/smlovr.csv', data=predict)
    if return_weights_all:
        return score * 100, acc_score * 100, weights_final / class_num, predict, weights_all
    return score * 100, acc_score * 100, weights_final / class_num, predict


def SML_OVR_online_vs_offline(preds_all, labels, class_num, args):
    preds_one_hot = []
    for i in range(len(preds_all)):
        max_indices = preds_all[i]
        encoded_arr = np.zeros((preds_all.shape[1], class_num), dtype=int)
        encoded_arr[np.arange(preds_all.shape[1]), max_indices] = 1
        preds_one_hot.append(encoded_arr)
    preds_all = np.stack(preds_one_hot)

    indices = np.arange(len(labels))
    random.shuffle(indices)
    preds_all = preds_all[:, indices, :]
    labels = labels[indices]

    # offline
    weights_all = []
    for i in range(class_num):
        # {-1, 1}
        pred = np.ones_like(preds_all) * -1
        for j in range(len(preds_all)):
            pred[j, np.arange(len(preds_all[j])), preds_all[j].argmax(1)] = 1
        pred = pred[:, :, i]

        mu = np.mean(pred, axis=1)
        deviations = pred - mu[:, np.newaxis]
        # Calculate the covariance matrix
        Q = np.dot(deviations, deviations.T) / (pred.shape[1] - 1)
        # Principal eigenvector
        v = np.linalg.eig(Q)[1][:, 0]
        if np.any(v < 0):
            v = np.abs(v)
        weights = v / np.sum(v)  # ensemble weights
        weights_all.append(weights)

    weights_final = np.sum(np.array(weights_all), axis=0)
    predictions = np.einsum('a,abc->bc', weights_final, preds_all)
    predict_offline = np.argmax(predictions, axis=1)

    start_time = time.perf_counter()
    # online
    # Assume preds_all is a 3D array of shape (num_models, total_samples, num_classes)
    class_num = preds_all.shape[2]
    num_models = preds_all.shape[0]
    total_samples = preds_all.shape[1]

    # Precompute class indicators: (class, model, sample) -> 1 if model predicted class for sample, else -1
    class_indicators = -np.ones((class_num, num_models, total_samples), dtype=np.float64)
    for j in range(num_models):
        argmax_classes = preds_all[j].argmax(axis=1)  # Get predicted class for each sample
        for k in range(total_samples):
            i = argmax_classes[k]
            class_indicators[i, j, k] = 1.0

    # Initialize incremental sums for each class
    sum_rows = {i: np.zeros(num_models) for i in range(class_num)}
    sum_products = {i: np.zeros((num_models, num_models)) for i in range(class_num)}

    cnt = 0
    start_ind = len(preds_all)
    predict_all = []
    for sample_num in range(1, total_samples + 1):
        k = sample_num - 1  # Current sample index

        # Update sums for each class with the new sample
        for i in range(class_num):
            new_col = class_indicators[i, :, k]
            sum_rows[i] += new_col
            sum_products[i] += np.outer(new_col, new_col)

        if sample_num <= start_ind:
            preds = np.copy(preds_all[:, :sample_num, :])
            pred = np.average(preds, axis=0)
            predict_online = np.argmax(pred, axis=1)[-1]
            predict_all.append(predict_online)
        else:
            weights_all = []
            for i in range(class_num):
                # Compute covariance matrix Q
                n = sample_num
                Q = (sum_products[i] - np.outer(sum_rows[i], sum_rows[i]) / n) / (n - 1)

                # Principal eigenvector
                eigenvalues, eigenvectors = np.linalg.eig(Q)
                v = eigenvectors[:, np.argmax(eigenvalues)]
                if np.any(v < 0):
                    v = np.abs(v)
                weights = v / np.sum(v)
                weights_all.append(weights)

            # Combine weights across classes
            weights_final = np.sum(weights_all, axis=0)

            # Compute prediction for the latest sample
            last_sample_preds = preds_all[:, k, :]  # Shape (num_models, num_classes)
            weighted_preds = np.einsum('a,ac->c', weights_final, last_sample_preds)
            predict_online = np.argmax(weighted_preds)
            predict_all.append(predict_online)

        if predict_offline[sample_num - 1] == predict_online:
            cnt += 1
        else:
            cnt += 0
    end_time = time.perf_counter()

    bca_score = balanced_accuracy_score(labels, predict_all) * 100
    acc_score = accuracy_score(labels, predict_all) * 100

    write_to_file_online(path='./results/' + args.dataset_name + '/', data=predict_all, data_offline=predict_offline,
                         label=labels)

    return bca_score, acc_score


def pred_voting_hard(preds, labels, n_classes, args, return_pred=False, write=True):
    # voting
    n_classifier, n_samples = preds.shape
    votes_mat = np.zeros((n_classes, n_samples))
    for i in range(n_classifier):
        for j in range(n_samples):
            class_id = preds[i, j]
            votes_mat[class_id, j] += 1
    votes_pred = []
    for i in range(n_samples):
        pred = np.random.choice(np.flatnonzero(votes_mat[:, i] == votes_mat[:, i].max()))
        votes_pred.append(pred)
    votes_pred = np.array(votes_pred)
    score = np.round(balanced_accuracy_score(labels, votes_pred), 5)
    acc_score = np.round(accuracy_score(labels, votes_pred), 5)
    # print('Voting BCA:{:.2f} ACC:{:.2f}'.format(score * 100, acc_score * 100))
    if not os.path.isdir('./results/' + args.dataset_name):
        os.mkdir('./results/' + args.dataset_name)
    if write:
        write_to_file(path='./results/' + args.dataset_name + '/voting.csv', data=votes_pred)
    if return_pred:
        return score * 100, acc_score * 100, votes_pred
    return score * 100, acc_score * 100


def pred_single(preds, labels, args):
    # every single
    print('####################BCA####################')
    scores_arr = []
    for i in range(len(preds)):
        predict = preds[i]
        score = np.round(balanced_accuracy_score(labels, predict), 5)
        print('{}: {:.2f}'.format(args.model_names[i], score * 100))
        scores_arr.append(score * 100)

    print('####################ACC####################')
    acc_scores_arr = []
    for i in range(len(preds)):
        predict = preds[i]
        acc_score = np.round(accuracy_score(labels, predict), 5)
        print('{}: {:.2f}'.format(args.model_names[i], acc_score * 100))
        acc_scores_arr.append(acc_score * 100)
    return scores_arr, acc_scores_arr


def seed_everything(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SML-OVR experiment')
    parser.add_argument('--seed', type=int, default=0, help='seed for everything')
    parser.add_argument('--dataset_name', type=str, default='MOSI', help='dataset name')
    args = parser.parse_args()
    seed_everything(args)

    print(args.dataset_name)

    model_names = ["neural-chat-7b-v3-1", "openchat_3.5", "zephyr-7b-beta", "phi-2", "gemma-7b-it",
                   "Llama-2-7b-chat-hf", "mpt-7b", "jais-13b-chat", "TinyLlama-1.1B-Chat-v1.0", "Mistral-7B-v0.1"]
    args.model_names = model_names

    preds = []
    for i in range(0, len(model_names), 1):
        preds.append(np.loadtxt('./results/' + args.dataset_name + '/' + model_names[i] + '.csv').astype(int))
    preds = np.stack(preds)
    print('preds.shape', preds.shape)
    labels = np.loadtxt('./data/' + args.dataset_name + '/y.txt').astype(int)

    if args.dataset_name == 'WOS':
        n_classes = 7
    else:
        n_classes = 3

    print('# BCA / ACC scores')

    print('Single')
    true_scores, true_acc_scores = pred_single(preds, labels, args)
    print('#' * 30)

    voting_bca, voting_acc, voting_pred = pred_voting_hard(preds, labels, n_classes, args, return_pred=True)
    print('Voting')
    print(np.round(voting_bca, 2), np.round(voting_acc, 2))

    smlovr_bca, smlovr_acc, weights_sml, sml_pred = SML_OVR(preds, labels, n_classes, args, write=True)  # offline
    print('SML-OVR')
    print(np.round(smlovr_bca, 2), np.round(smlovr_acc, 2))

    online_bca, online_acc = SML_OVR_online_vs_offline(preds, labels, n_classes, args)  # online
    print('SML-OVR online')
    print(np.round(smlovr_bca, 2), np.round(smlovr_acc, 2))

    num_samples = len(labels)
    num_models = len(preds)

    a_1 = np.repeat(np.arange(num_samples) + 1, num_models)
    a_2 = np.array((np.arange(num_models) + 1).tolist() * num_samples)
    a_3 = np.transpose(preds, (1, 0)).reshape(-1,) + 1
    A = np.stack([a_1, a_2, a_3])
    A = np.transpose(A, (1, 0))
    b_1 = np.arange(num_samples) + 1
    b_2 = labels.reshape(-1,) + 1
    B = np.stack([b_1, b_2])
    B = np.transpose(B, (1, 0))

    pred_df = pd.DataFrame(A, columns=['task', 'worker', 'label'])

    ds_pred = DawidSkene(n_iter=10).fit_predict(pred_df)
    ds_pred = ds_pred.to_numpy()
    ds_pred -= 1
    write_to_file(path='./results/' + args.dataset_name + '/dawidskene.csv', data=ds_pred)
    print('Dawid-Skene')
    print(np.round(balanced_accuracy_score(labels, ds_pred) * 100, 2), np.round(accuracy_score(labels, ds_pred) * 100, 2))

    wawa_pred = Wawa().fit_predict(pred_df)
    wawa_pred = wawa_pred.to_numpy()
    wawa_pred -= 1
    write_to_file(path='./results/' + args.dataset_name + '/wawa.csv', data=wawa_pred)
    print('WAwA')
    print(np.round(balanced_accuracy_score(labels, wawa_pred) * 100, 2), np.round(accuracy_score(labels, wawa_pred) * 100, 2))

    mmsr_pred = MMSR().fit_predict(pred_df)
    mmsr_pred = mmsr_pred.to_numpy()
    mmsr_pred -= 1
    write_to_file(path='./results/' + args.dataset_name + '/m-msr.csv', data=mmsr_pred)
    print('M-MSR')
    print(np.round(balanced_accuracy_score(labels, mmsr_pred) * 100, 2), np.round(accuracy_score(labels, mmsr_pred) * 100, 2))

    mace_pred = MACE().fit_predict(pred_df)
    mace_pred = mace_pred.to_numpy()
    mace_pred -= 1
    write_to_file(path='./results/' + args.dataset_name + '/mace.csv', data=mace_pred)
    print('MACE')
    print(np.round(balanced_accuracy_score(labels, mace_pred) * 100, 2), np.round(accuracy_score(labels, mace_pred) * 100, 2))

    glad_pred = GLAD().fit_predict(pred_df)
    glad_pred = glad_pred.to_numpy()
    glad_pred -= 1
    write_to_file(path='./results/' + args.dataset_name + '/glad.csv', data=glad_pred)
    print('GLAD')
    print(np.round(balanced_accuracy_score(labels, glad_pred) * 100, 2), np.round(accuracy_score(labels, glad_pred) * 100, 2))

    zc_pred = ZC(preds, n_classes, iterr=1).Run()
    write_to_file(path='./results/' + args.dataset_name + '/zencrowd.csv', data=zc_pred)
    print('ZenCrowd')
    print(np.round(balanced_accuracy_score(labels, zc_pred) * 100, 2), np.round(accuracy_score(labels, zc_pred) * 100, 2))

    pm_pred, weight_pm = PM(preds, n_classes, 1)
    write_to_file(path='./results/' + args.dataset_name + '/pm.csv', data=zc_pred)
    print('PM')
    print(np.round(balanced_accuracy_score(labels, pm_pred) * 100, 2), np.round(accuracy_score(labels, pm_pred) * 100, 2))

    num_workers = preds.shape[0]
    num_tasks = preds.shape[1]
    label_set = set(range(n_classes))
    workers = list(range(num_workers))
    e2wl = {}
    for t in range(num_tasks):
        e2wl[t] = [(w, preds[w][t]) for w in workers]
    w2el = {}
    for w in workers:
        w2el[w] = [(t, preds[w][t]) for t in range(num_tasks)]
    one_pass_truths, a = one_pass(e2wl, w2el, label_set, alpha=2, beta=2)
    la_pred = two_pass(e2wl, a, label_set)
    write_to_file(path='./results/' + args.dataset_name + '/la.csv', data=la_pred)
    print('LA')
    print(np.round(balanced_accuracy_score(labels, la_pred) * 100, 2), np.round(accuracy_score(labels, la_pred) * 100, 2))

    preds_one_hot_LAA = []
    for i in range(len(preds)):
        max_indices = preds[i]
        encoded_arr = np.zeros((preds.shape[1], n_classes), dtype=int)
        encoded_arr[np.arange(preds.shape[1]), max_indices] = 1
        preds_one_hot_LAA.append(encoded_arr)
    preds_one_hot_LAA = np.concatenate(preds_one_hot_LAA, axis=1)
    laa_pred = LAA_net(preds, preds_one_hot_LAA, n_classes, voting_pred)
    write_to_file(path='./results/' + args.dataset_name + '/laa.csv', data=laa_pred)
    print('LAA')
    print(np.round(balanced_accuracy_score(labels, laa_pred) * 100, 2), np.round(accuracy_score(labels, laa_pred) * 100, 2))

    ebcc_pred = ebcc_vb(preds, n_classes, num_groups=10, empirical_prior=True, max_iter=10)
    ebcc_pred = np.argmax(ebcc_pred, axis=1)
    write_to_file(path='./results/' + args.dataset_name + '/ebcc.csv', data=ebcc_pred)
    print('EBCC')
    print(np.round(balanced_accuracy_score(labels, ebcc_pred) * 100, 2), np.round(accuracy_score(labels, ebcc_pred) * 100, 2))

    input('Main results done.\nPress enter for ranking and pruning experiment.')

    # ensemble pruning
    # iterative removal of worst classifier
    minimum_ind = None
    curr_model_names = model_names.copy()
    curr_inds = np.arange(len(model_names), dtype=int)
    for i in range(len(curr_model_names) - 3):  # prune till three classifiers
        print('#' * 50)
        print('#' * 50)
        scores = [0, 0]
        if minimum_ind is None:
            inds = np.argsort(weights_sml)
            print('delete', inds[0])
            curr_model_names = np.delete(curr_model_names, inds[0])
            curr_inds = np.delete(curr_inds, inds[0])
        else:
            print('delete', minimum_ind)
            curr_model_names = np.delete(curr_model_names, minimum_ind)
            curr_inds = np.delete(curr_inds, minimum_ind)

        print('number of models:', len(curr_model_names))
        print(curr_model_names)
        print(curr_inds)
        args.model_names = curr_model_names

        preds = []
        for j in range(0, len(curr_model_names), 1):
            preds.append(np.loadtxt('./results/' + args.dataset_name + '/' + curr_model_names[j] + '.csv').astype(int))
        preds = np.stack(preds)
        print('preds.shape', preds.shape)

        scores[0] = np.round(pred_voting_hard(preds, labels, n_classes, args, write=False), 2)
        score, _, weights_sml, _ = SML_OVR(preds, labels, n_classes, args, write=False)  # offline
        scores[1] = np.round(score, 2)

        cnt_skip = 0
        print('[', end="")
        for p in range(len(model_names)):
            if p in curr_inds:
                if p == len(model_names) - 1:
                    print(str(weights_sml[p - cnt_skip]) + ']')
                else:
                    print(weights_sml[p - cnt_skip], end=",")
            else:
                if p == len(model_names) - 1:
                    print(str(0) + ']')
                else:
                    print(0, end=",")
                cnt_skip += 1
        minimum_ind = np.argsort(weights_sml)[0]

        print('Voting, SML-OVR')
        print(scores)



