import time, sys, argparse, random, os
import numpy as np
import torch
import scipy

from sklearn.metrics import balanced_accuracy_score
from torch.nn.functional import softmax


def write_to_file(data, path):
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


def SML_onevsrest_hard(preds, labels, n_classes, args, write=True):
    # SML-OVR, as proposed in our paper "Black-Box Test-Time Ensemble"
    '''
    :param preds: numpy array (M, n) of predictions of M classifiers on n test samples
    :param labels: numpy array (n, ) of true labels
    :param n_classes: int of number of classes
    :param args: arguments, see main func
    :param write: boolean, whether write to file to record the class prediction
    :return: bca score, weights of SML-OVR
    '''

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

    # loop over all classes in all ovr splits
    for i in range(class_num):

        # {0, 1}
        pred = np.zeros_like(preds)
        for j in range(len(preds)):
            pred[j, np.arange(len(preds[j])), preds[j].argmax(1)] = 1
        pred = pred[:, :, i]

        Q = pred @ pred.T  # pred is shape (M, n)
        v = np.linalg.eig(Q)[1][:, 0]  # principal eigenvector
        weights = v / np.sum(v)  # normalize
        weights_all.append(weights)  # ensemble weights

    weights_final = np.sum(np.array(weights_all), axis=0)  # average over all classes in all ovr splits

    predictions = np.einsum('a,abc->bc', weights_final, preds_one_hot)
    predict = np.argmax(predictions, axis=1)
    score = np.round(balanced_accuracy_score(labels, predict), 5)
    print('SML-OVR ensemble weights:')
    print(weights_final / class_num)
    print('SML-OVR: {:.2f}'.format(score * 100))
    if write:
        write_to_file(path='./results/' + args.dataset_name + '/smlmulti-hard.csv', data=predict)
    return score * 100, weights_final / class_num


def SML_onevsrest_online_vs_offline(preds_all, labels, class_num, args, interval=100):
    # offline vs. online (test-time) comparison of SML-OVR
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

    # offline results
    weights_all = []
    for i in range(class_num):
        pred = np.zeros_like(preds_all)
        for j in range(len(preds_all)):
            pred[j, np.arange(len(preds_all[j])), preds_all[j].argmax(1)] = 1
        pred = pred[:, :, i]
        Q = pred @ pred.T  # pred is shape (M, n)
        v = np.linalg.eig(Q)[1][:, 0]  # principal eigenvector
        weights = v / np.sum(v)  # normalize
        weights_all.append(weights)  # ensemble weights
    weights_final = np.sum(np.array(weights_all), axis=0)
    predictions = np.einsum('a,abc->bc', weights_final, preds_all)
    predict_offline = np.argmax(predictions, axis=1)

    # online results
    weights_final = None
    start_ind = len(preds_all)
    cnt = 0
    predict_all = []
    for sample_num in range(1, len(preds_all[0]) + 1):
        preds = np.copy(preds_all[:, :sample_num, :])
        if sample_num <= start_ind:
            pred = np.average(preds, axis=0)
            predict_online = np.argmax(pred, axis=1)[-1]
        else:
            if cnt % interval != 0 and weights_final is not None:
                predictions = np.einsum('a,abc->bc', weights_final, preds)
                predict_online = np.argmax(predictions, axis=1)[-1]
            else:
                weights_all = []
                for i in range(class_num):
                    pred = np.zeros_like(preds)
                    for j in range(len(preds)):
                        pred[j, np.arange(len(preds[j])), preds[j].argmax(1)] = 1
                    pred = pred[:, :, i]
                    Q = pred @ pred.T  # pred is shape (M, n)
                    v = np.linalg.eig(Q)[1][:, 0]  # principal eigenvector
                    weights = v / np.sum(v)  # normalize
                    weights_all.append(weights)  # ensemble weights
                    if np.array_equal(weights, np.zeros(len(weights))):
                        # this happens?
                        print('skipping this class', i)
                        continue
                    weights_all.append(weights)
                weights_final = np.sum(np.array(weights_all), axis=0)
                predictions = np.einsum('a,abc->bc', weights_final, preds)
                predict_online = np.argmax(predictions, axis=1)[-1]
        predict_all.append(predict_online)

        # uncomment the following print lines for exact comparison
        if predict_offline[sample_num-1] == predict_online:
            cnt += 1
        else:
            cnt = cnt

    offline_score = np.round(balanced_accuracy_score(labels, predict_offline), 5)

    online_score = np.round(balanced_accuracy_score(labels, predict_all), 5)

    print('offline_score, online score:', offline_score * 100, online_score * 100)
    write_to_file_online(path='./results/' + args.dataset_name + '/', data=predict_all, data_offline=predict_offline, label=labels)

    return online_score * 100


def pred_voting_hard(preds, labels, n_classes, args, write=True):
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
    print('Voting: {:.2f}'.format(score * 100))
    if write:
        write_to_file(path='./results/' + args.dataset_name + '/voting.csv', data=votes_pred)
    return score * 100


def pred_single(preds, labels, args):
    # single classifier in the ensemble
    scores_arr = []
    for i in range(len(preds)):
        predict = preds[i]
        score = np.round(balanced_accuracy_score(labels, predict), 5)
        print('{}: {:.2f}'.format(args.model_names[i], score * 100))
        scores_arr.append(score * 100)
    return scores_arr


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

    scores = [0, 0, 0, 0]

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

    scores[0] = np.round(pred_voting_hard(preds, labels, n_classes, args), 2)
    score, weights_final = SML_onevsrest_hard(preds, labels, n_classes, args, write=False)  # offline
    scores[1] = np.round(score, 2)
    scores[2] = np.round(SML_onevsrest_online_vs_offline(preds, labels, n_classes, args), 2)  # online

    true_scores = pred_single(preds, labels, args)
    preds_one_hot = []
    for i in range(len(preds)):
        max_indices = preds[i]
        encoded_arr = np.zeros((preds.shape[1], n_classes), dtype=int)
        encoded_arr[np.arange(preds.shape[1]), max_indices] = 1
        preds_one_hot.append(encoded_arr)
    preds_one_hot = np.stack(preds_one_hot)
    scores[3] = np.round(balanced_accuracy_score(labels, np.argmax(np.einsum('a,abc->bc', true_scores, preds_one_hot),
                                                                   axis=-1)) * 100, 2)

    print('Single:', true_scores, 'Mean:', np.mean(true_scores))
    print('#' * 30)
    print('Voting, SML-OVR offline, SML-OVR online, True-BCA-Weights Ensemble')
    print(scores)

    input('Main results done.\nPress any key for ranking and pruning experiment.')

    # continuous removal of worst classifier
    minimum_ind = None
    curr_model_names = model_names.copy()
    curr_inds = np.arange(len(model_names), dtype=int)
    for i in range(len(curr_model_names) - 3):  # removal till three classifiers
        print('#' * 50)
        print('#' * 50)
        scores = [0, 0]
        if minimum_ind is None:
            inds = np.argsort(weights_final)
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
        score, weights_final = SML_onevsrest_hard(preds, labels, n_classes, args, write=False)  # offline
        scores[1] = np.round(score, 2)

        cnt_skip = 0
        print('[', end="")
        for p in range(len(model_names)):
            if p in curr_inds:
                if p == len(model_names) - 1:
                    print(str(weights_final[p - cnt_skip]) + ']')
                else:
                    print(weights_final[p - cnt_skip], end=",")
            else:
                if p == len(model_names) - 1:
                    print(str(0) + ']')
                else:
                    print(0, end=",")
                cnt_skip += 1
        minimum_ind = np.argsort(weights_final)[0]

        print('Voting, SML-OVR')
        print(scores)



