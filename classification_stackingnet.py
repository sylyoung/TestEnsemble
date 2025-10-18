# -*- coding: utf-8 -*-
# @Time    : 2025/8/15
# @Author  : ???
# @File    : classification_stackingnet.py
# implementation for our paper "StackingNet: collective inference across independent AI foundation models"
# this file conducts combination methods for classification datasets from HELM benchmark
# datasets/LLMs source: https://crfm.stanford.edu/helm/classic/latest/

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, kendalltau, weightedtau
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, ndcg_score
from crowdkit.aggregation import DawidSkene, Wawa, MMSR, MACE, GLAD, KOS
import sys, argparse, random, os, statistics, json

from algs.PM import PM
from algs.ZC import ZC
from algs.LA_twopass import one_pass, two_pass
from algs.LAA import LAA_net
from algs.EBCC import ebcc_vb
from algs.StackingNet_classification import Stacking_Classification
from sklearn.model_selection import StratifiedShuffleSplit


def ReadLabel(args):
    if args.dataset_name == 'bbq':
        n_classes = 3
        args.dataset_path = 'bbq/bbq:subject=all,method=multiple_choice_joint'
        args.backend_path = ''
        labels = np.loadtxt(
            './logs/helm/bbq/bbq:subject=all,method=multiple_choice_joint,model=ai21_j1-grande-v2-beta_ground_truth.csv').astype(
            int)
    elif args.dataset_name == 'boolq':
        n_classes = 2
        args.dataset_path = 'boolq/boolq:'
        args.backend_path = ',data_augmentation=canonical'
        labels = np.loadtxt(
            './logs/helm/boolq/boolq:model=ai21_j1-jumbo,data_augmentation=canonical_ground_truth.csv').astype(int)
    elif args.dataset_name == 'civil_comments':
        n_classes = 2
        args.dataset_path = ['civil_comments/civil_comments:demographic=LGBTQ,',
                             'civil_comments/civil_comments:demographic=all,',
                             'civil_comments/civil_comments:demographic=black,',
                             'civil_comments/civil_comments:demographic=christian,',
                             'civil_comments/civil_comments:demographic=female,',
                             'civil_comments/civil_comments:demographic=male,',
                             'civil_comments/civil_comments:demographic=muslim,',
                             'civil_comments/civil_comments:demographic=other_religions,',
                             'civil_comments/civil_comments:demographic=white,']
        args.backend_path = ',data_augmentation=canonical'
        labels = [np.loadtxt(
            './logs/helm/civil_comments/civil_comments:demographic=LGBTQ,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
            int),
            np.loadtxt(
                './logs/helm/civil_comments/civil_comments:demographic=all,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
                int),
            np.loadtxt(
                './logs/helm/civil_comments/civil_comments:demographic=black,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
                int),
            np.loadtxt(
                './logs/helm/civil_comments/civil_comments:demographic=christian,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
                int),
            np.loadtxt(
                './logs/helm/civil_comments/civil_comments:demographic=female,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
                int),
            np.loadtxt(
                './logs/helm/civil_comments/civil_comments:demographic=male,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
                int),
            np.loadtxt(
                './logs/helm/civil_comments/civil_comments:demographic=muslim,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
                int),
            np.loadtxt(
                './logs/helm/civil_comments/civil_comments:demographic=other_religions,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
                int),
            np.loadtxt(
                './logs/helm/civil_comments/civil_comments:demographic=white,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
                int), ]
        labels = np.concatenate(labels)
    elif args.dataset_name == 'entity_matching':
        n_classes = 2
        args.dataset_path = ['entity_matching/entity_matching:dataset=Abt_Buy,',
                             'entity_matching/entity_matching:dataset=Beer,',
                             'entity_matching/entity_matching:dataset=Dirty_iTunes_Amazon,']
        args.backend_path = ''
        labels = [np.loadtxt(
            './logs/helm/entity_matching/entity_matching:dataset=Abt_Buy,model=AlephAlpha_luminous-base_ground_truth.csv').astype(
            int),
            np.loadtxt(
                './logs/helm/entity_matching/entity_matching:dataset=Beer,model=AlephAlpha_luminous-base_ground_truth.csv').astype(
                int),
            np.loadtxt(
                './logs/helm/entity_matching/entity_matching:dataset=Dirty_iTunes_Amazon,model=AlephAlpha_luminous-base_ground_truth.csv').astype(
                int)]
        labels = np.concatenate(labels)
    elif args.dataset_name == 'imdb':
        n_classes = 2
        args.dataset_path = 'imdb/imdb:'
        args.backend_path = ',data_augmentation=canonical'
        labels = np.loadtxt(
            './logs/helm/imdb/imdb:model=together_bloom,instructions=expert,groups=ablation_prompts_ground_truth.csv').astype(
            int)
    elif args.dataset_name == 'legal_support':
        # results of two extra models not used: openai_code-cushman-001 openai_code-davinci-002
        n_classes = 2
        args.dataset_path = 'legal_support/legal_support,method=multiple_choice_joint:'
        args.backend_path = ''
        labels = np.loadtxt(
            './logs/helm/legal_support/legal_support,method=multiple_choice_joint:model=ai21_j1-grande-v2-beta_ground_truth.csv').astype(
            int)
    elif args.dataset_name == 'lsat_qa':
        n_classes = 5
        args.dataset_path = 'lsat_qa/lsat_qa:task=all,method=multiple_choice_joint'
        args.backend_path = ''
        labels = np.loadtxt(
            './logs/helm/lsat_qa/lsat_qa:task=all,method=multiple_choice_joint,model=ai21_j1-grande-v2-beta_ground_truth.csv').astype(
            int)
    elif args.dataset_name == 'mmlu':
        n_classes = 4
        args.dataset_path = ['mmlu/mmlu:subject=abstract_algebra,method=multiple_choice_joint,',
                             'mmlu/mmlu:subject=college_chemistry,method=multiple_choice_joint,',
                             'mmlu/mmlu:subject=computer_security,method=multiple_choice_joint,',
                             'mmlu/mmlu:subject=econometrics,method=multiple_choice_joint,',
                             'mmlu/mmlu:subject=us_foreign_policy,method=multiple_choice_joint,']
        args.backend_path = ',data_augmentation=canonical'
        labels = [np.loadtxt(
            './logs/helm/mmlu/mmlu:subject=abstract_algebra,method=multiple_choice_joint,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
            int), np.loadtxt(
            './logs/helm/mmlu/mmlu:subject=college_chemistry,method=multiple_choice_joint,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
            int), np.loadtxt(
            './logs/helm/mmlu/mmlu:subject=computer_security,method=multiple_choice_joint,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
            int), np.loadtxt(
            './logs/helm/mmlu/mmlu:subject=econometrics,method=multiple_choice_joint,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
            int), np.loadtxt(
            './logs/helm/mmlu/mmlu:subject=us_foreign_policy,method=multiple_choice_joint,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
            int)]
        labels = np.concatenate(labels)
    elif args.dataset_name == 'raft':
        n_classes = 2
        args.dataset_path = ['raft/raft:subset=ade_corpus_v2,',
                             'raft/raft:subset=banking_77,',
                             'raft/raft:subset=neurips_impact_statement_risks,',
                             'raft/raft:subset=one_stop_english,',
                             'raft/raft:subset=overruling,',
                             'raft/raft:subset=semiconductor_org_types,',
                             'raft/raft:subset=systematic_review_inclusion,',
                             'raft/raft:subset=tai_safety_research,',
                             'raft/raft:subset=terms_of_service,',
                             'raft/raft:subset=tweet_eval_hate,',
                             'raft/raft:subset=twitter_complaints,', ]
        args.backend_path = ',data_augmentation=canonical'
        labels = [np.loadtxt(
            './logs/helm/raft/raft:subset=ade_corpus_v2,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
            int),
            np.loadtxt(
                './logs/helm/raft/raft:subset=banking_77,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
                int),
            np.loadtxt(
                './logs/helm/raft/raft:subset=neurips_impact_statement_risks,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
                int),
            np.loadtxt(
                './logs/helm/raft/raft:subset=one_stop_english,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
                int),
            np.loadtxt(
                './logs/helm/raft/raft:subset=overruling,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
                int),
            np.loadtxt(
                './logs/helm/raft/raft:subset=semiconductor_org_types,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
                int),
            np.loadtxt(
                './logs/helm/raft/raft:subset=systematic_review_inclusion,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
                int),
            np.loadtxt(
                './logs/helm/raft/raft:subset=tai_safety_research,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
                int),
            np.loadtxt(
                './logs/helm/raft/raft:subset=terms_of_service,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
                int),
            np.loadtxt(
                './logs/helm/raft/raft:subset=tweet_eval_hate,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
                int),
            np.loadtxt(
                './logs/helm/raft/raft:subset=twitter_complaints,model=ai21_j1-grande,data_augmentation=canonical_ground_truth.csv').astype(
                int)]
        labels = np.concatenate(labels)
    '''
    # all model list
    model_names = ['AlephAlpha_luminous-base',
                'AlephAlpha_luminous-extended',
                'AlephAlpha_luminous-supreme',
                'ai21_j1-grande',
                'ai21_j1-grande-v2-beta',
                'ai21_j1-jumbo',
                'ai21_j1-large',
                'ai21_j2-grande',
                'ai21_j2-jumbo',
                'ai21_j2-large',
                'anthropic_stanford-online-all-v4-s3',
                'cohere_command-medium-beta',
                'cohere_command-xlarge-beta',
                'cohere_large-20220720',
                'cohere_medium-20220720',
                'cohere_medium-20221108',
                'cohere_small-20220720',
                'cohere_xlarge-20220609',
                'cohere_xlarge-20221108',
                'eleutherai_pythia-12b-v0',
                'eleutherai_pythia-6.9b',
                'lmsys_vicuna-13b-v1.3',
                'lmsys_vicuna-7b-v1.3',
                'meta_llama-13b',
                'meta_llama-2-13b',
                'meta_llama-2-70b',
                'meta_llama-2-7b',
                'meta_llama-30b',
                'meta_llama-65b',
                'meta_llama-7b',
                'microsoft_TNLGv2_530B',
                'microsoft_TNLGv2_7B',
                'mosaicml_mpt-30b',
                'mosaicml_mpt-instruct-30b',
                'openai_ada',
                'openai_babbage',
                'openai_curie',
                'openai_davinci',
                'openai_gpt-3.5-turbo-0301',
                'openai_gpt-3.5-turbo-0613',
                'openai_text-ada-001',
                'openai_text-babbage-001',
                'openai_text-curie-001',
                'openai_text-davinci-002',
                'openai_text-davinci-003',
                'stanford_alpaca-7b',
                'tiiuae_falcon-40b',
                'tiiuae_falcon-40b-instruct',
                'tiiuae_falcon-7b',
                'tiiuae_falcon-7b-instruct',
                'together_bloom',
                'together_glm,stop=hash',
                'together_gpt-j-6b',
                'together_gpt-neox-20b',
                'together_opt-175b',
                'together_opt-66b',
                'together_redpajama-incite-base-3b-v1',
                'together_redpajama-incite-base-7b',
                'together_redpajama-incite-instruct-3b-v1',
                'together_redpajama-incite-instruct-7b',
                'together_t0pp,stop=hash',
                'together_t5-11b,stop=hash',
                'together_ul2,stop=hash,global_prefix=nlg',
                'together_yalm',
                'writer_palmyra-instruct-30',
                'writer_palmyra-x']
    '''
    return labels, n_classes


def ProcessData(args, model_names, n_classes):
    preds = []
    for i in range(0, len(model_names), 1):
        if args.dataset_name == 'boolq' or args.dataset_name == 'imdb' or args.dataset_name == 'legal_support':
            if ',stop=hash,global_prefix=nlg' in model_names[i]:
                model_pred = np.loadtxt('./logs/helm/' + args.dataset_path + 'model=' + model_names[i].replace(
                    ',stop=hash,global_prefix=nlg',
                    '') + args.backend_path + ',stop=hash,global_prefix=nlg' + '.csv').astype(int)
            elif ',stop=hash' in model_names[i]:
                model_pred = np.loadtxt(
                    './logs/helm/' + args.dataset_path + 'model=' + model_names[i].replace(',stop=hash',
                                                                                           '') + args.backend_path + ',stop=hash' + '.csv').astype(
                    int)
            # elif ',groups=ablation_in_context' in model_names[i]:
            #    model_pred = np.loadtxt('./logs/helm/' + args.dataset_path + 'model=' + model_names[i] + '.csv').astype(int)
            else:
                model_pred = np.loadtxt('./logs/helm/' + args.dataset_path + 'model=' + model_names[
                    i] + args.backend_path + '.csv').astype(int)
            if (args.dataset_name == 'legal_support' and len(model_pred) > 1000):
                model_pred = model_pred[:len(model_pred) // 3]
        elif args.dataset_name == 'mmlu' or args.dataset_name == 'civil_comments' or args.dataset_name == 'entity_matching' \
                or args.dataset_name == 'raft':
            model_pred = []
            for j in range(len(args.dataset_path)):
                if ',stop=hash,global_prefix=nlg' in model_names[i]:
                    model_pred_sub = np.loadtxt(
                        './logs/helm/' + args.dataset_path[j] + 'model=' + model_names[i].replace(
                            ',stop=hash,global_prefix=nlg',
                            '') + args.backend_path + ',stop=hash,global_prefix=nlg' + '.csv').astype(int)
                elif ',stop=hash' in model_names[i]:
                    model_pred_sub = np.loadtxt(
                        './logs/helm/' + args.dataset_path[j] + 'model=' + model_names[i].replace(',stop=hash',
                                                                                                  '') + args.backend_path + ',stop=hash' + '.csv').astype(
                        int)
                else:
                    model_pred_sub = np.loadtxt('./logs/helm/' + args.dataset_path[j] + 'model=' + model_names[
                        i] + args.backend_path + '.csv').astype(int)
                if (args.dataset_name == 'mmlu' and len(model_pred_sub) > 400) or (
                        args.dataset_name == 'civil_comments' and len(model_pred_sub) > 5000) \
                        or (args.dataset_name == 'entity_matching' and j == 0 and len(model_pred_sub) > 1000) \
                        or (args.dataset_name == 'entity_matching' and j == 1 and len(model_pred_sub) > 182) \
                        or (args.dataset_name == 'entity_matching' and j == 2 and len(model_pred_sub) > 218) \
                        or (args.dataset_name == 'raft' and len(model_pred_sub) > 200):
                    model_pred_sub = model_pred_sub[:len(model_pred_sub) // 3]
                model_pred.append(model_pred_sub)
                # print(args.model_names[i], model_pred_sub.shape)
            model_pred = np.concatenate(model_pred)
        else:
            model_pred = np.loadtxt('./logs/helm/' + args.dataset_path + ',model=' + model_names[
                i] + args.backend_path + '.csv').astype(int)
        # set random prediction to random values
        for n in range(len(model_pred)):
            if model_pred[n] == -1:
                model_pred[n] = random.randint(0, n_classes - 1)
                # print('set to random')
        if (args.dataset_name == 'boolq' and len(model_pred) == 15000) or (
                args.dataset_name == 'lsat_qa' and len(model_pred) == 1383):
            # print('cut to 1/3')strat /B python -u ranking.py > example_run.log 2>&1
            model_pred = model_pred[:len(model_pred) // 3]
        preds.append(model_pred)
    return preds


def SML_onevsrest(preds, n_classes):
    # SML One-Vs-Rest
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
        # ensemble weights
        weights = v / np.sum(v)
        weights_all.append(weights)
    weights_final = np.sum(np.array(weights_all), axis=0)

    predictions = np.einsum('a,abc->bc', weights_final, preds_one_hot)
    predict = np.argmax(predictions, axis=1)
    return weights_final / class_num, predict


def convert_label(predict):
    M, n = predict.shape
    pred = np.zeros_like(predict)
    for j in range(M):
        for i in range(n):
            if predict[j][i] == 0:
                pred[j][i] = -1
            elif predict[j][i] == 1:
                pred[j][i] = 1
    return pred


def SML(preds):
    # SML
    pred = np.ones((preds.shape[0], preds.shape[1])) * -1
    for j in range(len(preds)):
        for n in range(len(preds[1])):
            # convert prediction to {-1, 1}
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
    # Ensemble weights
    weights = v / np.sum(v)
    prediction = np.einsum('a,ab->b', v, pred)
    predict = np.where(prediction >= 0, 1, 0)
    return weights, predict


def log_odds_to_convex(log_odds):
    log_odds = np.array(log_odds)
    # Convert log-odds to unnormalized weights (sigmoid)
    unnormalized = 1 / (1 + np.exp(-log_odds))
    # Normalize to sum to 1 (convex combination)
    return unnormalized / unnormalized.sum()


def Stacking(args, preds_test, preds_golden=None, labels_golden=None, return_model=False):
    weight_init = None
    if args.use_weight_init:
        if args.golden_num != 0:
            weight_init = pred_single(preds_golden, labels_golden)  # Calculate the BCA of each model on golden labeled data
            weight = [a for a in weight_init]
            weight_init = weight / sum(weight)
            # weight_init = log_odds_to_convex(weight)
        elif args.golden_num == 0:
            voting_pred = voting(preds_test, n_classes)
            weight_init = pred_single(preds_test, voting_pred)  # Calculate the BCA of each model through Voting on test data
            weight = [a for a in weight_init]
            weight_init = weight / sum(weight)
            # weight_init = args.weights_sml
    else:
        weight_init = np.ones(args.workers, dtype=float) / args.workers
    print('StackingNet weight_init:', weight_init)

    if args.golden_num != 0:
        pred, net = Stacking_Classification(args, n_classes, preds_test, weight_init, preds_golden, labels_golden, return_model=True)
    else:
        pred, net = Stacking_Classification(args, n_classes, preds_test, weight_init, return_model=True)

    if return_model:
        return pred, net
    return pred


def Get_LA(preds, n_classes):
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
    two_pass_truths = two_pass(e2wl, a, label_set)
    pred = np.array(two_pass_truths)
    return pred


def Get_LAA(preds, n_classes, voting_pred):
    preds_one_hot = []
    for i in range(len(preds)):
        max_indices = preds[i]
        encoded_arr = np.zeros((preds.shape[1], n_classes), dtype=int)
        encoded_arr[np.arange(preds.shape[1]), max_indices] = 1
        preds_one_hot.append(encoded_arr)
    preds_one_hot = np.concatenate(preds_one_hot, axis=1)
    pred = LAA_net(preds, preds_one_hot, n_classes, voting_pred)
    return pred


def Get_EBCC(preds, n_classes, n_workers):
    pred = ebcc_vb(preds, n_classes, num_groups=n_workers, empirical_prior=True)
    predict = np.argmax(pred, axis=1)
    return predict


def Get_PM(preds, n_classes, voting_pred, n_workers):
    pred, weight_pm = PM(preds, n_classes, voting_pred, n_workers)
    return pred


def Get_DS(pred_df, return_class=False):
    ds = DawidSkene(n_iter=10)
    pred = ds.fit_predict(pred_df)
    pred = pred.to_numpy()
    pred -= 1
    if return_class:
        return pred, ds
    return pred


def Get_ZC(preds, n_classes):
    pred = ZC(preds, n_classes, iterr=20).Run()
    pred = np.array(pred)
    return pred


def Get_BCA(preds, preds_golden, labels_golden, n_classes):
    true_scores = pred_single(preds_golden, labels_golden)  # ground-truth BCA scores on golden labeled data
    preds_one_hot = []
    for i in range(len(preds)):
        max_indices = preds[i]
        encoded_arr = np.zeros((preds.shape[1], n_classes), dtype=int)
        encoded_arr[np.arange(preds.shape[1]), max_indices] = 1
        preds_one_hot.append(encoded_arr)
    preds = np.stack(preds_one_hot)
    pred = np.argmax(np.einsum('a,abc->bc', true_scores, preds), axis=-1)
    return pred


def voting(preds, n_classes):
    # voting
    n_classifier, n_samples = preds.shape
    votes_mat = np.zeros((n_classes, n_samples))
    for i in range(n_classifier):
        for j in range(n_samples):
            class_id = preds[i, j]
            votes_mat[class_id, j] += 1
    votes_pred = []
    for i in range(n_samples):
        # for equal amount of votes, choose a random one to break ties
        pred = np.random.choice(np.flatnonzero(votes_mat[:, i] == votes_mat[:, i].max()))
        votes_pred.append(pred)
    votes_pred = np.array(votes_pred)
    return votes_pred


def pred_single(preds, labels):
    # calculate BCA of each base classifier
    scores_arr = []
    for i in range(len(preds)):
        predict = preds[i]
        score = np.round(balanced_accuracy_score(labels, predict), 5)
        scores_arr.append(score * 100)
    return scores_arr


def serialize_args(args):
    serializable = {}
    for k, v in vars(args).items():
        try:
            json.dumps(v)  # test if serializable
            serializable[k] = v
        except (TypeError, OverflowError):
            serializable[k] = str(v)  # fallback to string
    return serializable


def Data_Logging(args, results, preds):
    scores = {}
    scores['Dataset'] = args.dataset_name
    scores['Single'] = str(results['Worst']) + '-' + str(results['Average']) + '-' + str(results['Best'])
    for key, value in results.items():
        if key in ['Best', 'Worst', 'Average']:
            continue
        scores[key] = f"{statistics.mean(results[key]):.2f}±{statistics.stdev(results[key]):.2f}"

    pd_result = pd.DataFrame(data=[scores])
    csv_file_path = './results/' + args.log + '_experiment_results.csv'
    mode = 'a' if os.path.exists(csv_file_path) else 'w'
    with open(csv_file_path, mode, newline='') as file:
        pd_result.to_csv(file, header=mode == 'w', index=False)
    print(f"Peformance results appended to {csv_file_path}")

    pd_preds = pd.DataFrame(preds)
    csv_file_path = './results/' + args.log + '_' + str(args.dataset_name) + '_predictions.csv'
    mode = 'w'
    with open(csv_file_path, mode, newline='') as file:
        pd_preds.to_csv(file, header=True, index=False)
    print(f"Prediction results written to {csv_file_path}")

    with open(f'./results/{args.log}_config.json', 'w') as f:
        json.dump(serialize_args(args), f, indent=4)

    # with open('./results/' + args.log + '_config.json', 'r') as f:
    #     loaded_args = argparse.Namespace(**json.load(f))


def construct_dataframe_crowdkit(preds, num_workers):
    num_samples = preds.shape[1]
    a_1 = np.repeat(np.arange(num_samples) + 1, num_workers)
    a_2 = np.array((np.arange(num_workers) + 1).tolist() * num_samples)
    a_3 = np.transpose(preds, (1, 0)).reshape(-1,) + 1
    A = np.stack([a_1, a_2, a_3])
    A = np.transpose(A, (1, 0))
    # b_1 = np.arange(num_samples) + 1
    # b_2 = labels.reshape(-1,) + 1
    # B = np.stack([b_1, b_2])
    # B = np.transpose(B, (1, 0))
    df = pd.DataFrame(A, columns=['task', 'worker', 'label'])
    return df


def seed_everything(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)


def calculate_spearmans_rank_correlation(rank1, rank2):
    """Calculate Spearman's rank correlation coefficient."""
    coef, p_value = spearmanr(rank1, rank2)
    return coef


def calculate_kendalls_tau(rank1, rank2, weighted=True):
    """Calculate Kendall's Tau correlation coefficient."""
    if weighted:
        '''
        n = len(rank1)
        num = 0
        den = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                num += (rank1[i] - rank1[j]) * (rank2[i] - rank2[j]) * (n - max(i, j))
                den += (n - max(i, j))
        if den != 0:
            tau = num / den
        else:
            tau = 0
        '''
        res = weightedtau(rank1, rank2)
        tau = res.statistic
        p_value = res.pvalue
    else:
        tau, p_value = kendalltau(rank1, rank2)
    return tau


def calculate_ndcg(rank_true, rank_pred, k=None):
    """Calculate Normalized Discounted Cumulative Gain."""

    rank_true = np.asarray(rank_true)
    rank_pred = np.asarray(rank_pred)

    if k is None:
        k = len(rank_pred)

    order = np.argsort(rank_pred)[::-1]
    rank_true = rank_true[order]

    gains = 2 ** rank_true - 1
    discounts = np.log2(np.arange(2, k + 2))

    dcg = np.sum(gains[:k] / discounts)
    idcg = np.sum((2 ** np.sort(rank_true)[::-1] - 1) / discounts)

    ndcgscore = dcg / idcg

    ndcgscore_sklearn = ndcg_score(np.array(rank_true).reshape(1, len(rank_true)),
                                   np.array(rank_pred).reshape(1, len(rank_pred)))

    return ndcgscore, ndcgscore_sklearn


def ranking(rank_true, rank_pred):
    spearman = calculate_spearmans_rank_correlation(rank_true, rank_pred)
    kendall = calculate_kendalls_tau(rank_true, rank_pred, weighted=True)
    ndcg = calculate_ndcg(rank_true, rank_pred)

    print("Spearman's Rank Correlation: {:.3f}".format(spearman))
    print("Kendall's Tau: {:.3f}".format(kendall))
    print("NDCG: {:.3f} {:.3f}".format(ndcg[0], ndcg[1]))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _diag_by_worker(errors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a worker x class table of diagonal entries P(y=c | z=c).
    errors_df: MultiIndex rows (worker, observed_label), columns=true_label
    returns: DataFrame indexed by worker, columns=classes
    """
    classes = list(errors_df.columns)
    parts = []
    for c in classes:
        # rows where observed_label == c, pick column c (true=c)
        s = errors_df.xs(c, level='label')[c]  # index=worker
        s.name = c
        parts.append(s)
    diag = pd.concat(parts, axis=1)  # index=worker, columns=classes
    return diag


def worker_expected_accuracy_binary(errors_df: pd.DataFrame,
                                    priors: pd.Series) -> pd.Series:
    """
    Expected accuracy for K=2 using class priors.
    Preserves original worker order from errors_df.
    """
    # original worker order
    worker_order = errors_df.index.get_level_values('worker').drop_duplicates()

    diag = _diag_by_worker(errors_df)         # worker x classes
    classes = list(diag.columns)
    if len(classes) != 2:
        raise ValueError(f"Binary expected accuracy requested but K={len(classes)}.")
    # align priors, normalize just in case
    p = priors.reindex(classes).astype(float)
    p = p / p.sum()
    acc = (diag * p).sum(axis=1)              # sum_c pi_c * P(y=c|z=c)
    return acc.reindex(worker_order)


def worker_expected_bca_multiclass(errors_df: pd.DataFrame) -> pd.Series:
    """
    Balanced accuracy (macro recall) for K>2.
    Preserves original worker order from errors_df.
    """
    worker_order = errors_df.index.get_level_values('worker').drop_duplicates()
    diag = _diag_by_worker(errors_df)         # worker x classes
    if diag.shape[1] <= 2:
        raise ValueError(f"Multiclass BCA requested but K={diag.shape[1]}.")
    bca = diag.mean(axis=1)                   # (1/K) * sum_c P(y=c|z=c)
    return bca.reindex(worker_order)


def worker_reliability(ds) -> pd.Series:
    """
    Convenience wrapper:
    - If K=2: expected accuracy using ds.priors_
    - If K>2: BCA (macro recall)
    """
    errors_df = ds.errors_
    priors = ds.priors_
    K = len(errors_df.columns)
    if K == 2:
        return worker_expected_accuracy_binary(errors_df, priors)
    else:
        return worker_expected_bca_multiclass(errors_df)


if __name__ == '__main__':

    # Prediction results to 8 datasets are almost available to all model_names listed below
    # Datasets: boolq, raft, mmlu, civil_comments, lsat, legal_support, imdb, entity_matching
    # For better performance (removing close-to-random bad models), only 10 are retained
    # see below model_namesmodels

    parser = argparse.ArgumentParser(description='ranking experiment')
    parser.add_argument('--seed', type=int, default=1, help='random seed for start')
    parser.add_argument('--dataset_name', type=str, default='all', help='dataset name')
    parser.add_argument('--about', type=str, default='classification', help='about')
    parser.add_argument('--loss', type=str, default='PM', help='Loss function')
    parser.add_argument('--sigma', type=str2bool, default=True, help='whether to use weight norm regularization in loss objective')
    parser.add_argument('--use_dropout', type=str2bool, default=False, help='whether to use dropout for StackingNet')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate, only used when --use_dropout=True')
    parser.add_argument('--use_weight_init', type=str2bool, default=True, help='whether to use weight initialization for StackingNet')
    parser.add_argument('--net', type=str, default='10->1', help='net structure')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate on loss objective for labeled data')
    parser.add_argument('--unsupervised_weight', type=float, default=0.001, help='weight for loss objective for unlabeled data')
    parser.add_argument('--regularization_relu', type=str2bool, default=False, help='whether to use relu-like nonnegative clamping instead of regularization_weight for StackingNet weights')
    parser.add_argument('--regularization_weight', type=float, default=100., help='weight for loss objective for regularization')
    parser.add_argument('--epoch', type=int, default=200, help='epoch of golden tasks')
    parser.add_argument('--golden_num', type=int, default=5, help='the percentage of the ratio of (labeled data / all data)')
    parser.add_argument('--golden_num_not_ratio', type=int, default=-1, help='the exact number of labeled data, per class')
    parser.add_argument('--log', type=str, default='none', help='time log')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu ID')
    parser.add_argument('--malicious', type=int, default=0, help='experiment on malicious attack of model compromisation. 0 = no attack, 1 = add a model with random outputs, 2 = the model with best groundtruth performance is set to predict randomly')
    args = parser.parse_args()

    seed_everything(args)

    if (args.dataset_name == 'all'):
        dataset_name = ['boolq', 'entity_matching', 'imdb', 'legal_support', 'lsat_qa', 'mmlu', 'raft',
                        'civil_comments']
    else:
        dataset_name = [args.dataset_name]

    if args.malicious == 1:
        args.workers = 11
    else:
        args.workers = 10

    args.path = './args/' + args.log + args.dataset_name + '.txt'
    with open(args.path, 'w', encoding='utf-8') as f:
        json.dump(args.__dict__, f, ensure_ascii=False, indent=2)

    # GPU device id
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
    print(f"Available CUDA devices: {torch.cuda.device_count()}")
    try:
        if torch.cuda.device_count() == 1:
            args.device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
        else:
            args.device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    except:
        args.device = 'local'
    print('Device:', args.device)

    for args.dataset_name in dataset_name:
        print('#' * 100)
        print('#' * 40, 'dataset name:', args.dataset_name, '#' * 40)
        print(f"dataset name: {args.dataset_name}")
        print('#' * 100)

        scores = {}
        scores['data'] = args.dataset_name
        scores['log'] = args.log

        # selected 10 models with better performance, also with diversity (not multiple versions of models from the same company)
        model_names = ['meta_llama-2-70b', 'openai_gpt-3.5-turbo-0301', 'ai21_j2-jumbo', 'writer_palmyra-x',
                       'tiiuae_falcon-40b', 'cohere_command-xlarge-beta', 'microsoft_TNLGv2_530B',
                       'mosaicml_mpt-instruct-30b', 'anthropic_stanford-online-all-v4-s3',
                       'together_redpajama-incite-instruct-7b']

        args.model_names = model_names

        # Get labelse from args dataset_name
        labels, n_classes = ReadLabel(args)
        print('labels.shape:', labels.shape)
        args.classes = n_classes
        # Get predictions from 10 different models
        preds = ProcessData(args, model_names, n_classes)
        print('model_names:', args.model_names)

        if args.golden_num == -1:
            args.golden_num = 100 * args.golden_num_not_ratio * n_classes / labels.shape[0]
        print('args.golden_num:', args.golden_num)

        if args.malicious != 0:
            length = labels.shape[0]
            choices = np.arange(n_classes)
            result = np.random.choice(choices, size=length)
            if args.malicious == 2:
                if args.dataset_name == 'boolq' or args.dataset_name == 'entity_matching' or args.dataset_name == 'legal_support' or args.dataset_name == 'mmlu':
                    for i in range(length):
                        available_values = [x for x in choices if x != preds[3][i]]
                        result[i] = np.random.choice(available_values)
                    preds[3] = result
                elif args.dataset_name == 'imdb':
                    for i in range(length):
                        available_values = [x for x in choices if x != preds[0][i]]
                        result[i] = np.random.choice(available_values)
                    preds[0] = result
                elif args.dataset_name == 'lsat_qa' or args.dataset_name == 'raft' or args.dataset_name == 'civil_comments':
                    for i in range(length):
                        available_values = [x for x in choices if x != preds[9][i]]
                        result[i] = np.random.choice(available_values)
                    preds[9] = result
            malicious = np.random.choice(choices, size=length)
            if args.malicious == 1:
                preds.append(malicious)
                model_names = ['meta_llama-2-70b', 'openai_gpt-3.5-turbo-0301', 'ai21_j2-jumbo', 'writer_palmyra-x',
                               'tiiuae_falcon-40b', 'cohere_command-xlarge-beta', 'microsoft_TNLGv2_530B',
                               'mosaicml_mpt-instruct-30b', 'anthropic_stanford-online-all-v4-s3',
                               'together_redpajama-incite-instruct-7b', 'malicious']
                print('appending a malicious model with random outputs!')
        # If malicious==1, we add a randomly guessed model as a distractor
        # Plus, we let the model with the highest accuracy do the random output if the malicious == 2

        # ---- after you finished injecting attacks into preds / model_names ----
        # record attack model index
        if args.malicious != 0:
            attacked_idx_fixed = None
            if args.malicious == 1:
                # the random attack base model index is at the end
                attacked_idx_fixed = len(model_names) - 1
            elif args.malicious == 2:
                # the targeted attack base model index
                attacked_map = {
                    'boolq': 3, 'entity_matching': 3, 'legal_support': 3, 'mmlu': 3,
                    'imdb': 0, 'lsat_qa': 9, 'raft': 9, 'civil_comments': 9,
                }
                attacked_idx_fixed = attacked_map.get(args.dataset_name, None)

        # preds' shape is (num_classifiers, num_samples)
        preds = np.stack(preds)
        print('preds.shape', preds.shape)

        # duplicate labels for certain datasets to match number of samples (as the original dataset used "augmented" queries under few-shot examples)
        if preds.shape[-1] % len(labels) == 0:
            if preds.shape[-1] // len(labels) == 3:
                labels = np.concatenate((labels, labels, labels))
            elif preds.shape[-1] // len(labels) == 1:
                print('labels shape all good')
            else:
                print('check preds shape!')
                sys.exit(0)
        # label's shape is (task_nums,)
        print('labels.shape', labels.shape)

        #####################################################Done with polishing predictions################################
        voting_score = []
        wawa_score = []
        dawidskene_score = []
        mmsr_score = []
        mace_score = []
        glad_score = []
        kos_score = []
        stacking_score = []
        sml_score = []
        la_score = []
        laa_score = []
        ebcc_score = []
        pm_score = []
        zencrowd_score = []
        best_score = []

        start_seed = 0

        print('shuffle, and retain 80% data as test data for all random seeds cases')

        preds_T = preds.T  # shape: (num_samples, num_classifiers)

        # split data into 20% retained POOL for labeled "training" data
        # the actual x% labeled data will be further randomly selected in this pool (x < 20)
        # while the 80% fixed test set is used for actual performance evaluation in all cases under all random seeds
        preds_train_T, preds_test_T, labels_train_T, labels_test_T, idx_train, idx_test = train_test_split(
            preds_T, labels, np.arange(len(labels)),
            test_size=0.8,
            random_state=start_seed,
        )

        preds_train, preds_test, labels_train, labels_test = preds_train_T.T, preds_test_T.T, labels_train_T.T, labels_test_T.T

        print('preds_train.shape, labels_train.shape (Pool of Labeled Data):', preds_train.shape, labels_train.shape)
        print('preds_test.shape, labels_test.shape:', preds_test.shape, labels_test.shape)

        # Save the split indices (for pool of labeled data and test data, not for different golden data under random seeds) for reproducibility
        np.savez('./results/' + args.dataset_name + '_' + args.log + '_split_indices.npz', idx_train=idx_train, idx_test=idx_test)

        singles_all = []
        for i in range(len(preds)):
            model = model_names[i]
            score = np.round(balanced_accuracy_score(labels_test, preds_test[i]), 5)
            # these ground-truth performance scores on the test set are supposed to be the same across all random seeds, due to the split scenario
            print('Test set ground-truth BCA: {:.2f}'.format(score * 100), 'Model_Name: {}'.format(model))
            singles_all.append(np.round(score * 100, 2))
        single_avg = np.round(np.mean(singles_all), 2)
        print('Test set average ground-truth BCA: {:.2f}'.format(single_avg))
        single_worst = min(singles_all)
        single_best = max(singles_all)
        global_model = model_names[singles_all.index(single_best)]
        print('Test set worst & best single-model BCA: {:.2f} & {:.2f}'.format(single_worst, single_best), 'Best_Model_Name: {}'.format(global_model))

        if args.malicious != 0:
            attack_buf = {
                "seeds": [],
                "weights_stack": [],  # StackingNet weight for each seed (length=M)
                "weights_sml": [],  # SML weight for each seed (length=M)
                "labels": []  # label vector for each seed (0/1) (length=M)
            }

        # repeat five experiments with different random seeds (affects labeled data portions, etc.)
        # we call the actual labeled data with ground-truth annotations as golden from here on
        for seed in range(5):
            print('#' * 60)
            print('#' * 20, 'random seed:', seed, '#' * 20)
            print('#' * 60)
            args.seed = seed
            seed_everything(args)

            if args.golden_num == 0:
                golden_num = 0
            else:
                golden_num = int(preds.shape[1] // (100 // args.golden_num))

            # golden_indices = np.random.choice(len(preds_train_T), size=golden_num, replace=False)

            # class-balanced sampling of labeled data
            if args.golden_num == 0:
                preds_golden_and_test = preds_test
                labels_golden_and_test = labels_test
                preds_golden = None
                labels_golden = None
            else:
                if args.golden_num_not_ratio != -1:
                    sss = StratifiedShuffleSplit(n_splits=1, train_size=int(args.golden_num_not_ratio * n_classes), random_state=seed)
                    golden_num = int(args.golden_num_not_ratio * n_classes)
                else:
                    sss = StratifiedShuffleSplit(n_splits=1, train_size=int(golden_num), random_state=seed)  # must be int to be interpreted as actual numbers
                (golden_indices, _), = sss.split(np.zeros(len(labels_train_T)),
                                                 labels_train_T)

                preds_golden = preds_train_T[golden_indices].T
                labels_golden = labels_train_T[golden_indices].T
                print('preds_golden.shape, labels_golden.shape:', preds_golden.shape, labels_golden.shape)
                preds_golden_and_test = np.concatenate([preds_golden, preds_test], axis=1)
                labels_golden_and_test = np.concatenate([labels_golden, labels_test])
                print('preds_golden_and_test.shape, labels_golden_and_test.shape:', preds_golden_and_test.shape, labels_golden_and_test.shape)

            print('golden_num:', golden_num)

            if golden_num > 0:
                # Per-model BCA on the small labeled subset, in model_names order
                small_bca = pred_single(preds_golden, labels_golden)  # returns percentages
                # Emit a single comma-separated list in model_names order.
                print('Model estimated BCA with only labeled golden data: ' + ", ".join(f"{v:.2f}" for v in small_bca))

            # Voting
            voting_pred = voting(preds_test, n_classes)
            score = np.round(balanced_accuracy_score(labels_test, voting_pred) * 100, 2)
            voting_score.append(score)
            print('Voting: {:.2f}'.format(score))

            # build dataframes for algorithms from crowdkit library
            pred_test_df = construct_dataframe_crowdkit(preds_test, args.workers)
            pred_golden_and_test_df = construct_dataframe_crowdkit(preds_golden_and_test, args.workers)

            # WAwA
            if args.golden_num == 0:
                wawa_pred = Wawa().fit_predict(pred_test_df)
                wawa_pred = wawa_pred.to_numpy()
                wawa_pred -= 1
            # when ground-truth labels are available, directly use the BCAs as weights
            else:
                wawa_pred = Get_BCA(preds_test, preds_golden, labels_golden, n_classes)
            score = np.round(balanced_accuracy_score(labels_test, wawa_pred) * 100, 2)
            wawa_score.append(score)
            print('WAwA: {:.2f}'.format(score))

            # Dawid-Skene
            # dawidskene_pred = Get_DS(pred_golden_and_test_df)
            dawidskene_pred, ds = Get_DS(pred_golden_and_test_df, return_class=True)
            rel = worker_reliability(ds).tolist()
            print('Dawid-Skene Weights:', rel)  # printed as weights, but are actually the estimated accuracy/BCA from the estimated confusion matrices and the estimated class priors after EM from D-S

            if args.golden_num == 0:
                dawidskene_pred = dawidskene_pred[golden_num:]
            else:
                dawidskene_pred = dawidskene_pred[golden_num:]
            score = np.round(balanced_accuracy_score(labels_test, dawidskene_pred) * 100, 2)
            dawidskene_score.append(score)
            print('Dawid-Skene Estimated BCA:', ", ".join(f"{v:.3f}" for v in pred_single(preds_test, dawidskene_pred)))
            print('Dawid-Skene: {:.2f}'.format(score))

            # M-MSR
            mmsr_pred = MMSR(random_state=seed).fit_predict(pred_golden_and_test_df)
            mmsr_pred = mmsr_pred.to_numpy()
            mmsr_pred -= 1
            mmsr_pred = mmsr_pred[golden_num:]
            score = np.round(balanced_accuracy_score(labels_test, mmsr_pred) * 100, 2)
            mmsr_score.append(score)
            print('M-MSR: {:.2f}'.format(score))

            # MACE
            mace_pred = MACE(random_state=seed).fit_predict(pred_golden_and_test_df)
            mace_pred = mace_pred.to_numpy()
            mace_pred -= 1
            mace_pred = mace_pred[golden_num:]
            score = np.round(balanced_accuracy_score(labels_test, mace_pred) * 100, 2)
            mace_score.append(score)
            print('MACE: {:.2f}'.format(score))

            # GLAD
            glad_pred = GLAD().fit_predict(pred_golden_and_test_df)
            glad_pred = glad_pred.to_numpy()
            glad_pred -= 1
            glad_pred = glad_pred[golden_num:]
            score = np.round(balanced_accuracy_score(labels_test, glad_pred) * 100, 2)
            glad_score.append(score)
            print('GLAD: {:.2f}'.format(score))

            # KOS
            if n_classes == 2:
                kos_pred = KOS(random_state=seed).fit_predict(pred_golden_and_test_df)
                kos_pred = kos_pred.to_numpy()
                kos_pred -= 1
                kos_pred = kos_pred[golden_num:]
                score = np.round(balanced_accuracy_score(labels_test, kos_pred) * 100, 2)
                kos_score.append(score)
                print('KOS: {:.2f}'.format(score))
            else:
                kos_pred = voting_pred
                kos_score.append(0.)
                print('KOS not implemented for multi-class case.')

            # SML (including binary and multi-class)
            if n_classes > 2:
                weights_sml, sml_pred = SML_onevsrest(preds_golden_and_test, n_classes)
            else:
                weights_sml, sml_pred = SML(preds_golden_and_test)
            sml_pred = sml_pred[golden_num:]
            score = np.round(balanced_accuracy_score(labels_test, sml_pred) * 100, 2)
            sml_score.append(score)
            print('SML Weights:', ", ".join(f"{v:.3f}" for v in weights_sml))
            print('SML Estimated BCA:', ", ".join(f"{v:.3f}" for v in pred_single(preds_test, sml_pred)))
            print('SML: {:.2f}'.format(score))

            ##################################################################################
            # StackingNet, proposed in our paper "StackingNet: Inference Aggregation of Independent Large Foundation Models Enables Collective AI"
            if golden_num != 0:
                stacking_pred, stackingnet = Stacking(args, preds_test, preds_golden, labels_golden, return_model=True)
            else:
                stacking_pred, stackingnet = Stacking(args, preds_test, return_model=True)
            score = np.round(balanced_accuracy_score(labels_test, stacking_pred) * 100, 2)
            stacking_score.append(score)
            print('StackingNet Weights:', ", ".join(f"{v:.3f}" for v in stackingnet.weights.detach().cpu().flatten()))
            print('StackingNet Estimated BCA:', ", ".join(f"{v:.3f}" for v in pred_single(preds_test, stacking_pred)))
            print('Stacking: {:.2f}'.format(score))
            ##################################################################################

            if args.malicious != 0:
                attack_buf["seeds"].append(seed)

                w_stack = stackingnet.weights.detach().cpu().flatten().numpy().astype(float)
                attack_buf["weights_stack"].append(w_stack)

                w_sml = np.asarray(weights_sml, dtype=float)
                attack_buf["weights_sml"].append(w_sml)

                # labels：which model been attacked
                lab = np.zeros_like(w_stack, dtype=int)
                if attacked_idx_fixed is not None and 0 <= attacked_idx_fixed < lab.size:
                    lab[attacked_idx_fixed] = 1
                attack_buf["labels"].append(lab)
                # ====================================

            # LA
            la_pred = Get_LA(preds_golden_and_test, n_classes)
            la_pred = la_pred[golden_num:]
            score = np.round(balanced_accuracy_score(labels_test, la_pred) * 100, 2)
            la_score.append(score)
            print('LA: {:.2f}'.format(score))

            # LAA
            if golden_num == 0:
                laa_pred = Get_LAA(preds_golden_and_test, n_classes, voting_pred)
            else:
                laa_pred = Get_LAA(preds_golden_and_test, n_classes, np.concatenate([labels_golden, voting_pred]))
            laa_pred = laa_pred[golden_num:]
            score = np.round(balanced_accuracy_score(labels_test, laa_pred) * 100, 2)
            laa_score.append(score)
            print('LAA: {:.2f}'.format(score))

            # EBCC
            ebcc_pred = Get_EBCC(preds_golden_and_test, n_classes, args.workers)
            ebcc_pred = ebcc_pred[golden_num:]
            score = np.round(balanced_accuracy_score(labels_test, ebcc_pred) * 100, 2)
            ebcc_score.append(score)
            print('EBCC: {:.2f}'.format(score))

            pm_pred = Get_PM(preds_golden_and_test, n_classes, voting_pred, args.workers)
            pm_pred = pm_pred[golden_num:]
            score = np.round(balanced_accuracy_score(labels_test, pm_pred) * 100, 2)
            pm_score.append(score)
            print('PM: {:.2f}'.format(score))

            zencrowd_pred = Get_ZC(preds_golden_and_test, n_classes)
            zencrowd_pred = zencrowd_pred[golden_num:]
            score = np.round(balanced_accuracy_score(labels_test, zencrowd_pred) * 100, 2)
            zencrowd_score.append(score)
            print('ZenCrowd: {:.2f}'.format(score))

        if args.malicious != 0:
            # save attack detection results
            outdir = f'./results/attack_eval/{args.log}'
            os.makedirs(outdir, exist_ok=True)
            outpath = os.path.join(outdir, f'{args.dataset_name}_atk{args.malicious}.npz')

            # save with metadata
            np.savez_compressed(
                outpath,
                dataset=args.dataset_name,
                attack=np.array(['none', 'random', 'targeted'][args.malicious]),
                model_names=np.array(model_names),
                seeds=np.array(attack_buf["seeds"], dtype=int),
                weights_stack=np.vstack(attack_buf["weights_stack"]),  # shape: (S, M)
                weights_sml=np.vstack(attack_buf["weights_sml"]),  # shape: (S, M)
                labels=np.vstack(attack_buf["labels"])  # shape: (S, M), 1=attack exists
            )
            print(f'[ATTACK_EVAL] saved -> {outpath}')
            # ===========================================================

        # Document the performance results and prediction values
        results = {'Worst': single_worst, 'Average': single_avg, 'Best': single_best,
                   'Voting': voting_score, 'WAwA': wawa_score, 'Dawid-Skene': dawidskene_score,
                  'M-MSR': mmsr_score, 'MACE': mace_score, 'GLAD': glad_score, 'KOS': kos_score,
                   'SML': sml_score, 'LA': la_score, 'LAA': laa_score, 'EBCC': ebcc_score,
                   'PM': pm_score, 'ZenCrowd': zencrowd_score, 'Stacking': stacking_score}
        preds = {}
        for i, model_name in enumerate(model_names):
            preds[model_name] = preds_test[i]
        preds.update({'Voting': voting_pred, 'WAwA': wawa_pred, 'Dawid-Skene': dawidskene_pred,
                'M-MSR': mmsr_pred, 'MACE': mace_pred, 'GLAD': glad_pred, 'KOS': kos_pred,
                'SML': sml_pred, 'LA': la_pred, 'LAA': laa_pred, 'EBCC': ebcc_pred, 'PM': pm_pred,
                'ZenCrowd': zencrowd_pred, 'Stacking': stacking_pred, 'Ground-Truth': labels_test})

        Data_Logging(args, results, preds)

