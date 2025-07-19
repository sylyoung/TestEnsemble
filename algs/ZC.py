# -*- coding: utf-8 -*-
# @Time    : 2025/7/18
# @Author  : Chenhao Liu
# @File    : ZC.py
# ZenCrowd: leveraging probabilistic reasoning and crowdsourcing techniques for large-scale entity linking
import math
import random
import numpy as np


class ZC:

    def __init__(self, preds, num_labels, iterr = 20):
        self.num_workers = preds.shape[0]
        self.num_tasks = preds.shape[1]
        self.num_labels = num_labels
        self.label_set = set(range(num_labels))
        self.workers = list(range(self.num_workers))
        self.preds = preds
        self.iterr = iterr

        self.e2wl = {}
        for t in range(self.num_tasks):
            self.e2wl[t] = [(w, preds[w][t]) for w in self.workers]

        self.w2el = {}
        for w in self.workers:
            self.w2el[w] = [(t, preds[w][t]) for t in range(self.num_tasks)]

        ###################################################################
        # Expectation Maximization
        ###################################################################

    def InitPj(self):
        l2pd = {}
        for label in self.label_set:
            l2pd[label] = 1.0 / len(self.label_set)
        return l2pd

    def InitWM(self, workers):
        wm = {}

        if workers == {}:

            workers = self.w2el.keys()
            for worker in workers:
                wm[worker] = 0.8
        else:
            for worker in workers:
                if worker not in wm:  # workers --> wm
                    wm[worker] = 0.8
                else:
                    wm[worker] = workers[worker]

        return wm

    # E-step
    def ComputeTij(self, e2wl, l2pd, wm):
        e2lpd = {}
        for e, workerlabels in e2wl.items():
            e2lpd[e] = {}
            for label in self.label_set:
                e2lpd[e][label] = 1.0  # l2pd[label]

            for worker, label in workerlabels:
                for candlabel in self.label_set:
                    if label == candlabel:
                        e2lpd[e][candlabel] *= wm[worker]
                    else:
                        e2lpd[e][candlabel] *= (1 - wm[worker]) * 1.0 / (len(self.label_set) - 1)  # 极大似然估计

            sums = 0
            for label in self.label_set:
                sums += e2lpd[e][label]

            if sums == 0:
                for label in self.label_set:
                    e2lpd[e][label] = 1.0 / self.len(self.label_set)
            else:
                for label in self.label_set:
                    e2lpd[e][label] = e2lpd[e][label] * 1.0 / sums

        # print e2lpd
        return e2lpd

    # M-step

    def ComputeWM(self, w2el, e2lpd):
        wm = {}
        for worker, examplelabels in w2el.items():
            wm[worker] = 0.0
            for e, label in examplelabels:
                wm[worker] += e2lpd[e][label] * 1.0 / len(examplelabels)

        return wm

    def Run(self, workers={}):
        # wm     worker_to_confusion_matrix = {}
        # e2pd   example_to_softlabel = {}
        # l2pd   label_to_priority_probability = {}
        iter = self.iterr
        l2pd = self.InitPj()
        wm = self.InitWM(workers)
        while iter > 0:
            # E-step
            e2lpd = self.ComputeTij(self.e2wl, {}, wm)

            # M-step
            wm = self.ComputeWM(self.w2el, e2lpd)

            # print l2pd,wm

            iter -= 1

        pred = getaccuracy(e2lpd)
        return pred


def getaccuracy(e2lpd):
    preds = []
    for e in e2lpd:
        temp = 0
        for label in e2lpd[e]:
            if temp < e2lpd[e][label]:
                temp = e2lpd[e][label]

        candidate = []

        for label in e2lpd[e]:
            if temp == e2lpd[e][label]:
                candidate.append(label)

        pred = random.choice(candidate)
        preds.append(pred)

    return preds


