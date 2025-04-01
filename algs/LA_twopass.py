import random
from copy import deepcopy

def one_pass(e2wl, w2el, label_set,alpha=2,beta=2,shuffle = True, record_a_evo = False):
    items = list(e2wl.keys())
    m = len(w2el)
    n = len(e2wl)
    K = len(label_set)

    c={}
    t={}
    a={}
    truths={}
    if record_a_evo:
        a_evo = {}
    for worker in w2el.keys():
        c[worker] = alpha - 1
        t[worker] = alpha + beta - 2
        a[worker] = c[worker] / t[worker]

    if shuffle:
        random.shuffle(items)

    for item in items:
        #aggregate truth
        votes = {}
        for worker, worker_label in e2wl[item]:
            if votes.get(worker_label) is None:
                votes[worker_label] = 0
            votes[worker_label]  = votes[worker_label] + a[worker]
        candidate = []
        max_ = -1
        for class_ in label_set:
            if votes.get(class_) is None:
                continue
            if votes.get(class_) > max_:
                candidate.clear()
                candidate.append(class_)
                max_ = votes.get(class_)
            elif votes.get(class_) == max_:
                candidate.append(class_)
        truths[item] = random.choice(candidate)

        #update ability
        for worker, worker_label in e2wl[item]:
            t[worker] = t[worker] + 1
            if worker_label == truths[item]:
                c[worker] = c[worker] + 1
            a[worker] = c[worker] / t[worker]
        if record_a_evo:
            a_evo[item] = deepcopy(a)
    if record_a_evo:
        return truths, a, a_evo
    else:
        return truths, a

def two_pass(e2wl,a,label_set):
    K = len(label_set)
    truths = []
    items = list(e2wl.keys())
    for item in items:
        votes = {}
        for worker, worker_label in e2wl[item]:
            if votes.get(worker_label) is None:
                votes[worker_label] = 0
            votes[worker_label] = votes[worker_label] + (a[worker] * K - 1)
        candidate = []
        max_ = -1
        for class_ in label_set:
            if votes.get(class_) is None:
                continue
            if votes.get(class_) > max_:
                candidate.clear()
                candidate.append(class_)
                max_ = votes.get(class_)
            elif votes.get(class_) == max_:
                candidate.append(class_)
        truths.append(random.choice(candidate))

    return truths
