"""
analysis/stacknet_clf.py
------------------------
Thin wrapper to run the REAL supervised StackingNet classification combiner
(classification_stackingnet.Stacking -> algs.StackingNet_classification) on HELM data,
reusing the exact split logic from the manuscript's main script:
  * 80/20 test/pool split (train_test_split, test_size=0.8, random_state=0)
  * per-seed class-balanced golden labels (StratifiedShuffleSplit, train_size=golden_num)
  * golden_num = n_samples // (100 // golden_pct)   (golden_pct=5 -> 5% labeled, script default)

Used by Exp C to obtain S-StackingNet balanced accuracy per HELM dataset (the recorded
results CSV only contains boolq) and for the strong-subset experiment. Verified to
reproduce the recorded boolq number, which (with the Exp-B base-BCA check) closes the
research-integrity gate on the classification path.
"""
import os, sys, argparse
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from helm_io import load_helm, HELM_DATASETS          # also injects crowdkit stub
import classification_stackingnet as C                 # already importable via the stub


def _default_args(M, K, golden_pct=5):
    a = argparse.Namespace()
    a.loss = 'PM'; a.sigma = True
    a.use_dropout = False; a.dropout_rate = 0.1
    a.use_weight_init = True
    a.net = '10->1'; a.lr = 0.001
    a.unsupervised_weight = 0.001
    a.regularization_relu = False; a.regularization_weight = 100.0
    a.epoch = 200
    a.golden_num = golden_pct; a.golden_num_not_ratio = -1
    a.device = 'cpu'
    a.workers = M; a.classes = K
    a.seed = 0
    return a


def _silent(fn, *a, **k):
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _onehot_stack(preds_MxN, K):
    """Replicate the script's StackingClassifier feature map (lines 952-955)."""
    M, N = preds_MxN.shape
    eye = np.eye(K)
    return np.concatenate([eye[preds_MxN[m]] for m in range(M)], axis=1)  # (N, M*K)


def run_helm_stackingnet(preds, labels, K, golden_pct=5, seeds=5, subset_cols=None, verbose=False):
    """Return base best/worst BCA, StackingNet BCA mean/std, and (supervised only)
    LogisticRegression BCA. golden_pct=0 => U-StackingNet (unsupervised, no golden).
    preds: (M, n) class ids; labels: (n,). subset_cols: optional list of model indices."""
    if subset_cols is not None:
        preds = preds[subset_cols]
    M = preds.shape[0]
    C.n_classes = K                                    # Stacking()/Get_BCA reference this global
    args = _default_args(M, K, golden_pct)

    preds_T = preds.T
    tr_T, te_T, ltr, lte = train_test_split(preds_T, labels, test_size=0.8, random_state=0)
    preds_test = te_T.T
    base_bca = [100 * balanced_accuracy_score(lte, preds_test[i]) for i in range(M)]

    sn, logreg = [], []
    golden_num = 0 if golden_pct == 0 else preds.shape[1] // (100 // golden_pct)
    for seed in range(seeds):
        args.seed = seed
        C.seed_everything(args)
        if golden_pct == 0:
            args.golden_num = 0
            pred = (C.Stacking(args, preds_test) if verbose
                    else _silent(C.Stacking, args, preds_test))
        else:
            args.golden_num = golden_pct
            sss = StratifiedShuffleSplit(n_splits=1, train_size=int(golden_num), random_state=seed)
            (gidx, _), = sss.split(np.zeros(len(ltr)), ltr)
            preds_golden, labels_golden = tr_T[gidx].T, ltr[gidx]
            pred = (C.Stacking(args, preds_test, preds_golden, labels_golden) if verbose
                    else _silent(C.Stacking, args, preds_test, preds_golden, labels_golden))
            # supervised StackingClassifier (LogisticRegression) baseline
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000, solver="lbfgs")
            clf.fit(_onehot_stack(preds_golden, K), labels_golden)
            lp = clf.predict(_onehot_stack(preds_test, K))
            logreg.append(100 * balanced_accuracy_score(lte, lp))
        sn.append(100 * balanced_accuracy_score(lte, pred))
    out = {"n_models": M, "golden_pct": golden_pct, "golden_num": int(golden_num),
           "base_best": float(max(base_bca)), "base_worst": float(min(base_bca)),
           "stackingnet_bca_mean": float(np.mean(sn)), "stackingnet_bca_std": float(np.std(sn)),
           "gain_over_best": float(np.mean(sn) - max(base_bca))}
    if logreg:
        out["logreg_bca_mean"] = float(np.mean(logreg))
        out["logreg_bca_std"] = float(np.std(logreg))
    return out


if __name__ == '__main__':
    # smoke: boolq should land near the recorded 'Stacking' value (~89.1) and >= base best
    preds, labels, K, names = load_helm('boolq')
    r = run_helm_stackingnet(preds, labels, K, seeds=5)
    print("boolq:", r)
