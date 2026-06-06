"""
Exp C -- Dispersion vs gain, and strong-subset robustness.
==========================================================
Answers Reviewer 1 comment 5: "relate gains to the dispersion between best and worst
base models across tasks", and clarify how much improvement remains when the base pool
is uniformly strong.

Part 1 -- dispersion vs gain across all tasks:
  For each task we compute
    dispersion = |best base perf - worst base perf|   (native metric)
    gain       = StackingNet perf - best base perf    (signed so + = StackingNet better)
  Sources (clearly separated by metric):
    * CFD (13, regression, MAE in [0,1])      : from results/CFD/<log>/*_predictions.xlsx
    * paper-review (4, regression, MAE in [1,10]) : StackingNet trained here (real
      StackingRegression), few-shot 1% labels, 5-seed mean
    * HELM (8, classification, balanced acc %)  : the repo's recorded full-pipeline
      results (results/none_experiment_results.csv): base 'Single' worst-med-best and
      supervised StackingNet 'Stacking'. (We do not re-run the classifier here; these
      are the manuscript's own numbers.)
  We report Spearman rho(dispersion, gain) within each metric type -- the prediction is
  positive: more gain where base models are more dispersed (part of the gain is robust
  down-weighting of weak contributors, which is a feature, not a confound).

Part 2 -- strong-subset:
  On each paper-review task, restrict the pool to the top-k strongest base models (lowest
  train MAE) -- a uniformly strong, low-dispersion pool -- and report the residual gain of
  StackingNet over the best base. (The complementary HELM evidence already exists in the
  manuscript as the sequential weak-model pruning experiment, Fig. 4f.)

Usage:
  python analysis/exp_C_dispersion_gain.py
"""
import os, sys, json, argparse, re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from exp_B2_controlled_degradation import fit_stackingnet  # reuse real StackingRegression fit

CFD_ATTRS = ['afraid', 'angry', 'attractive', 'babyfaced', 'disgusted', 'feminine',
             'happy', 'masculine', 'sad', 'surprised', 'threatening', 'trustworthy', 'unusual']
CFD_BASE = ['BLIP', 'DeepSeek-VL', 'H2OVL', 'InternVL-2', 'LLaVA',
            'Molmo', 'Paligemma', 'Phi-3.5', 'SAIL-VL', 'SmolVLM']
PR_TASKS = ['ICLR2025', 'ICLR2024', 'NeurIPS2024', 'NeurIPS2023']
PR_LLMS = ['deepseek', 'doubao', 'gemini-2', 'gpt-5', 'gpt4o', 'qwent_turbo']


# ---------------- CFD (regression, from saved per-sample preds) ----------------
def cfd_points(log):
    pts = []
    for a in CFD_ATTRS:
        df = pd.read_excel(os.path.join(REPO, 'results', 'CFD', log, f'{a}_predictions.xlsx'))
        y = df['label'].to_numpy(float)
        base_mae = {m: float(np.mean(np.abs(df[m].to_numpy(float) - y))) for m in CFD_BASE}
        sn_mae = float(np.mean(np.abs(df['StackingNet'].to_numpy(float) - y)))
        best, worst = min(base_mae.values()), max(base_mae.values())
        pts.append({"task": f"CFD-{a}", "type": "regression",
                    "dispersion": worst - best, "best_base": best,
                    "stackingnet": sn_mae, "gain": best - sn_mae})  # +gain = lower MAE
    return pts


# ---------------- paper-review (regression, train StackingNet here) ----------------
def pr_points(label_frac=0.01, seeds=5):
    pts = []
    for t in PR_TASKS:
        df = pd.read_excel(os.path.join(REPO, 'data', 'Research_Review', f'{t}.xlsx'))
        df = df[df['paperID'] != 'MAE']
        y = np.clip(df['label'].to_numpy(float), 1, 10)
        P = np.clip(np.vstack([df[m].to_numpy(float) for m in PR_LLMS]).T, 1, 10)  # (n, M)
        n = len(y); k = max(int(label_frac * n), P.shape[1] + 2)
        sn_maes, best_maes, disp = [], [], []
        for s in range(seeds):
            rng = np.random.default_rng(s)
            idx = rng.permutation(n); tr, te = idx[:k], idx[k:]
            base_mae = [np.mean(np.abs(P[te, j] - y[te])) for j in range(P.shape[1])]
            pred, _, _ = fit_stackingnet(P[tr], y[tr], P[te], seed=s)
            sn_maes.append(np.mean(np.abs(pred - y[te])))
            best_maes.append(min(base_mae))
            disp.append(max(base_mae) - min(base_mae))
        best = float(np.mean(best_maes)); sn = float(np.mean(sn_maes))
        pts.append({"task": t, "type": "regression", "dispersion": float(np.mean(disp)),
                    "best_base": best, "stackingnet": sn, "gain": best - sn})
    return pts


# ---------------- HELM (classification, computed with the real combiner) ----------------
def helm_points(golden_pct=10, seeds=5):
    """Compute base dispersion + S-StackingNet gain for all 8 HELM datasets using the real
    classification combiner (the recorded results CSV only contains boolq)."""
    from helm_io import load_helm, HELM_DATASETS
    from stacknet_clf import run_helm_stackingnet
    pts = []
    for ds in HELM_DATASETS:
        preds, labels, K, names = load_helm(ds)
        r = run_helm_stackingnet(preds, labels, K, golden_pct=golden_pct, seeds=seeds)
        pts.append({"task": ds, "type": "classification",
                    "dispersion": r["base_best"] - r["base_worst"], "best_base": r["base_best"],
                    "stackingnet": r["stackingnet_bca_mean"], "gain": r["gain_over_best"]})
    return pts


# ---------------- Part 2: strong-subset on paper-review ----------------
def strong_subset(topk=3, label_frac=0.01, seeds=5):
    rows = []
    for t in PR_TASKS:
        df = pd.read_excel(os.path.join(REPO, 'data', 'Research_Review', f'{t}.xlsx'))
        df = df[df['paperID'] != 'MAE']
        y = np.clip(df['label'].to_numpy(float), 1, 10)
        P = np.clip(np.vstack([df[m].to_numpy(float) for m in PR_LLMS]).T, 1, 10)
        n = len(y)
        for pool_name, selector in [("full", None), (f"top{topk}", topk)]:
            gains, disps = [], []
            for s in range(seeds):
                rng = np.random.default_rng(s)
                idx = rng.permutation(n); k = max(int(label_frac * n), P.shape[1] + 2)
                tr, te = idx[:k], idx[k:]
                if selector is None:
                    cols = list(range(P.shape[1]))
                else:
                    train_mae = [np.mean(np.abs(P[tr, j] - y[tr])) for j in range(P.shape[1])]
                    cols = list(np.argsort(train_mae)[:selector])
                Psub = P[:, cols]
                base_mae = [np.mean(np.abs(Psub[te, j] - y[te])) for j in range(Psub.shape[1])]
                pred, _, _ = fit_stackingnet(Psub[tr], y[tr], Psub[te], seed=s)
                gains.append(min(base_mae) - np.mean(np.abs(pred - y[te])))
                disps.append(max(base_mae) - min(base_mae))
            rows.append({"task": t, "pool": pool_name, "n_models": (P.shape[1] if selector is None else topk),
                         "dispersion": float(np.mean(disps)), "gain_over_best": float(np.mean(gains))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default='fairrun')
    ap.add_argument('--out', default=os.path.join(REPO, 'results', 'analysis', 'expC_dispersion_gain.json'))
    args = ap.parse_args()

    pts = cfd_points(args.log) + pr_points() + helm_points()
    reg = [p for p in pts if p['type'] == 'regression']
    clf = [p for p in pts if p['type'] == 'classification']

    print("== Part 1: dispersion vs gain ==")
    print(f"{'task':16s} {'type':12s} {'dispersion':>10s} {'best_base':>10s} {'stacknet':>9s} {'gain':>8s}")
    for p in pts:
        print(f"{p['task']:16s} {p['type']:12s} {p['dispersion']:10.3f} {p['best_base']:10.3f} "
              f"{p['stackingnet']:9.3f} {p['gain']:+8.3f}")
    for name, grp in [("regression (MAE, gain=lower)", reg), ("classification (BCA, gain=higher)", clf)]:
        d = [p['dispersion'] for p in grp]; g = [p['gain'] for p in grp]
        rho, pv = spearmanr(d, g)
        print(f"  Spearman rho(dispersion, gain) [{name}] = {rho:+.3f} (p={pv:.3f}, n={len(grp)})")

    ss = strong_subset()
    print("\n== Part 2: strong-subset (paper-review), gain of StackingNet over best base ==")
    print(f"{'task':12s} {'pool':6s} {'n':>2s} {'dispersion':>10s} {'gain_over_best':>14s}")
    for r in ss:
        print(f"{r['task']:12s} {r['pool']:6s} {r['n_models']:2d} {r['dispersion']:10.3f} {r['gain_over_best']:+14.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"points": pts, "strong_subset": ss}, open(args.out, 'w'), indent=1)
    print(f"\nSaved -> {args.out}")


if __name__ == '__main__':
    main()
