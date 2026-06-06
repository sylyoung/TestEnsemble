"""
Exp D -- Leave-one-reviewer-out (LOO) fair human vs AI comparison.
=================================================================
Answers Reviewer 1 comment 4 and Reviewer 2 comment 1: the manuscript scores each
individual human reviewer against a consensus that INCLUDES that reviewer. Self-inclusion
pulls the consensus toward the reviewer, so |rating - consensus| understates a single
human's deviation -- an unfair, by-construction comparison.

Fair protocol (apples-to-apples): for every paper and every reviewer i, form the
leave-one-out consensus C_{-i} = mean rating of the OTHER reviewers, and score BOTH the
held-out human i and each AI predictor against the SAME C_{-i} (neither is part of it):
    human_dev(p) = mean_i |r_i - C_{-i}|
    ai_dev(p, m) = mean_i |pred_m(p) - C_{-i}|
StackingNet is trained few-shot (1% labels = full consensus) on train papers and evaluated
on held-out test papers with >=3 reviews. We report both the LOO ("fair") and the original
full-consensus comparison for contrast.

Interpretation we will use in the reply: any aggregate (the human consensus, or the AI
ensemble) is closer to the consensus than a single noisy rater BY CONSTRUCTION. So a small
AI-vs-human gap reflects variance reduction / consensus alignment, NOT superior scientific
judgment. We soften the "surpass expert judgment" claim accordingly.

Usage:
  python analysis/exp_D_human_loo.py
"""
import os, sys, json, argparse, ast, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from exp_B2_controlled_degradation import fit_stackingnet  # real StackingRegression fit

PR_TASKS = ['ICLR2025', 'ICLR2024', 'NeurIPS2024', 'NeurIPS2023']
PR_LLMS = ['deepseek', 'doubao', 'gemini-2', 'gpt-5', 'gpt4o', 'qwent_turbo']


def extract_rating(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            o = ast.literal_eval(v)
            if isinstance(o, dict) and 'value' in o:
                return float(o['value'])
        except Exception:
            pass
        m = re.search(r'-?\d+(\.\d+)?', v)
        return float(m.group()) if m else np.nan
    return np.nan


def load_reviews(task):
    """paperID -> list of individual human ratings (>=1)."""
    rev = pd.read_excel(os.path.join(REPO, 'data', 'Paper_Review', f'{task}_reviews.xlsx'),
                        dtype={'paperID': str})
    rev['r'] = rev['rating'].apply(extract_rating)
    rev = rev.dropna(subset=['r'])
    return {pid: g['r'].tolist() for pid, g in rev.groupby('paperID')}


def loo_devs(ratings, pred):
    """Per-paper mean LOO deviation for the human raters and for one AI prediction.
    Returns (human_dev, ai_dev) or (None, None) if <3 reviews."""
    R = np.asarray(ratings, float)
    K = len(R)
    if K < 3:
        return None, None
    tot = R.sum()
    Cmi = (tot - R) / (K - 1)              # leave-one-out consensus per reviewer
    human_dev = float(np.mean(np.abs(R - Cmi)))
    ai_dev = float(np.mean(np.abs(pred - Cmi)))
    return human_dev, ai_dev


def analyse(task, label_frac=0.01, seeds=5):
    df = pd.read_excel(os.path.join(REPO, 'data', 'Research_Review', f'{task}.xlsx'),
                       dtype={'paperID': str})
    df = df[df['paperID'] != 'MAE'].reset_index(drop=True)
    pid = df['paperID'].to_numpy()
    y = np.clip(df['label'].to_numpy(float), 1, 10)        # full consensus
    P = np.clip(np.vstack([df[m].to_numpy(float) for m in PR_LLMS]).T, 1, 10)  # (n, M)
    reviews = load_reviews(task)

    methods = PR_LLMS + ['Averaging', 'StackingNet']
    acc = {m: {"loo": [], "full": []} for m in methods}
    human_loo, human_full = [], []
    n_eval_papers = 0

    for s in range(seeds):
        rng = np.random.default_rng(s)
        n = len(y); k = max(int(label_frac * n), P.shape[1] + 2)
        idx = rng.permutation(n); tr, te = idx[:k], idx[k:]
        sn_pred, _, _ = fit_stackingnet(P[tr], y[tr], P[te], seed=s)
        preds_by_method = {m: P[te, j] for j, m in enumerate(PR_LLMS)}
        preds_by_method['Averaging'] = P[te].mean(axis=1)
        preds_by_method['StackingNet'] = sn_pred

        for local, p in enumerate(te):
            R = reviews.get(pid[p])
            if R is None or len(R) < 3:
                continue
            if s == 0:
                n_eval_papers += 1
            full_cons = float(np.mean(R))
            hd, _ = loo_devs(R, 0.0)
            human_loo.append(hd)
            human_full.append(float(np.mean(np.abs(np.asarray(R) - full_cons))))
            for m in methods:
                pm = float(preds_by_method[m][local])
                _, ad = loo_devs(R, pm)
                acc[m]["loo"].append(ad)
                acc[m]["full"].append(abs(pm - full_cons))

    out = {"task": task, "n_eval_papers": n_eval_papers,
           "human_loo_mae": float(np.mean(human_loo)), "human_full_mae": float(np.mean(human_full)),
           "methods": {}}
    for m in methods:
        out["methods"][m] = {"loo_mae": float(np.mean(acc[m]["loo"])),
                             "full_mae": float(np.mean(acc[m]["full"]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(REPO, 'results', 'analysis', 'expD_human_loo.json'))
    args = ap.parse_args()

    res = {}
    print(f"{'task':12s} {'#papers':>7s} | {'HUMAN_full':>10s} {'HUMAN_loo':>9s} | "
          f"{'SN_full':>8s} {'SN_loo':>7s} | {'Avg_loo':>7s} {'bestLLM_loo':>11s}")
    for t in PR_TASKS:
        r = analyse(t)
        res[t] = r
        best_llm = min(PR_LLMS, key=lambda m: r['methods'][m]['loo_mae'])
        print(f"{t:12s} {r['n_eval_papers']:7d} | {r['human_full_mae']:10.3f} {r['human_loo_mae']:9.3f} | "
              f"{r['methods']['StackingNet']['full_mae']:8.3f} {r['methods']['StackingNet']['loo_mae']:7.3f} | "
              f"{r['methods']['Averaging']['loo_mae']:7.3f} {r['methods'][best_llm]['loo_mae']:11.3f} ({best_llm})")

    # pooled summary
    print("\nInterpretation: compare HUMAN_loo vs SN_loo (fair). HUMAN_full is the manuscript's"
          "\nself-inclusion number (understates single-human deviation).")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, 'w'), indent=1)
    print(f"Saved -> {args.out}")


if __name__ == '__main__':
    main()
