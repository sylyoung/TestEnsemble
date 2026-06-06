"""
Exp E -- "Can the AI ensemble predict the peer consensus as well as, or better than,
a single human reviewer?" (Reviewer 1 Comment 4 / Reviewer 2 Comment 1, redesigned.)

Protocol (one bootstrap iteration), per paper with >=3 human reviewer ratings:
  * Randomly designate ONE reviewer as the "alignment target" t. This single rating is
    the ONLY human supervision either side gets.
  * The REMAINING reviewers form the held-out ground truth  G = mean(other ratings)
    (>= 2 reviewers).
  * Single-human predictor: use the target rating t to predict G.  error = |t - G|.
  * AI predictors: train StackingNet on a small labelled subset of papers using their
    target rating t as the label (exactly the signal a single human gives), then predict
    every test paper.  error = |pred - G|.  Each individual LLM and unweighted Averaging
    are scored against the same G.
Both the single human and the AI are thus judged as predictors of the SAME external
target (the consensus of the OTHER reviewers), so the comparison is apples-to-apples and
does not suffer the self-inclusion problem of scoring a reviewer against a consensus that
contains their own score.

We repeat over B bootstrap resamples (random target reviewer + random train/test split)
and report mean +/- std. If StackingNet's error is below the single human's, the ensemble
predicts the peer consensus better than an individual reviewer, i.e. it reduces the
idiosyncratic variance of a single human judgement.

Outputs: results/analysis/expE_reviewer_target.json
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_D_human_loo import extract_rating, PR_LLMS
from exp_B2_controlled_degradation import fit_stackingnet

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = ['ICLR2025', 'ICLR2024', 'NeurIPS2024', 'NeurIPS2023']
B = 50              # bootstrap resamples
LABEL_FRAC = 0.01   # few-shot supervision, as in the paper

def load(task):
    df = pd.read_excel(os.path.join(REPO,'data','Research_Review',f'{task}.xlsx'), dtype={'paperID':str})
    df = df[df['paperID']!='MAE'].reset_index(drop=True)
    rev = pd.read_excel(os.path.join(REPO,'data','Paper_Review',f'{task}_reviews.xlsx'), dtype={'paperID':str})
    rev['r'] = rev['rating'].apply(extract_rating); rev = rev.dropna(subset=['r'])
    reviews = {pid:g['r'].tolist() for pid,g in rev.groupby('paperID')}
    # keep papers with LLM preds AND >=3 reviews
    keep = [i for i,p in enumerate(df['paperID']) if len(reviews.get(p,[]))>=3]
    df = df.iloc[keep].reset_index(drop=True)
    P = np.clip(np.vstack([df[m].to_numpy(float) for m in PR_LLMS]).T, 1, 10)   # (n,M)
    R = [np.asarray(reviews[p],float) for p in df['paperID']]                   # per-paper ratings
    return P, R

def one_boot(P, R, rng):
    n = len(R)
    t = np.array([rng.choice(r) for r in R])                       # target reviewer per paper
    G = np.array([ (r.sum()-ti)/(len(r)-1) for r,ti in zip(R,t) ]) # mean of OTHER reviewers
    k = max(int(LABEL_FRAC*n), P.shape[1]+2)
    idx = rng.permutation(n); tr, te = idx[:k], idx[k:]
    sn,_,_ = fit_stackingnet(P[tr], t[tr], P[te], seed=int(rng.integers(1e6)))  # trained on single-reviewer target
    err = {'Human (single reviewer)': np.abs(t[te]-G[te]).mean(),
           'Averaging': np.abs(P[te].mean(1)-G[te]).mean(),
           'StackingNet': np.abs(sn-G[te]).mean()}
    for j,m in enumerate(PR_LLMS): err[m] = np.abs(P[te,j]-G[te]).mean()
    return err

out={}
for task in TASKS:
    P,R = load(task)
    rng = np.random.default_rng(0)
    runs=[one_boot(P,R,rng) for _ in range(B)]
    methods=list(runs[0])
    res={m:{'mae_mean':float(np.mean([r[m] for r in runs])),
            'mae_std':float(np.std([r[m] for r in runs]))} for m in methods}
    out[task]={'n_papers':len(R),'results':res}
    h=res['Human (single reviewer)']; sn=res['StackingNet']
    print(f"\n=== {task} (n={len(R)} papers, B={B}) ===   MAE vs consensus of OTHER reviewers")
    for m in ['Human (single reviewer)','Averaging','StackingNet']+PR_LLMS:
        print(f"  {m:26s} {res[m]['mae_mean']:.3f} +/- {res[m]['mae_std']:.3f}")
    print(f"  --> StackingNet beats single human by {h['mae_mean']-sn['mae_mean']:+.3f} "
          f"({'YES' if sn['mae_mean']<h['mae_mean'] else 'no'})")
json.dump(out, open(os.path.join(REPO,'results','analysis','expE_reviewer_target.json'),'w'), indent=1)
print("\nsaved -> results/analysis/expE_reviewer_target.json")
