"""
Exp C2 -- "How much improvement remains when the base-model pool is uniformly strong
and performance is less dispersed?" (Reviewer 1 Comment 5, the direct test.)

Using the full HELM model set, we build, for each dataset, a UNIFORMLY-STRONG pool: the
top-K models by accuracy (so the best-to-worst spread / dispersion is small), and compare
StackingNet's gain over the best single model there against the gain on a DISPERSED pool
(top model + K-1 weak models). If StackingNet still improves over the best model when the
pool is uniformly strong, the gain is not merely pruning of weak contributors.

Datasets: boolq, civil_comments (the two HELM datasets with a single clean instance set).
"""
import os, glob, re, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.metrics import balanced_accuracy_score
from stacknet_clf import run_helm_stackingnet
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG=np.random.default_rng(0)

def load(ds):
    d=os.path.join(REPO,'data','helm_full',ds)
    gt=np.loadtxt(glob.glob(os.path.join(d,'*canonical_ground_truth.csv'))[0]); L=len(gt)
    models=[]; P=[]
    for f in sorted(glob.glob(os.path.join(d,'*data_augmentation=canonical.csv'))):
        if 'ground_truth' in f: continue
        v=np.loadtxt(f)
        if len(v)==L: pass
        elif len(v)%L==0: v=v[:L]
        else: continue
        v=v.astype(int); m=v<0
        if m.any(): v[m]=RNG.integers(0,int(gt.max())+1,m.sum())
        models.append(re.search(r'model=([^,]+),',f).group(1)); P.append(v)
    return np.vstack(P), gt.astype(int), models

for ds in ['boolq','civil_comments']:
    P,y,models=load(ds); K=int(y.max())+1
    acc=np.array([balanced_accuracy_score(y,P[i])*100 for i in range(len(P))])
    order=np.argsort(-acc)
    for poolK in [5,8]:
        strong=list(order[:poolK])                                   # uniformly strong (top-K)
        disp=[order[0]]+list(order[-(poolK-1):])                     # best + weakest (dispersed)
        rs=run_helm_stackingnet(P,y,K,golden_pct=5,seeds=5,subset_cols=strong)
        rd=run_helm_stackingnet(P,y,K,golden_pct=5,seeds=5,subset_cols=disp)
        ds_strong=acc[strong].max()-acc[strong].min(); ds_disp=acc[disp].max()-acc[disp].min()
        print(f"{ds} K={poolK}")
        print(f"  UNIFORM-STRONG pool: acc {acc[strong].min():.1f}-{acc[strong].max():.1f} (dispersion {ds_strong:.1f}) | "
              f"best {rs['base_best']:.2f}  StackingNet {rs['stackingnet_bca_mean']:.2f}  gain {rs['stackingnet_bca_mean']-rs['base_best']:+.2f}")
        print(f"  DISPERSED pool:      acc {acc[disp].min():.1f}-{acc[disp].max():.1f} (dispersion {ds_disp:.1f}) | "
              f"best {rd['base_best']:.2f}  StackingNet {rd['stackingnet_bca_mean']:.2f}  gain {rd['stackingnet_bca_mean']-rd['base_best']:+.2f}")
