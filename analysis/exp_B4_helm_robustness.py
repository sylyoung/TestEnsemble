"""
Exp B4 -- Robustness of StackingNet when conditional independence is violated by a
real, same-lineage CORRELATED CLUSTER (Reviewer 1 Comment 3).

We start from a DIVERSE pool (the single strongest model from each distinct
organisation, i.e. independent lineages) and then contaminate it with a growing
cluster of additional versions from ONE organisation. Members of one organisation
share architecture, pre-training data and tuning recipe, so their errors are
strongly correlated -- exactly the situation in which the conditional-independence
assumption behind unweighted averaging fails.

To show the effect is not specific to one company or one dataset, we repeat the
sweep for four distinct lineages -- OpenAI (GPT), Cohere (Command), AI21 (Jurassic)
and Meta (Llama) -- on two HELM datasets (boolq, civil_comments). For every cluster
size we record unweighted Averaging (majority vote) and the supervised StackingNet,
plus the best single model in the pool.

Expectation: as the correlated cluster grows, majority-vote Averaging is dragged
toward the cluster's shared answers and degrades, whereas StackingNet down-weights
the redundant members and holds its accuracy.

Outputs: results/analysis/expB4_helm_robustness.json
"""
import os, glob, re, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.metrics import balanced_accuracy_score
from stacknet_clf import run_helm_stackingnet

HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(HERE)
RNG=np.random.default_rng(0)
def org(m): return m.split('_')[0]

# distinct single-organisation lineages (each shares one architecture/recipe family)
FAMILIES=['openai','cohere','ai21','meta']
FAM_LABEL={'openai':'OpenAI (GPT)','cohere':'Cohere (Command)','ai21':'AI21 (Jurassic)','meta':'Meta (Llama)'}
DATASETS=['boolq','civil_comments']
MAXK=8

def load(ds):
    d=os.path.join(REPO,'data','helm_full',ds)
    # civil_comments is split into demographic slices, each a different instance set;
    # keep only demographic=all so every model is scored on the same shared instances.
    tag='demographic=all,' if ds=='civil_comments' else ''
    gtf=[f for f in glob.glob(os.path.join(d,'*canonical_ground_truth.csv')) if tag in f]
    gt=np.loadtxt(gtf[0]); L=len(gt)
    models=[]; P=[]
    for f in sorted(glob.glob(os.path.join(d,'*data_augmentation=canonical.csv'))):
        if 'ground_truth' in f or (tag and tag not in f): continue
        v=np.loadtxt(f)
        if len(v)==L: pass
        elif len(v)%L==0: v=v[:L]
        else: continue
        v=v.astype(int); miss=v<0
        if miss.any(): v[miss]=RNG.integers(0,int(gt.max())+1,miss.sum())
        models.append(re.search(r'model=([^,]+),',f).group(1)); P.append(v)
    return np.vstack(P), gt.astype(int), models

def maj(P,labels,idx,K):
    sub=P[idx]; pred=np.array([np.bincount(sub[:,j],minlength=K).argmax() for j in range(P.shape[1])])
    return 100*balanced_accuracy_score(labels,pred)

out={}
for ds in DATASETS:
    P,labels,models=load(ds); K=int(labels.max())+1; M=len(models)
    # rank by balanced accuracy: civil_comments is class-imbalanced, so raw accuracy
    # would pick majority-class predictors that are actually at chance.
    acc=np.array([balanced_accuracy_score(labels,P[i]) for i in range(M)])
    byorg={}
    for i,m in enumerate(models): byorg.setdefault(org(m),[]).append(i)
    print(f"\n##### {ds}: {M} models, K={K}")
    out[ds]={'K':K,'families':{}}
    for fam in FAMILIES:
        if fam not in byorg or len(byorg[fam])<3:
            print(f"  {fam}: <3 aligned versions -- skipped"); continue
        # fixed diverse base: the 5 strongest models from 5 OTHER distinct organisations
        base=sorted([max(ix,key=lambda i:acc[i]) for o,ix in byorg.items() if o!=fam],
                    key=lambda i:-acc[i])[:5]
        base_best=float(acc[base].max()*100)
        # correlated cluster: same-lineage versions from this organisation, strongest first
        cluster=sorted(byorg[fam],key=lambda i:-acc[i])[:MAXK]
        ks=list(range(0,len(cluster)+1)); avg=[]; sn=[]; sne=[]
        for k in ks:
            idx=sorted(set(base)|set(cluster[:k]))
            r=run_helm_stackingnet(P,labels,K,golden_pct=5,seeds=5,subset_cols=idx,verbose=False)
            avg.append(maj(P,labels,idx,K)); sn.append(r['stackingnet_bca_mean']); sne.append(r['stackingnet_bca_std'])
        out[ds]['families'][fam]={'k':ks,'averaging':avg,'stackingnet':sn,'stackingnet_std':sne,
                                  'base_best':base_best,'n_versions':len(cluster)}
        print(f"  {FAM_LABEL[fam]:18s} +{len(cluster)} versions (base_best {base_best:.2f}): "
              f"Averaging {avg[0]:.2f}->{avg[-1]:.2f} | StackingNet {sn[0]:.2f}->{sn[-1]:.2f}")
json.dump(out, open(os.path.join(REPO,'results','analysis','expB4_helm_robustness.json'),'w'), indent=1)
print("\nsaved -> results/analysis/expB4_helm_robustness.json")
