"""
Exp B3 -- Real inter-model error dependence by model LINEAGE, using the full HELM
model set (66 models per dataset) that includes same-organisation / different-version
families. Addresses R1.3: shows that the conditional-independence assumption is
violated *structurally* by shared lineage -- models from the same organisation (and
different versions of the same model family) make highly correlated errors, far more
than models from different organisations.

Data: data/helm_full/<dataset>/<dataset>:model=<org_model>,data_augmentation=canonical.csv
      (column of class predictions in {-1,0,1,...}; -1 = abstain) and matching
      *_ground_truth.csv (identical across models; same instances/order).

Outputs:
  results/analysis/expB3_helm_families.json
  results/analysis/figures/si_helm_corr_heatmap.png  (boolq, models ordered by org)
"""
import os, glob, re, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(HERE)
ROOT=os.path.join(REPO,'data','helm_full')
# boolq + civil_comments have a single shared 5000/clean instance set; imdb/mmlu/raft
# store ragged per-subtask instance counts that are not alignable across models.
DATASETS=['boolq','civil_comments']
HEATMAP_DS='civil_comments'

def org(model):  # family = organisation prefix before first underscore
    return model.split('_')[0]

def load(dataset):
    d=os.path.join(ROOT,dataset)
    raw={}; gt=None
    for f in sorted(glob.glob(os.path.join(d,'*data_augmentation=canonical.csv'))):
        if 'ground_truth' in f: continue
        m=re.search(r'model=([^,]+),', f).group(1)
        raw[m]=np.loadtxt(f)
        if gt is None:
            gtf=f.replace('canonical.csv','canonical_ground_truth.csv')
            if os.path.exists(gtf): gt=np.loadtxt(gtf)
    L=len(gt)
    preds={}
    for m,v in raw.items():
        if len(v)==L: preds[m]=v
        elif len(v)%L==0: preds[m]=v[:L]      # 15000 = 3 concatenated blocks; first block is canonical
    dropped=[m for m in raw if m not in preds]
    if dropped: print(f"  [{dataset}] dropped {len(dropped)} unaligned models")
    return preds, gt

summary={}
for ds in DATASETS:
    preds,gt=load(ds)
    if gt is None: print(ds,'no gt'); continue
    models=list(preds); n=len(models)
    # error indicator (abstain -1 counts as error)
    E=np.vstack([(preds[m]!=gt).astype(float) for m in models])  # n_models x n_inst
    # pairwise Pearson correlation of error vectors
    C=np.corrcoef(E)
    orgs=[org(m) for m in models]
    within=[]; cross=[]
    for i in range(n):
        for j in range(i+1,n):
            (within if orgs[i]==orgs[j] else cross).append(C[i,j])
    # per-organisation within-family correlation (orgs with >=3 aligned versions)
    from collections import defaultdict
    byorg=defaultdict(list)
    for i,m in enumerate(models): byorg[org(m)].append(i)
    perorg={}
    for o,idx in byorg.items():
        if len(idx)>=3:
            wc=[C[i,j] for a,i in enumerate(idx) for j in idx[a+1:]]
            perorg[o]={'n_versions':len(idx),'mean_within_corr':float(np.nanmean(wc))}
    summary[ds]={'n_models':n,'n_inst':int(E.shape[1]),
                 'mean_within_family_corr':float(np.nanmean(within)),
                 'mean_cross_family_corr':float(np.nanmean(cross)),
                 'n_within_pairs':len(within),'n_cross_pairs':len(cross),
                 'per_org_within':perorg}
    print(f"{ds:15s} n={n:3d} | within-family corr {np.nanmean(within):.3f} (N={len(within)}) | "
          f"cross-family corr {np.nanmean(cross):.3f} (N={len(cross)}) | ratio {np.nanmean(within)/max(np.nanmean(cross),1e-6):.2f}x")
    for o,v in sorted(perorg.items(), key=lambda kv:-kv[1]['mean_within_corr']):
        print(f"     {o:14s} {v['n_versions']:2d} versions  mean within-corr {v['mean_within_corr']:.3f}")

# ---- heatmap, models ordered by organisation ----
preds,gt=load(HEATMAP_DS); models=list(preds)
order=sorted(range(len(models)), key=lambda i: (org(models[i]), models[i]))
models_o=[models[i] for i in order]; orgs_o=[org(m) for m in models_o]
E=np.vstack([(preds[m]!=gt).astype(float) for m in models_o]); C=np.corrcoef(E)
fig,ax=plt.subplots(figsize=(9,8))
im=ax.imshow(C, vmin=0, vmax=1, cmap='magma')
# family boundary lines + labels
bounds=[0];
for k in range(1,len(orgs_o)):
    if orgs_o[k]!=orgs_o[k-1]: bounds.append(k)
bounds.append(len(orgs_o))
for b in bounds[1:-1]:
    ax.axhline(b-0.5,color='cyan',lw=0.6); ax.axvline(b-0.5,color='cyan',lw=0.6)
ticks=[(bounds[i]+bounds[i+1])/2-0.5 for i in range(len(bounds)-1)]
labs=[orgs_o[bounds[i]] for i in range(len(bounds)-1)]
ax.set_xticks(ticks); ax.set_xticklabels(labs, rotation=90, fontsize=8)
ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=8)
ax.set_title(f'{HEATMAP_DS}: pairwise error-indicator correlation among HELM models\n(grouped by organisation; bright on-diagonal blocks = same-lineage models err together)', fontsize=9)
plt.colorbar(im, fraction=0.046, pad=0.04, label='error correlation')
plt.tight_layout()
os.makedirs(os.path.join(REPO,'results','analysis','figures'),exist_ok=True)
fp=os.path.join(REPO,'results','analysis','figures','si_helm_corr_heatmap.png')
plt.savefig(fp,dpi=200,bbox_inches='tight'); print('saved heatmap ->',fp)
json.dump(summary,open(os.path.join(REPO,'results','analysis','expB3_helm_families.json'),'w'),indent=1)
