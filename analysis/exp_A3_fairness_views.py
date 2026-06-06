"""
Exp A3 -- Comprehensive group-fairness battery for the CFD attribute-rating task,
organised by the metric "views" defined in the literature we cite, computed for
GENDER (2 groups) and RACE/ethnicity (6 groups) separately, for the best single
base model, unweighted Averaging, and StackingNet, with bootstrap 95% CIs.

Views (each grounded in a cited paper):
  V1 ACCURACY PARITY / accuracy disparity in regression [Chi 2021; Agarwal 2019 BGL;
     Diana 2021 minimax]:
       - worst-group MAE  (minimax / bounded-group-loss)
       - max-min group-MAE gap
       - group-MAE coefficient of variation (dispersion) [Castelnovo 2022]
       (bootstrap 95% CI on StackingNet - best base for worst-group MAE and gap)
  V2 DEMOGRAPHIC / STATISTICAL PARITY for regression [Chzhen 2020]:
       - W1(pred): max pairwise Wasserstein-1 distance between group PREDICTION
         distributions  (DP disparity of the model's outputs)
       - W1(label): same on the reference labels  (inherent group difference)
       - amplification = W1(pred) - W1(label)  (<=0 means the model does not widen
         the group difference already in the labels)
  V3 BIAS AMPLIFICATION [Zhao 2017; Wang & Russakovsky 2021]:
       - gap(method) - mean gap(base models)  (directional, vs the average base model)

Inputs : results/CFD/<run>/<attr>_predictions.xlsx  (normalised [0,1] label space)
Outputs: results/analysis/expA3_fairness_views.json
         results/analysis/figures/si_fairness_dp.png   (V2 figure, race + gender)
"""
import os, glob, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(HERE)
ATTRS=['afraid','angry','attractive','babyfaced','disgusted','feminine','happy',
       'masculine','sad','surprised','threatening','trustworthy','unusual']
BASE=['BLIP','DeepSeek-VL','H2OVL','InternVL-2','LLaVA','Molmo','Paligemma','Phi-3.5','SAIL-VL','SmolVLM']
RNG=np.random.default_rng(0)
run=sorted(glob.glob(os.path.join(REPO,'results','CFD','*')))[0]

def grp_mae(ea,gid,gs): return np.array([ea[gid==g].mean() for g in gs if (gid==g).sum()>0])
def worst(ea,gid,gs): v=grp_mae(ea,gid,gs); return float(v.max())
def gap(ea,gid,gs): v=grp_mae(ea,gid,gs); return float(v.max()-v.min())
def cov(ea,gid,gs): v=grp_mae(ea,gid,gs); return float(v.std()/v.mean()) if v.mean()>0 else 0.0
def maxw1(v,gid,gs):
    pr=[g for g in gs if (gid==g).sum()>0]; w=0.0
    for i in range(len(pr)):
        for j in range(i+1,len(pr)):
            w=max(w,float(wasserstein_distance(v[gid==pr[i]],v[gid==pr[j]])))
    return w
def ci(es_sn,es_bb,gid,gs,fn,n=1000):
    idx=np.arange(len(gid)); d=[]
    for _ in range(n):
        s=RNG.choice(idx,len(idx),True)
        d.append(fn(np.abs(es_sn[s]),gid[s],gs)-fn(np.abs(es_bb[s]),gid[s],gs))
    return float(np.mean(d)),float(np.percentile(d,2.5)),float(np.percentile(d,97.5))

out={}
fig_data={'race':{'lab':[],'bb':[],'sn':[]}, 'gender':{'lab':[],'bb':[],'sn':[]}}
for attr in ATTRS:
    df=pd.read_excel(f'{run}/{attr}_predictions.xlsx')
    df['Gender']=df['Gender'].astype(str).str.strip(); df['Ethnicity']=df['Ethnicity'].astype(str).str.strip()
    y=df['label'].to_numpy(float)
    P={m:df[m].to_numpy(float) for m in BASE+['StackingNet']}; P['Averaging']=np.mean([df[m].to_numpy(float) for m in BASE],0)
    best=min(BASE,key=lambda m:np.abs(P[m]-y).mean())
    rec={'best_base':best,'axes':{}}
    for axis,col in [('race','Ethnicity'),('gender','Gender')]:
        gid=df[col].to_numpy(); gs=sorted(df[col].unique())
        methods={}
        base_gaps=[gap(np.abs(P[m]-y),gid,gs) for m in BASE]; mean_base_gap=float(np.mean(base_gaps))
        for m in [best,'Averaging','StackingNet']:
            es=P[m]-y; ea=np.abs(es)
            methods[m]={'worst':worst(ea,gid,gs),'gap':gap(ea,gid,gs),'cov':cov(ea,gid,gs),
                        'w1_pred':maxw1(P[m],gid,gs),'bias_amp':gap(ea,gid,gs)-mean_base_gap}
        w1_label=maxw1(y,gid,gs)
        ci_worst=ci(P['StackingNet']-y,P[best]-y,gid,gs,worst)
        ci_gap=ci(P['StackingNet']-y,P[best]-y,gid,gs,gap)
        rec['axes'][axis]={'methods':methods,'w1_label':w1_label,
                           'ci_worst_sn_minus_bb':ci_worst,'ci_gap_sn_minus_bb':ci_gap}
        fig_data[axis]['lab'].append(w1_label); fig_data[axis]['bb'].append(methods[best]['w1_pred']); fig_data[axis]['sn'].append(methods['StackingNet']['w1_pred'])
    out[attr]=rec
os.makedirs(os.path.join(REPO,'results','analysis','figures'),exist_ok=True)
json.dump(out,open(os.path.join(REPO,'results','analysis','expA3_fairness_views.json'),'w'),indent=1)

# ---- summary by view ----
def tally(axis,key,better='lt'):
    c=0
    for a in ATTRS:
        m=out[a]['axes'][axis]['methods']; bb=out[a]['best_base']
        s=m['StackingNet'][key]; b=m[bb][key]
        c+= (s<b) if better=='lt' else (s>b)
    return c
print("VIEW 1  Accuracy parity (worst-group MAE / gap), StackingNet better than best base:")
for ax in ['race','gender']:
    cw=sum(out[a]['axes'][ax]['ci_worst_sn_minus_bb'][2]<0 for a in ATTRS)
    print(f"  {ax:6s}: worst-MAE improved {tally(ax,'worst')}/13 ({cw}/13 CI<0) ; gap improved {tally(ax,'gap')}/13")
print("VIEW 2  Demographic parity (Wasserstein-1 pred vs label):")
for ax in ['race','gender']:
    noamp=sum(out[a]['axes'][ax]['methods']['StackingNet']['w1_pred']<=out[a]['axes'][ax]['w1_label']+1e-9 for a in ATTRS)
    al=np.mean(fig_data[ax]['lab']); ab=np.mean(fig_data[ax]['bb']); asn=np.mean(fig_data[ax]['sn'])
    print(f"  {ax:6s}: StackingNet W1(pred)<=W1(label) {noamp}/13 ; avg W1 label {al:.3f} best-base {ab:.3f} StackingNet {asn:.3f}")
print("VIEW 3  Bias amplification (gap vs mean base), StackingNet negative (reduces) on:")
for ax in ['race','gender']:
    neg=sum(out[a]['axes'][ax]['methods']['StackingNet']['bias_amp']<0 for a in ATTRS)
    print(f"  {ax:6s}: {neg}/13 attributes")

# ---- Figure S8: demographic parity (W1) label vs best base vs StackingNet ----
fig,axes=plt.subplots(1,2,figsize=(12,4.2))
x=np.arange(13); w=0.27
for k,ax in enumerate(['race','gender']):
    a=axes[k]
    a.bar(x-w, fig_data[ax]['lab'], w, label='Human labels', color='#888888')
    a.bar(x,   fig_data[ax]['bb'],  w, label='Best base model', color='#E1812C')
    a.bar(x+w, fig_data[ax]['sn'],  w, label='StackingNet', color='#3A75C4')
    a.set_xticks(x); a.set_xticklabels(ATTRS, rotation=60, ha='right', fontsize=8)
    a.set_ylabel('Wasserstein-1 between group\nprediction distributions'); a.set_title(f'{"Race/ethnicity (6 groups)" if ax=="race" else "Gender (2 groups)"}')
    a.legend(fontsize=8, frameon=False)
plt.tight_layout()
fp=os.path.join(REPO,'results','analysis','figures','si_fairness_dp.png')
plt.savefig(fp,dpi=200,bbox_inches='tight'); print('saved figure ->',fp)
