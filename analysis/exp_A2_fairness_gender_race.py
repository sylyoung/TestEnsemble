"""
Exp A2 -- Independent re-computation + verification of the CFD group-fairness battery,
disaggregated by GENDER (2 groups) and RACE/ETHNICITY (6 groups) separately.

Purpose (R1 Q1 & Q2, Advanced Science revision):
  * Verify exp_A numbers by recomputing straight from the per-attribute prediction
    files results/CFD/<run>/<attr>_predictions.xlsx (orig_index,Gender,Ethnicity,label,
    10 base models, StackingNet), all in the normalised [0,1] label space.
  * Report, for best base / unweighted Averaging / StackingNet, on EACH axis:
      - worst-group MAE                (minimax / Rawlsian; Martinez 2020, Diana 2021)
      - max-min group-MAE gap          (reviewer's explicit ask)
      - Wasserstein-1 disparity        (max pairwise W1 between group signed-error
                                        distributions; Chzhen 2020) -- a *distributional*
                                        fairness metric that goes beyond mean error
      - statistical-parity gap         (max-min of per-group MEAN PREDICTION) and the
                                        same gap on the LABELS, so we can say whether the
                                        model AMPLIFIES the group difference already in the
                                        reference ratings (demographic-parity, reported
                                        honestly: labels legitimately differ by group)
  * Nonparametric bootstrap 95% CI (1,000 resamples, paired) on StackingNet - best base
    worst-group MAE, i.e. the confidence that the ensemble beats the single best model.
Outputs: results/analysis/expA2_gender_race.json  (+ console summary)
"""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
ATTRS = ['afraid','angry','attractive','babyfaced','disgusted','feminine','happy',
         'masculine','sad','surprised','threatening','trustworthy','unusual']
BASE = ['BLIP','DeepSeek-VL','H2OVL','InternVL-2','LLaVA','Molmo','Paligemma','Phi-3.5','SAIL-VL','SmolVLM']
RNG = np.random.default_rng(0)

def worst_gap(err_abs, gid, groups):
    v = np.array([err_abs[gid==g].mean() for g in groups if (gid==g).sum()>0])
    return float(v.max()), float(v.max()-v.min())

def w1_disp(err_signed, gid, groups):
    pres=[g for g in groups if (gid==g).sum()>0]; w=0.0
    for i in range(len(pres)):
        for j in range(i+1,len(pres)):
            w=max(w, float(wasserstein_distance(err_signed[gid==pres[i]], err_signed[gid==pres[j]])))
    return w

def sp_gap(vals, gid, groups):  # statistical-parity: max-min of per-group MEAN of vals
    m=np.array([vals[gid==g].mean() for g in groups if (gid==g).sum()>0])
    return float(m.max()-m.min())

def boot_ci(d_abs_sn, d_abs_bb, gid, groups, n=1000):
    """CI on (worst-group MAE StackingNet) - (worst-group MAE best base)."""
    idx=np.arange(len(gid)); diffs=[]
    for _ in range(n):
        s=RNG.choice(idx, len(idx), replace=True)
        w_sn,_=worst_gap(d_abs_sn[s], gid[s], groups); w_bb,_=worst_gap(d_abs_bb[s], gid[s], groups)
        diffs.append(w_sn-w_bb)
    lo,hi=np.percentile(diffs,[2.5,97.5]); return float(np.mean(diffs)),float(lo),float(hi)

run=sorted(glob.glob(os.path.join(REPO,'results','CFD','*')))[0]
out={}
# verification cross-check against exp_A
try: ref=json.load(open(os.path.join(REPO,'results','analysis','expA_fairness_battery.json')))
except Exception: ref=None
print(f"{'attr':11s} | axis  | bb_wg  sn_wg  | bb_gap sn_gap | bb_W1  sn_W1 | SNbestΔ[95%CI]      | verif")
for attr in ATTRS:
    fp=os.path.join(run,f'{attr}_predictions.xlsx')
    df=pd.read_excel(fp); df['Gender']=df['Gender'].astype(str).str.strip(); df['Ethnicity']=df['Ethnicity'].astype(str).str.strip()
    y=df['label'].to_numpy(float)
    preds={m:df[m].to_numpy(float) for m in BASE+['StackingNet']}
    preds['Averaging']=np.mean([df[m].to_numpy(float) for m in BASE],axis=0)
    best=min(BASE, key=lambda m: np.abs(preds[m]-y).mean())
    rec={'best_base':best,'axes':{}}
    for axis,col,groups in [('gender','Gender',sorted(df['Gender'].unique())),
                            ('ethnicity','Ethnicity',sorted(df['Ethnicity'].unique()))]:
        gid=df[col].to_numpy()
        d={}
        for m in [best,'Averaging','StackingNet']:
            es=preds[m]-y; ea=np.abs(es)
            wg,gp=worst_gap(ea,gid,groups)
            d[m]={'worst_group_mae':wg,'max_min_gap':gp,'w1_disp':w1_disp(es,gid,groups),
                  'sp_gap_pred':sp_gap(preds[m],gid,groups)}
        sp_label=sp_gap(y,gid,groups)
        mean,lo,hi=boot_ci(np.abs(preds['StackingNet']-y),np.abs(preds[best]-y),gid,groups)
        rec['axes'][axis]={'methods':d,'sp_gap_label':sp_label,'ci_sn_minus_best_worstmae':[mean,lo,hi]}
        bb,sn=d[best],d['StackingNet']
        vmark=''
        if axis=='ethnicity' and ref and attr in ref:
            try: vmark=f"json_sn_wg={ref[attr]['axes']['ethnicity']['per_method']['StackingNet']['worst_group_mae']:.3f}"
            except Exception: vmark='n/a'
        print(f"{attr:11s} | {axis[:5]:5s} | {bb['worst_group_mae']:.3f}  {sn['worst_group_mae']:.3f} | "
              f"{bb['max_min_gap']:.3f}  {sn['max_min_gap']:.3f} | {bb['w1_disp']:.3f}  {sn['w1_disp']:.3f} | "
              f"{mean:+.3f}[{lo:+.3f},{hi:+.3f}] | {vmark}")
    out[attr]=rec
json.dump(out, open(os.path.join(REPO,'results','analysis','expA2_gender_race.json'),'w'), indent=1)
print("\nsaved -> results/analysis/expA2_gender_race.json")
