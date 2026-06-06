"""
Regenerate the revision SI figures (S6, S8, S9, S10) in the manuscript's plotting
style (matches StackingNet_plot_maintext/plot: Arial 11, no top/right spines,
StackingNet=#C51B7D, Averaging=#5E6A71, etc.), with interpretability fixes.
"""
import os, glob, re, json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 11,
    "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "lines.linewidth": 1.6, "axes.linewidth": 0.8,
    "xtick.major.size": 4, "ytick.major.size": 4,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 600,
})
SN="#C51B7D"; AVG="#5E6A71"; HUM="#E69F00"; REF="#9AA0A6"; BB="#4C9F70"; LAB="#7B8794"
HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(HERE)
ANA=os.path.join(REPO,'results','analysis'); FIG=os.path.join(ANA,'figures')
def J(n): return json.load(open(os.path.join(ANA,n)))
def panel(ax,letter,x=-0.13,y=1.03): ax.text(x,y,letter.upper(),transform=ax.transAxes,fontsize=12,fontweight="bold",va="bottom")

# ---------------- Figure S6: redundancy degrades averaging but not StackingNet ----------------
d=J('expB2_degradation.json'); red=d['redundancy']
fig,ax=plt.subplots(figsize=(5.7,4.3))
b=ax; rr=red['rows']; c=[r['cluster_size'] for r in rr]
av=np.array([r['averaging_rel_mean'] for r in rr]); ave=np.array([r['averaging_rel_std'] for r in rr])
sn=np.array([r['stackingnet_rel_mean'] for r in rr]); sne=np.array([r['stackingnet_rel_std'] for r in rr])
b.axhline(1.0,ls='--',color=REF,lw=1.1)
b.text(rr[-1]['cluster_size'],1.013,'best single model',color=REF,fontsize=8.5,ha='right',va='bottom')
b.errorbar(c,av,yerr=ave,fmt='s-',color=AVG,mec='black',mew=0.5,capsize=2.5,label='Averaging')
b.errorbar(c,sn,yerr=sne,fmt='o-',color=SN,mec='black',mew=0.5,capsize=2.5,label='StackingNet')
b.set_xlabel(f"size of redundant cluster (of M = {red['M']} equally accurate models)"); b.set_ylabel('test MSE / single-model MSE')
b.set_title('Synthetic equally accurate pool (no real dataset)',fontsize=10,color='0.4',pad=8)
b.set_ylim(0,1.16)
b.legend(loc='center left',frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(FIG,'si_degradation.png'),bbox_inches='tight'); plt.close(); print('S6 ok')

# ---------------- Figure S7: where does the gain come from? dispersion vs gain ----------------
from scipy.stats import spearmanr
Cd=J('expC_dispersion_gain.json')
reg=[p for p in Cd['points'] if p['type']=='regression']
cfd=[p for p in reg if p['task'].startswith('CFD')]            # A: Chicago Face Database only
clf=[p for p in Cd['points'] if p['type']=='classification']
fig,ax=plt.subplots(1,2,figsize=(10.6,4.6))
# A: CFD regression dispersion vs gain, with trend
a=ax[0]; xd=np.array([p['dispersion'] for p in cfd]); yg=np.array([p['gain'] for p in cfd])
rho_r,_=spearmanr(xd,yg)
a.axhline(0,color=REF,lw=1.0,ls='--')
a.scatter(xd,yg,color=SN,s=36,edgecolor='black',linewidth=0.4,zorder=3)
xs=np.linspace(xd.min(),xd.max(),50); b1,b0=np.polyfit(xd,yg,1)
a.plot(xs,b1*xs+b0,color=SN,lw=1.4,alpha=0.7)
a.set_xlabel('base-model dispersion\n(worst $-$ best MAE)'); a.set_ylabel('StackingNet gain over best base (MAE)')
a.set_title(f'Chicago Face Database, 13 attributes\nSpearman $\\rho$ = {rho_r:+.2f}',fontsize=10.5); panel(a,'a')
# B: classification dispersion vs gain, every dataset labeled by name
a=ax[1]; xc=np.array([p['dispersion'] for p in clf]); yc=np.array([p['gain'] for p in clf]); tc=[p['task'] for p in clf]
rho_c,_=spearmanr(xc,yc)
NICE={'boolq':'BoolQ','entity_matching':'EntityMatching','imdb':'IMDB','legal_support':'LegalSupport',
      'lsat_qa':'LSAT','mmlu':'MMLU','raft':'RAFT','civil_comments':'CivilComments'}
OFF={'boolq':(0,11),'mmlu':(0,-13),'entity_matching':(-4,12),'imdb':(11,8),
     'legal_support':(0,-13),'lsat_qa':(16,4),'raft':(-12,13),'civil_comments':(14,12)}
a.axhline(0,color=REF,lw=1.0,ls='--')
a.scatter(xc,yc,color=AVG,marker='^',s=44,edgecolor='black',linewidth=0.4,zorder=3)
for i,t in enumerate(tc):
    dx,dy=OFF.get(t,(0,12))
    a.annotate(NICE[t],(xc[i],yc[i]),xytext=(dx,dy),textcoords='offset points',
               fontsize=7.5,color='0.25',ha='center',va='bottom' if dy>0 else 'top',
               arrowprops=dict(arrowstyle='-',color='0.6',lw=0.5))
a.margins(y=0.20)
a.set_xlabel('base-model dispersion\n(best $-$ worst balanced accuracy, pts)'); a.set_ylabel('StackingNet gain over best base (BCA pts)')
a.set_title(f'HELM, 8 datasets\nSpearman $\\rho$ = {rho_c:+.2f}',fontsize=10.5); panel(a,'b')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'si_dispersiongain.png'),bbox_inches='tight'); plt.close(); print('S7 ok')

# ---------------- Figure S8: demographic parity (Wasserstein) ----------------
# clean, well-separated palette; one shared legend ABOVE the panels so tall bars are never covered.
LABc="#BCC3CB"; BBc="#2C7FB8"; SNc="#C51B7D"
A=J('expA3_fairness_views.json'); ATTR=list(A.keys())
fig,ax=plt.subplots(1,2,figsize=(11.5,4.4))
for p,(axis,ttl) in enumerate([('race','Race/ethnicity (6 groups)'),('gender','Gender (2 groups)')]):
    a=ax[p]; x=np.arange(len(ATTR)); w=0.27
    lab=[A[at]['axes'][axis]['w1_label'] for at in ATTR]
    bb=[A[at]['axes'][axis]['methods'][A[at]['best_base']]['w1_pred'] for at in ATTR]
    sn=[A[at]['axes'][axis]['methods']['StackingNet']['w1_pred'] for at in ATTR]
    a.bar(x-w,lab,w,color=LABc,label='Human labels (reference)',edgecolor='black',linewidth=0.4)
    a.bar(x,bb,w,color=BBc,label='Best base model',edgecolor='black',linewidth=0.4)
    a.bar(x+w,sn,w,color=SNc,label='StackingNet',edgecolor='black',linewidth=0.4)
    a.set_xticks(x); a.set_xticklabels(ATTR,rotation=55,ha='right')
    a.set_ylabel('between-group prediction disparity\n(Wasserstein-1 distance)'); a.set_title(ttl)
    a.margins(y=0.18); panel(a,'ab'[p])
h,l=ax[0].get_legend_handles_labels()
fig.legend(h,l,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(0.5,1.04))
plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(os.path.join(FIG,'si_fairness_dp.png'),bbox_inches='tight'); plt.close(); print('S8 ok')

# ---------------- Figure 5 (helm-corr): error correlations violate conditional independence ----------------
# Panel A: lower-triangular heatmap of the mean error-indicator correlation between the ten
#          HELM base models, averaged over ALL EIGHT HELM classification datasets. Both axes
#          carry the same model ordering; only the lower triangle is shown (the matrix is
#          symmetric and the diagonal is the trivial self-correlation).
ORGLAB=['Meta','OpenAI','AI21','Writer','Falcon','Cohere','Microsoft','MosaicML','Anthropic','Together']  # helm_io.MODEL_NAMES order
D=J('expB_dependence.json')
helm_mats=np.stack([np.array(v['err_indicator_corr_matrix']) for v in D['helm'].values()])
C=np.nanmean(helm_mats,axis=0)
C[np.triu(np.ones_like(C,dtype=bool))]=np.nan          # show strictly lower triangle (i>j); diagonal omitted
fig,ax=plt.subplots(1,2,figsize=(12.8,5.8),gridspec_kw={'width_ratios':[1.0,1.0]})
a=ax[0]; im=a.imshow(C,vmin=0,vmax=0.45,cmap='YlGnBu')   # full 10x10 grid so all ten models appear on both axes
a.set_xticks(range(len(ORGLAB))); a.set_xticklabels(ORGLAB,rotation=55,ha='right',fontsize=9)
a.set_yticks(range(len(ORGLAB))); a.set_yticklabels(ORGLAB,fontsize=9)
for i in range(len(ORGLAB)):
    for j in range(len(ORGLAB)):
        if not np.isnan(C[i,j]): a.text(j,i,f"{C[i,j]:.2f}",ha='center',va='center',fontsize=7.5,color='white' if C[i,j]>0.30 else 'black')
for s in a.spines.values(): s.set_visible(False)
a.tick_params(length=0)
a.set_title('Error correlation between the ten HELM base models,\naveraged over the eight datasets',fontsize=10.5)
cb=fig.colorbar(im,ax=a,fraction=0.046,pad=0.04); cb.set_label('error correlation (Pearson)',fontsize=9)
panel(a,'a',x=-0.16)
# Panel B: distribution of off-diagonal pairwise error correlations, grouped by dataset.
#          Each group is annotated with the correlation it uses, because the regression and
#          classification tasks measure errors on different scales (see caption).
def offdiag(M): M=np.array(M); return M[~np.eye(M.shape[0],dtype=bool)]
groups=[('CFD\n(residual corr.)',[offdiag(v['resid_corr_matrix']) for v in D['cfd'].values()],SN),
        ('PaperReview\n(residual corr.)',[offdiag(v['resid_corr_matrix']) for v in D['paper_review'].values()],HUM),
        ('HELM\n(error-indicator corr.)',[offdiag(v['err_indicator_corr_matrix']) for v in D['helm'].values()],AVG)]
b=ax[1]
data=[np.concatenate(g[1]) for g in groups]; pos=list(range(1,len(groups)+1))
bp=b.boxplot(data,positions=pos,widths=0.58,showfliers=False,patch_artist=True,medianprops=dict(color='black',lw=1.4))
for patch,g in zip(bp['boxes'],groups): patch.set_facecolor(g[2]); patch.set_alpha(0.55)
for i,g in enumerate(groups):
    means=[np.mean(x) for x in g[1]]
    b.plot(np.full(len(means),i+1)+np.linspace(-0.16,0.16,len(means)),means,'o',color='black',ms=3.5,zorder=5)
b.axhline(0,ls='--',color=REF,lw=1.1); b.text(0.55,0.015,'independent\n(zero correlation)',color=REF,fontsize=8,va='bottom')
b.set_xticks(pos); b.set_xticklabels([g[0] for g in groups],fontsize=8.5)
b.set_ylabel('pairwise error correlation between base models'); b.set_title('Error correlation by dataset',fontsize=10.5)
panel(b,'b',x=-0.14)
plt.tight_layout(); plt.savefig(os.path.join(FIG,'si_helm_corr_heatmap.png'),bbox_inches='tight'); plt.close(); print('Fig5 ok')

# ---------------- Figure S10: redundant clusters degrade averaging but not StackingNet (4 lineages x 2 datasets) ----------------
R=J('expB4_helm_robustness.json')
FAMc={'openai':'#4C72B0','cohere':'#DD8452','ai21':'#55A868','meta':'#C44E52'}
FAMn={'openai':'OpenAI (GPT)','cohere':'Cohere (Command)','ai21':'AI21 (Jurassic)','meta':'Meta (Llama)'}
fig,ax=plt.subplots(1,2,figsize=(12.8,5.4),sharey=True)
for p,(ds,nice) in enumerate([('boolq','BoolQ'),('civil_comments','CivilComments')]):
    a=ax[p]; fams=R[ds]['families']
    for fam,col in FAMc.items():
        if fam not in fams: continue
        d=fams[fam]; gap=[s-v for s,v in zip(d['stackingnet'],d['averaging'])]
        a.plot(d['k'],gap,'o-',color=col,ms=6,lw=1.8,mec='black',mew=0.5,label=FAMn[fam])
    a.axhline(0,ls='--',color=REF,lw=1.2); a.text(0.1,0.12,'averaging = StackingNet',color=REF,fontsize=9,va='bottom')
    a.set_xlabel('redundant same-organization versions added',fontsize=11)
    if p==0: a.set_ylabel('StackingNet $-$ Averaging\n(balanced accuracy, points)',fontsize=11)
    a.set_title(nice,fontsize=12); a.tick_params(labelsize=10); panel(a,'ab'[p])
ax[1].legend(loc='upper left',frameon=False,fontsize=10,title='redundant family',title_fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG,'si_helm_robustness.png'),bbox_inches='tight'); plt.close(); print('S10 ok')
print('done')
