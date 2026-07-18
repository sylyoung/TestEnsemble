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
    "figure.dpi": 600, "savefig.dpi": 600,
})
SN="#C51B7D"; AVG="#5E6A71"; HUM="#E69F00"; REF="#9AA0A6"; BB="#4C9F70"; LAB="#7B8794"
HERE=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.dirname(HERE)
ANA=os.path.join(REPO,'results','analysis'); FIG=os.path.join(ANA,'figures')
def J(n): return json.load(open(os.path.join(ANA,n)))
def panel(ax,letter,x=-0.13,y=1.03): ax.text(x,y,letter.upper(),transform=ax.transAxes,fontsize=12,fontweight="bold",va="bottom")

# ---------------- Figure S5 (si_degradation): redundancy degrades averaging but not StackingNet ----------------
# Two pool sizes (M = 10 and M = 100) to show the effect does not hinge on a particular M.
d=J('expB2_degradation.json'); RBM=d['redundancy_by_M']; Ms=[10,100]
fig,ax=plt.subplots(1,2,figsize=(11.0,4.4))
for p,M in enumerate(Ms):
    a=ax[p]; rr=RBM[str(M)]['rows']; c=np.array([r['cluster_size'] for r in rr])
    av=np.array([r['averaging_rel_mean'] for r in rr]); ave=np.array([r['averaging_rel_std'] for r in rr])
    sn=np.array([r['stackingnet_rel_mean'] for r in rr]); sne=np.array([r['stackingnet_rel_std'] for r in rr])
    # break-even reference: a single base model has relative MSE 1, so averaging climbing
    # toward this dashed line means it has lost all benefit of combining
    a.axhline(1.0,ls='--',color=REF,lw=1.1)
    a.text(c[-1],1.06,'single-model error',color=REF,fontsize=8.5,ha='right',va='bottom')
    # shaded +/- std bands plus markers and error bars so the run-to-run spread stays visible.
    # a log y-axis magnifies the low region where StackingNet sits so its standard-deviation
    # band is legible while averaging still reads from its near-zero values up to the single-model line.
    a.fill_between(c, av-ave, av+ave, color=AVG, alpha=0.30, lw=0)
    a.fill_between(c, sn-sne, sn+sne, color=SN, alpha=0.30, lw=0)
    a.errorbar(c,av,yerr=ave,fmt='s-',color=AVG,mec='black',mew=0.3,ms=3,lw=1.3,capsize=3.5,elinewidth=1.3,label='Averaging')
    a.errorbar(c,sn,yerr=sne,fmt='o-',color=SN,mec='black',mew=0.3,ms=3,lw=1.3,capsize=3.5,elinewidth=1.3,label='StackingNet')
    a.set_xlabel(f"size of redundant cluster (of M = {M} equally accurate models)")
    a.set_yscale('log'); a.set_ylim(0.008,1.5); panel(a,'ab'[p])
    a.legend(loc='upper left',frameon=False)
    if p==0: a.set_ylabel('test MSE / single-model MSE')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'si_degradation.png'),bbox_inches='tight'); plt.close(); print('S5 ok')

# ---------------- Figure S7: where does the gain come from? dispersion vs gain ----------------
from scipy.stats import spearmanr
Cd=J('expC_dispersion_gain.json')
reg=[p for p in Cd['points'] if p['type']=='regression']
cfd=[p for p in reg if p['task'].startswith('CFD')]            # A: Chicago Face Database only
clf=[p for p in Cd['points'] if p['type']=='classification']
fig,ax=plt.subplots(1,2,figsize=(10.6,4.6))
# A: CFD regression dispersion vs gain, with trend
a=ax[0]; xd=np.array([p['dispersion'] for p in cfd]); yg=np.array([p['gain_over_avg'] for p in cfd])
rho_r,_=spearmanr(xd,yg)
a.scatter(xd,yg,color=SN,s=36,edgecolor='black',linewidth=0.4,zorder=3)
xs=np.linspace(xd.min(),xd.max(),50); b1,b0=np.polyfit(xd,yg,1)
a.plot(xs,b1*xs+b0,color=SN,lw=1.4,alpha=0.7)
a.margins(y=0.18)
a.axhline(0,color='0.35',lw=1.0,ls='--',zorder=1)
a.set_xlabel('base-model dispersion\n(worst $-$ best MAE)'); a.set_ylabel('StackingNet gain over averaging (MAE)')
a.set_title(f'Chicago Face Database, 13 attributes\nrank correlation $r = {rho_r:+.2f}$',fontsize=10.5); panel(a,'a')
# B: classification dispersion vs gain, every dataset labeled by name
a=ax[1]; xc=np.array([p['dispersion'] for p in clf]); yc=np.array([p['gain_over_avg'] for p in clf]); tc=[p['task'] for p in clf]
rho_c,_=spearmanr(xc,yc)
NICE={'boolq':'BoolQ','entity_matching':'EntityMatching','imdb':'IMDB','legal_support':'LegalSupport',
      'lsat_qa':'LSAT','mmlu':'MMLU','raft':'RAFT','civil_comments':'CivilComments'}
OFF={'boolq':(0,12),'mmlu':(0,-13),'entity_matching':(-22,4),'imdb':(13,4),
     'legal_support':(-30,7),'lsat_qa':(15,5),'raft':(8,-13),'civil_comments':(16,8)}
xsc=np.linspace(xc.min(),xc.max(),50); c1,c0=np.polyfit(xc,yc,1)
a.plot(xsc,c1*xsc+c0,color=AVG,lw=1.4,alpha=0.7,zorder=2)
a.scatter(xc,yc,color=AVG,marker='^',s=44,edgecolor='black',linewidth=0.4,zorder=3)
for i,t in enumerate(tc):
    dx,dy=OFF.get(t,(0,12))
    a.annotate(NICE[t],(xc[i],yc[i]),xytext=(dx,dy),textcoords='offset points',
               fontsize=7.5,color='0.25',ha='center',va='bottom' if dy>0 else 'top',
               arrowprops=dict(arrowstyle='-',color='0.6',lw=0.5))
a.margins(y=0.20)
a.axhline(0,color='0.35',lw=1.0,ls='--',zorder=1)
a.set_xlabel('base-model dispersion\n(best $-$ worst balanced accuracy, pts)'); a.set_ylabel('StackingNet gain over averaging (BCA pts)')
a.set_title(f'HELM, 8 datasets\nrank correlation $r = {rho_c:+.2f}$',fontsize=10.5); panel(a,'b')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'si_dispersiongain.png'),bbox_inches='tight'); plt.close(); print('S7 ok')

# ---------------- Figure S8: demographic parity (between-group mean-prediction gap) ----------------
# The gender panel uses a BROKEN y-axis because the two gender-defining attributes (masculine,
# feminine) are large by construction while the rest are small, so a single axis would flatten the
# small bars near zero. The break sits in the empty 0.06-0.33 band that no bar occupies, and each
# tall bar is drawn solid on BOTH sub-axes (full to the top of the lower one, full to the bottom of
# the upper one) so it reads as one continuous bar interrupted only by the thin axis-break marks.
from matplotlib.gridspec import GridSpec
LABc="#BCC3CB"; BBc="#2C7FB8"; SNc="#C51B7D"; w=0.27
A=J('expA3_fairness_views.json'); ATTR=list(A.keys()); x=np.arange(len(ATTR))
def triplet(axis):
    lab=[A[at]['axes'][axis]['dp_gap_label'] for at in ATTR]
    bb=[A[at]['axes'][axis]['methods'][A[at]['best_base']]['dp_gap'] for at in ATTR]
    sn=[A[at]['axes'][axis]['methods']['StackingNet']['dp_gap'] for at in ATTR]
    return lab,bb,sn
def draw(a,lab,bb,sn):
    a.bar(x-w,lab,w,color=LABc,label='Human labels (reference)',edgecolor='black',linewidth=0.4)
    a.bar(x,bb,w,color=BBc,label='Best base model',edgecolor='black',linewidth=0.4)
    a.bar(x+w,sn,w,color=SNc,label='StackingNet',edgecolor='black',linewidth=0.4)
fig=plt.figure(figsize=(11.5,4.6))
gs=GridSpec(2,2,height_ratios=[1.0,2.4],hspace=0.06,figure=fig)
# --- panel a: race/ethnicity, single axis spanning both rows ---
aR=fig.add_subplot(gs[:,0]); lab,bb,sn=triplet('race'); draw(aR,lab,bb,sn)
aR.set_xticks(x); aR.set_xticklabels(ATTR,rotation=55,ha='right')
aR.set_ylabel('between-group difference in mean prediction'); aR.set_title('Race/ethnicity (6 groups)')
aR.margins(y=0.16); panel(aR,'a')
# --- panel b: gender, broken y-axis; tall bars span both sub-axes so they read as one solid bar ---
lab,bb,sn=triplet('gender'); allv=lab+bb+sn
tall=[v for v in allv if v>=0.2]; small=[v for v in allv if v<0.2]
aGt=fig.add_subplot(gs[0,1]); aGb=fig.add_subplot(gs[1,1])
draw(aGt,lab,bb,sn); draw(aGb,lab,bb,sn)        # same full bars on both: solid through the break
aGt.set_ylim(min(tall)*0.94, max(tall)*1.05)    # upper window: only the tall cluster
aGb.set_ylim(0, max(small)*1.30)                # lower window: baseline through the small cluster
aGt.spines['bottom'].set_visible(False); aGb.spines['top'].set_visible(False)
aGt.tick_params(bottom=False, labelbottom=False)
aGb.set_xticks(x); aGb.set_xticklabels(ATTR,rotation=55,ha='right')
# diagonal break marks straddling the shared boundary
dk=dict(marker=[(-1,-0.6),(1,0.6)],markersize=7,linestyle='none',color='0.4',mec='0.4',mew=1,clip_on=False)
aGt.plot([0],[0],transform=aGt.transAxes,**dk); aGb.plot([0],[1],transform=aGb.transAxes,**dk)  # left (y-axis) break mark only; no spurious mark on the open right side
aGt.set_title('Gender (2 groups)'); panel(aGt,'b',y=1.08)
aGb.set_ylabel('between-group difference in mean prediction'); aGb.yaxis.set_label_coords(-0.13,0.72)
h,l=aR.get_legend_handles_labels()
fig.legend(h,l,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(0.5,1.03))
plt.tight_layout(rect=[0,0,1,0.95])
# A matplotlib broken axis leaves a white band between the two sub-axes that cuts through the tall
# bars. Fill that band with each tall bar's colour so the bar reads as one solid block from baseline
# to its true value; the break stays indicated only by the diagonal marks on the y-axis.
import matplotlib.patches as mpatches, matplotlib.lines as mlines
fig.canvas.draw()
ybot=aGb.get_position().y1; ytop=aGt.get_position().y0; inv=fig.transFigure.inverted(); brk=aGb.get_ylim()[1]
for off,color,vals in [(-w,LABc,lab),(0,BBc,bb),(w,SNc,sn)]:
    for xc,v in zip(x,vals):
        if v>brk:
            xl=inv.transform(aGb.transData.transform((xc+off-w/2,0)))[0]
            xr=inv.transform(aGb.transData.transform((xc+off+w/2,0)))[0]
            fig.add_artist(mpatches.Rectangle((xl,ybot),xr-xl,ytop-ybot,facecolor=color,edgecolor='none',transform=fig.transFigure,zorder=3))
            fig.add_artist(mlines.Line2D([xl,xl],[ybot,ytop],color='black',lw=0.4,transform=fig.transFigure,zorder=3.1))
            fig.add_artist(mlines.Line2D([xr,xr],[ybot,ytop],color='black',lw=0.4,transform=fig.transFigure,zorder=3.1))
plt.savefig(os.path.join(FIG,'si_fairness_dp.png'),bbox_inches='tight'); plt.close(); print('S8 ok')

# ---------------- Figure 5 (helm-corr): error correlations violate conditional independence ----------------
# Panel A: lower-triangular heatmap of the mean error-indicator correlation between the ten
#          HELM base models, averaged over ALL EIGHT HELM classification datasets. Both axes
#          carry the same model ordering; only the lower triangle is shown (the matrix is
#          symmetric and the diagonal is the trivial self-correlation).
ORGLAB=['Meta','OpenAI','AI21','Writer','Falcon','Cohere','Microsoft','MosaicML','Anthropic','Together']  # helm_io.MODEL_NAMES order
D=J('expB_dependence.json')
helm_mats=np.stack([np.array(v['err_indicator_corr_matrix']) for v in D['helm'].values()])
C=np.nanmean(helm_mats,axis=0)
C[np.triu(np.ones_like(C,dtype=bool))]=np.nan          # strictly lower triangle (i>j); diagonal omitted
# Trim the structurally empty Meta row (top, no j<0) and Together column (right, no i>9)
# so every axis tick sits over a filled cell instead of dangling above blank space.
Ctri=C[1:,:-1]; YL=ORGLAB[1:]; XL=ORGLAB[:-1]
fig,ax=plt.subplots(1,2,figsize=(12.8,5.8),gridspec_kw={'width_ratios':[1.0,1.0]})
a=ax[0]; im=a.imshow(Ctri,vmin=0,vmax=0.45,cmap='YlGnBu')
a.set_xticks(range(len(XL))); a.set_xticklabels(XL,rotation=55,ha='right',fontsize=9)
a.set_yticks(range(len(YL))); a.set_yticklabels(YL,fontsize=9)
for i in range(Ctri.shape[0]):
    for j in range(Ctri.shape[1]):
        if not np.isnan(Ctri[i,j]): a.text(j,i,f"{Ctri[i,j]:.2f}",ha='center',va='center',fontsize=7.5,color='white' if Ctri[i,j]>0.30 else 'black')
for s in a.spines.values(): s.set_visible(False)
a.tick_params(length=0)
# in-figure title removed; the caption already describes what the panel shows
cb=fig.colorbar(im,ax=a,fraction=0.046,pad=0.04); cb.set_label('error correlation (Pearson)',fontsize=9)
panel(a,'a',x=-0.16)
# Panel B: distribution of off-diagonal pairwise error correlations, grouped by dataset.
#          Each group is annotated with the correlation it uses, because the regression and
#          classification tasks measure errors on different scales (see caption).
def offdiag(M): M=np.array(M); return M[~np.eye(M.shape[0],dtype=bool)]            # both triangles (box stats)
def lowertri(M): M=np.array(M); return M[np.tril(np.ones_like(M,dtype=bool),k=-1)]  # unique pairs (point cloud)
groups=[('PaperReview\n(residual corr.)',[v['resid_corr_matrix'] for v in D['paper_review'].values()],HUM),
        ('CFD\n(residual corr.)',[v['resid_corr_matrix'] for v in D['cfd'].values()],SN),
        ('HELM\n(error-indicator corr.)',[v['err_indicator_corr_matrix'] for v in D['helm'].values()],AVG)]
b=ax[1]
data=[np.concatenate([offdiag(m) for m in g[1]]) for g in groups]; pos=list(range(1,len(groups)+1))
bp=b.boxplot(data,positions=pos,widths=0.58,showfliers=False,patch_artist=True,medianprops=dict(color='black',lw=1.4),zorder=4)
for patch,g in zip(bp['boxes'],groups): patch.set_facecolor(g[2]); patch.set_alpha(0.45)
# overlay every individual model-pair correlation (unique pairs) as a faint jittered cloud
jit=np.random.default_rng(0)
for i,g in enumerate(groups):
    pairs=np.concatenate([lowertri(m) for m in g[1]])
    # crisp dark dots with a thin white halo so individual pairs stay legible where they overlap
    b.scatter(np.full(len(pairs),i+1)+jit.uniform(-0.28,0.28,len(pairs)),pairs,
              s=9,facecolor='#1a1a1a',edgecolor='white',linewidths=0.3,alpha=0.6,zorder=3)
b.set_xticks(pos); b.set_xticklabels([g[0] for g in groups],fontsize=8.5)
b.set_ylabel('pairwise error correlation between base models')  # in-figure title removed; the caption describes the panel
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
    a.set_axisbelow(True); a.grid(axis='y',ls=':',lw=0.5,color='0.85')
    a.axhline(0,ls='--',color=REF,lw=1.2)
    a.set_xlabel('redundant same-organization variants added',fontsize=11)
    if p==0: a.set_ylabel('StackingNet $-$ Averaging\n(balanced accuracy, points)',fontsize=11)
    a.set_title(nice,fontsize=12); a.tick_params(labelsize=10); panel(a,'ab'[p])
    a.legend(loc='upper left',frameon=False,fontsize=9.5,title='Organization (model family)',title_fontsize=9.5)
plt.tight_layout(); plt.savefig(os.path.join(FIG,'si_helm_robustness.png'),bbox_inches='tight'); plt.close(); print('S10 ok')
print('done')
