"""
Consolidate Exp A-D JSON outputs into figures for the StackingNet revision.
Saves PNGs to results/analysis/figures/.
  fig_A_fairness.png   -- worst-group MAE + bias-amplification per CFD attribute (R1.1/1.2)
  fig_B_dependence.png -- error-correlation heatmap + per-task dependence (R1.3)
  fig_B2_degradation.png -- synthetic theory curve + real injection (R1.3b)
  fig_C_dispersion.png -- dispersion vs gain scatter + strong-subset (R1.5)
  fig_D_human.png      -- leave-one-out human vs AI (R1.4/R2.1)
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
AN = os.path.join(REPO, 'results', 'analysis')
FIG = os.path.join(AN, 'figures')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'figure.dpi': 130, 'savefig.bbox': 'tight'})
C_SN, C_AVG, C_BASE = '#1f77b4', '#ff7f0e', '#7f7f7f'


def load(name):
    return json.load(open(os.path.join(AN, name)))


def fig_A():
    d = load('expA_fairness_battery.json')
    attrs = list(d.keys())
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    x = np.arange(len(attrs)); w = 0.27
    for k, (m, col, lab) in enumerate([('StackingNet', C_SN, 'StackingNet'),
                                       ('Averaging', C_AVG, 'Averaging'), ('best', C_BASE, 'best base')]):
        wg, lo, hi = [], [], []
        for a in attrs:
            e = d[a]['axes']['ethnicity']; bb = d[a]['best_base']
            pm = e['per_method'][bb if m == 'best' else m]
            wg.append(pm['worst_group_mae'])
            ci = e['ci'][bb if m == 'best' else m]['worst_group_mae']
            lo.append(pm['worst_group_mae'] - ci[0]); hi.append(ci[1] - pm['worst_group_mae'])
        ax[0].bar(x + (k - 1) * w, wg, w, yerr=[lo, hi], capsize=2, color=col, label=lab)
    ax[0].set_ylabel('worst-group MAE\n(ethnicity, [0,1])'); ax[0].legend(ncol=3, fontsize=8)
    ax[0].set_title('Exp A: StackingNet reduces worst-group error on 10/13 attributes '
                    '(exception: trustworthy)')
    ba = [d[a]['axes']['ethnicity']['bias_amplification']['StackingNet']['gap'] for a in attrs]
    cols = ['#d62728' if v > 0 else C_SN for v in ba]
    ax[1].bar(x, ba, color=cols)
    ax[1].axhline(0, color='k', lw=0.8)
    ax[1].set_ylabel('bias amplification\n(gap vs base mean)')
    ax[1].set_xticks(x); ax[1].set_xticklabels(attrs, rotation=45, ha='right')
    ax[1].set_title('Negative = StackingNet reduces between-group disparity (12/13; only trustworthy amplifies)')
    plt.savefig(os.path.join(FIG, 'fig_A_fairness.png')); plt.close()
    print('fig_A_fairness.png')


def fig_B():
    d = load('expB_dependence.json')
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    M = np.array(d['cfd']['trustworthy']['resid_corr_matrix'])
    base = ['BLIP', 'DeepSeek-VL', 'H2OVL', 'InternVL-2', 'LLaVA', 'Molmo',
            'Paligemma', 'Phi-3.5', 'SAIL-VL', 'SmolVLM']
    im = ax[0].imshow(M, vmin=-1, vmax=1, cmap='RdBu_r')
    ax[0].set_xticks(range(len(base))); ax[0].set_xticklabels(base, rotation=90, fontsize=7)
    ax[0].set_yticks(range(len(base))); ax[0].set_yticklabels(base, fontsize=7)
    ax[0].set_title("CFD 'trustworthy' residual-error correlation\n(positive => shared directional bias)")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    # per-task dependence summary
    cfd = np.mean([v['resid_corr_mean_offdiag'] for v in d['cfd'].values()])
    pr = np.mean([v['resid_corr_mean_offdiag'] for v in d['paper_review'].values()])
    helm_q = np.mean([v['Q_mean'] for v in d['helm'].values()])
    helm_e = np.mean([v['err_indicator_corr_mean_offdiag'] for v in d['helm'].values()])
    labels = ['CFD\nresid-corr', 'paper-rev\nresid-corr', 'HELM\nerr-corr', 'HELM\nQ-stat']
    vals = [cfd, pr, helm_e, helm_q]
    ax[1].bar(labels, vals, color=['#2ca02c', '#2ca02c', '#9467bd', '#9467bd'])
    ax[1].axhline(0, color='k', lw=0.8)
    ax[1].set_ylabel('mean inter-model dependence')
    ax[1].set_title('Errors are correlated across tasks\n(Assumption 2 is an idealization)')
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.01, f'{v:+.2f}', ha='center', fontsize=8)
    plt.savefig(os.path.join(FIG, 'fig_B_dependence.png')); plt.close()
    print('fig_B_dependence.png')


def fig_B2():
    d = load('expB2_degradation.json')
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    syn = d['synthetic']
    rho = [r['rho'] for r in syn]
    ax[0].plot(rho, [r['empirical_reduction'] for r in syn], 'o', color=C_SN, label='empirical')
    ax[0].plot(rho, [r['theory_reduction'] for r in syn], '-', color='k', label='theory (1+(M-1)ρ)/M')
    ax[0].set_xlabel('error correlation ρ'); ax[0].set_ylabel('ensemble MSE / single MSE')
    ax[0].set_title('Synthetic: averaging benefit\ndegrades as independence is violated')
    ax[0].legend()
    inj = d['injection']['rows']
    k = [r['k_duplicates'] for r in inj]
    ax[1].plot(k, [r['avg_mae'] for r in inj], 's-', color=C_AVG, label='Averaging')
    ax[1].plot(k, [r['stackingnet_mae'] for r in inj], 'o-', color=C_SN, label='StackingNet')
    ax[1].axhline(d['injection']['best_base_test_mae'], color=C_BASE, ls='--', label='best base')
    ax[1].set_xlabel('# injected duplicate (correlated) models')
    ax[1].set_ylabel('test MAE (ICLR2025, [1,10])')
    ax[1].set_title('Real injection: StackingNet down-weights\nredundancy; averaging collapses')
    ax[1].legend()
    plt.savefig(os.path.join(FIG, 'fig_B2_degradation.png')); plt.close()
    print('fig_B2_degradation.png')


def fig_C():
    d = load('expC_dispersion_gain.json')
    reg = [p for p in d['points'] if p['type'] == 'regression']
    clf = [p for p in d['points'] if p['type'] == 'classification']
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    # (1) regression dispersion-gain (own MAE units)
    ax[0].scatter([p['dispersion'] for p in reg], [p['gain'] for p in reg], color=C_SN)
    ax[0].axhline(0, color='k', lw=0.6)
    ax[0].set_xlabel('base dispersion (worst-best MAE)')
    ax[0].set_ylabel('StackingNet gain (MAE, +=better)')
    ax[0].set_title('Regression (17 tasks)\nρ(dispersion,gain)=+0.78')
    # (2) classification dispersion-gain (own BCA units)
    ax[1].scatter([p['dispersion'] for p in clf], [p['gain'] for p in clf],
                  color='#9467bd', marker='^')
    for p in clf:
        ax[1].annotate(p['task'][:5], (p['dispersion'], p['gain']), fontsize=6,
                       xytext=(2, 2), textcoords='offset points')
    ax[1].axhline(0, color='k', lw=0.6)
    ax[1].set_xlabel('base dispersion (best-worst BCA pts)')
    ax[1].set_ylabel('StackingNet gain (BCA pts)')
    ax[1].set_title('Classification (8 tasks)\nno clean trend (gain != pruning)')
    # (3) strong subset
    ss = d['strong_subset']
    tasks = sorted(set(r['task'] for r in ss))
    full = [next(r['gain_over_best'] for r in ss if r['task'] == t and r['pool'] == 'full') for t in tasks]
    top = [next(r['gain_over_best'] for r in ss if r['task'] == t and r['pool'] != 'full') for t in tasks]
    x = np.arange(len(tasks)); w = 0.38
    ax[2].bar(x - w / 2, full, w, color=C_BASE, label='full pool (6)')
    ax[2].bar(x + w / 2, top, w, color=C_SN, label='top-3 strong')
    ax[2].set_xticks(x); ax[2].set_xticklabels(tasks, rotation=30, ha='right')
    ax[2].set_ylabel('gain over best base (MAE)')
    ax[2].set_title('Strong-subset: gain persists\nwhen pool uniformly strong')
    ax[2].legend(fontsize=8)
    plt.savefig(os.path.join(FIG, 'fig_C_dispersion.png')); plt.close()
    print('fig_C_dispersion.png')


def fig_D():
    d = load('expD_human_loo.json')
    tasks = list(d.keys())
    x = np.arange(len(tasks)); w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.2))
    series = [('human_full_mae', '#cccccc', 'Human vs full consensus (self-incl.)'),
              ('human_loo_mae', '#000000', 'Human (fair, LOO)'),
              ('SN_loo', C_SN, 'StackingNet (LOO)'),
              ('bestLLM_loo', '#9467bd', 'best single LLM (LOO)')]
    for k, (key, col, lab) in enumerate(series):
        if key == 'SN_loo':
            vals = [d[t]['methods']['StackingNet']['loo_mae'] for t in tasks]
        elif key == 'bestLLM_loo':
            vals = [min(d[t]['methods'][m]['loo_mae']
                        for m in d[t]['methods'] if m not in ('Averaging', 'StackingNet')) for t in tasks]
        else:
            vals = [d[t][key] for t in tasks]
        ax.bar(x + (k - 1.5) * w, vals, w, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylabel('MAE vs consensus ([1,10])')
    ax.set_title('Exp D: fair leave-one-out comparison. Single LLM ≈ single human;\n'
                 "ensemble edge = variance reduction, not superior judgment")
    ax.legend(fontsize=7)
    plt.savefig(os.path.join(FIG, 'fig_D_human.png')); plt.close()
    print('fig_D_human.png')


if __name__ == '__main__':
    fig_A(); fig_B(); fig_B2(); fig_C(); fig_D()
    print('All figures ->', FIG)
