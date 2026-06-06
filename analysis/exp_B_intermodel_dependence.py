"""
Exp B -- Inter-model dependence & ensemble diversity.
=====================================================
Answers Reviewer 1 comment 3: quantify how far Assumption 2 (conditional
independence of base-model predictions given the label) is violated, for each task.

Why error/residual correlation (not raw prediction correlation) is the right probe:
Assumption 2 is about predictions GIVEN the true label, so the operational signature
of a violation is correlated *errors* (residuals for regression, error indicators for
classification). Raw predictions are correlated even for independent-but-accurate models
simply because they track the shared true signal; we report both but read dependence
from the error/residual correlations.

Per task group we compute:
  Regression (CFD, paper-review):
    - residual (signed-error) correlation matrix; mean off-diagonal = "error-dependence index"
    - raw prediction correlation matrix; mean off-diagonal (context)
  Classification (HELM):
    - error-indicator (phi) correlation; mean off-diagonal
    - Kuncheva (2003) pairwise diversity: Q-statistic, correlation rho, disagreement, double-fault
The CFD residual correlations also serve as the DIRECTIONAL-bias-sharing measure that
explains the Exp A 'trustworthy' exception (shared directional bias cannot be cancelled
by aggregation).

Usage:
  python analysis/exp_B_intermodel_dependence.py --smoke
  python analysis/exp_B_intermodel_dependence.py
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

CFD_ATTRS = ['afraid', 'angry', 'attractive', 'babyfaced', 'disgusted', 'feminine',
             'happy', 'masculine', 'sad', 'surprised', 'threatening', 'trustworthy', 'unusual']
CFD_BASE = ['BLIP', 'DeepSeek-VL', 'H2OVL', 'InternVL-2', 'LLaVA',
            'Molmo', 'Paligemma', 'Phi-3.5', 'SAIL-VL', 'SmolVLM']
PR_TASKS = ['ICLR2025', 'ICLR2024', 'NeurIPS2024', 'NeurIPS2023']
PR_LLMS = ['deepseek', 'doubao', 'gemini-2', 'gpt-5', 'gpt4o', 'qwent_turbo']


def mean_offdiag(M):
    """Mean of off-diagonal entries, ignoring NaNs (a constant-output model gives an
    undefined prediction-correlation row; such NaNs are excluded rather than propagated)."""
    n = M.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.nanmean(M[mask]))


def corr_matrix(X):
    """Pearson correlation matrix of rows (X: n_models x n_samples).
    Rows with zero variance (constant output) yield NaN correlations by definition."""
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.corrcoef(X)


def kuncheva_pair(c_i, c_j):
    """Pairwise classifier diversity from oracle correctness vectors (1=correct).
    Returns Q-statistic, correlation rho, disagreement, double-fault."""
    N = len(c_i)
    N11 = float(np.sum((c_i == 1) & (c_j == 1)))
    N00 = float(np.sum((c_i == 0) & (c_j == 0)))
    N10 = float(np.sum((c_i == 1) & (c_j == 0)))
    N01 = float(np.sum((c_i == 0) & (c_j == 1)))
    q_den = N11 * N00 + N01 * N10
    Q = (N11 * N00 - N01 * N10) / q_den if q_den > 0 else 0.0
    r_den = np.sqrt((N11 + N10) * (N01 + N00) * (N11 + N01) * (N10 + N00))
    rho = (N11 * N00 - N01 * N10) / r_den if r_den > 0 else 0.0
    dis = (N01 + N10) / N
    df = N00 / N
    return Q, rho, dis, df


def reg_task(preds, labels):
    """preds: n_models x n_samples; labels: n_samples. Returns dependence summary."""
    resid = preds - labels[None, :]                 # signed errors
    Rr = corr_matrix(resid)
    Rp = corr_matrix(preds)
    return {"resid_corr_mean_offdiag": mean_offdiag(Rr),
            "pred_corr_mean_offdiag": mean_offdiag(Rp),
            "resid_corr_matrix": Rr.tolist(),
            "pred_corr_matrix": Rp.tolist()}


def clf_task(preds, labels):
    """preds: n_models x n_samples (class ids); labels: n_samples."""
    M = preds.shape[0]
    correct = (preds == labels[None, :]).astype(int)
    err_ind = 1 - correct
    Re = corr_matrix(err_ind.astype(float))         # phi-like correlation of error indicators
    Qs, rhos, diss, dfs = [], [], [], []
    for i in range(M):
        for j in range(i + 1, M):
            Q, rho, dis, df = kuncheva_pair(correct[i], correct[j])
            Qs.append(Q); rhos.append(rho); diss.append(dis); dfs.append(df)
    return {"err_indicator_corr_mean_offdiag": mean_offdiag(Re),
            "Q_mean": float(np.mean(Qs)), "rho_mean": float(np.mean(rhos)),
            "disagreement_mean": float(np.mean(diss)), "double_fault_mean": float(np.mean(dfs)),
            "err_indicator_corr_matrix": Re.tolist()}


def run_cfd(log):
    out = {}
    for a in CFD_ATTRS:
        df = pd.read_excel(os.path.join(REPO, 'results', 'CFD', log, f'{a}_predictions.xlsx'))
        y = df['label'].to_numpy(float)
        P = np.vstack([df[m].to_numpy(float) for m in CFD_BASE])
        out[a] = reg_task(P, y)
    return out


def run_pr():
    out = {}
    for t in PR_TASKS:
        df = pd.read_excel(os.path.join(REPO, 'data', 'Research_Review', f'{t}.xlsx'))
        df = df[df['paperID'] != 'MAE']
        y = df['label'].to_numpy(float)
        P = np.vstack([df[m].to_numpy(float) for m in PR_LLMS])
        out[t] = reg_task(P, y)
    return out


def run_helm(datasets=None):
    from helm_io import load_helm, HELM_DATASETS
    out = {}
    for ds in (datasets or HELM_DATASETS):
        preds, labels, K, names = load_helm(ds)
        out[ds] = clf_task(preds, labels)
        out[ds]["n_classes"] = K
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default='fairrun')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=os.path.join(REPO, 'results', 'analysis', 'expB_dependence.json'))
    args = ap.parse_args()

    res = {"cfd": run_cfd(args.log),
           "paper_review": run_pr(),
           "helm": run_helm(['boolq'] if args.smoke else None)}

    # ---- summary printout ----
    print("\n== CFD (regression): residual-error correlation (mean off-diag) ==")
    for a, r in res["cfd"].items():
        print(f"  {a:11s} resid-corr {r['resid_corr_mean_offdiag']:+.3f}   pred-corr {r['pred_corr_mean_offdiag']:+.3f}")
    cfd_mean = np.mean([r['resid_corr_mean_offdiag'] for r in res['cfd'].values()])
    print(f"  CFD mean residual-corr across attributes: {cfd_mean:+.3f}")
    print(f"  -> trustworthy residual-corr = {res['cfd']['trustworthy']['resid_corr_mean_offdiag']:+.3f} "
          f"(directional-bias sharing; Exp A exception)")

    print("\n== Paper-review (regression): residual-error correlation (mean off-diag) ==")
    for t, r in res["paper_review"].items():
        print(f"  {t:11s} resid-corr {r['resid_corr_mean_offdiag']:+.3f}   pred-corr {r['pred_corr_mean_offdiag']:+.3f}")

    print("\n== HELM (classification): dependence + Kuncheva diversity ==")
    for ds, r in res["helm"].items():
        print(f"  {ds:16s} err-corr {r['err_indicator_corr_mean_offdiag']:+.3f}  "
              f"Q {r['Q_mean']:+.3f}  rho {r['rho_mean']:+.3f}  "
              f"disagree {r['disagreement_mean']:.3f}  double-fault {r['double_fault_mean']:.3f}")

    if not args.smoke:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, 'w'), indent=1)
        print(f"\nSaved -> {args.out}")


if __name__ == '__main__':
    main()
