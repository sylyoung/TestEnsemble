"""
Exp B2 -- Controlled degradation under violated independence.
============================================================
Answers Reviewer 1 comment 3(b): "discuss how StackingNet degrades when independence
is violated." Both parts are fully controlled synthetic experiments, so the only thing
that varies is the dependence between base models; everything else (number of models,
their individual accuracy) is held fixed. The real-model counterpart, in which a
correlated same-organisation cluster is grown inside a real HELM pool, is Exp B4
(Supporting Information Figure S10).

Part 1 (the averaging mechanism):
  M base regressors of EQUAL error variance sigma^2 and a common pairwise error
  correlation rho. The averaged-error variance has the closed form
  sigma^2 * (1 + (M-1)*rho) / M, so averaging's MSE relative to a single model rises
  from 1/M at rho=0 (independent: full benefit) to 1 at rho=1 (perfectly correlated:
  no benefit). We confirm the Monte-Carlo curve matches this closed form. Because the
  models are equally good, the best single model equals the average single model, so
  the y-axis is exactly "averaging MSE / best-single-model MSE".

Part 2 (StackingNet vs averaging under a growing redundant cluster):
  We hold M=8 base regressors of EQUAL error variance, so the best single model is the
  same throughout and no model is "better" or "worse". We then make a cluster of c of
  them mutually redundant: the c cluster members share one common error term (perfectly
  correlated), while the remaining M-c members have independent errors. As c grows from
  1 (all independent) to M (all redundant), the number of effectively independent votes
  falls. Unweighted averaging is increasingly dominated by the cluster's shared error and
  its MSE rises toward the single-model level; the SUPERVISED StackingNet, fit by ERM on
  a small labelled subset, learns to treat the redundant cluster as roughly one vote and
  so degrades far more gracefully. We report test MSE vs c for best-single, averaging,
  and StackingNet, all relative to the single-model MSE.

Usage:
  python analysis/exp_B2_controlled_degradation.py
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from regression_stackingnet_cfd import StackingRegression  # reuse the real model class

PR_LLMS = ['deepseek', 'doubao', 'gemini-2', 'gpt-5', 'gpt4o', 'qwent_turbo']


# ----------------------- Part 1: synthetic -----------------------
def synthetic_curve(M=10, n=20000, sigma=1.0, rhos=None, seed=0):
    rng = np.random.default_rng(seed)
    rhos = rhos if rhos is not None else np.linspace(0, 1, 11)
    rows = []
    for rho in rhos:
        # equicorrelated standard normals: z_j = sqrt(rho)*c + sqrt(1-rho)*eps_j
        c = rng.standard_normal((n, 1))
        eps = rng.standard_normal((n, M))
        z = np.sqrt(rho) * c + np.sqrt(max(1 - rho, 0.0)) * eps
        err = sigma * z                                   # (n, M) base errors
        single_mse = float(np.mean(err ** 2))             # avg single-model MSE
        ens_err = err.mean(axis=1)                        # averaged error
        ens_mse = float(np.mean(ens_err ** 2))
        rows.append({"rho": float(rho),
                     "empirical_reduction": ens_mse / single_mse,
                     "theory_reduction": (1 + (M - 1) * rho) / M})
    return rows


# ----------------------- Part 2: real-data injection -----------------------
def fit_stackingnet(Xtr, ytr, Xte, lr=0.01, epochs=300, seed=0):
    """Train the real StackingRegression (w>=0, b>=0) and return (test_preds, weights, bias)."""
    torch.manual_seed(seed)
    m = StackingRegression(Xtr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    crit = torch.nn.MSELoss()
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    Xe = torch.tensor(Xte, dtype=torch.float32)
    m.train()
    for _ in range(epochs):
        loss = crit(m(Xt), yt)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for p in m.parameters():
                p.data.clamp_(min=0)
    m.eval()
    with torch.no_grad():
        pred = m(Xe).numpy()
    return pred, m.scales.detach().numpy().copy(), float(m.biases.detach())


def redundancy_experiment(M=8, n=6000, sigma=1.0, train_frac=0.3, seeds=range(10)):
    """Equal-quality base regressors; grow a redundant (perfectly correlated) cluster of
    size c. Report test MSE (relative to the single-model MSE) for best-single, averaging,
    and the supervised StackingNet, averaged over seeds with std."""
    rows = []
    for c in range(1, M + 1):
        bs, av, sn = [], [], []
        for seed in seeds:
            rng = np.random.default_rng(1000 + seed)
            y = rng.standard_normal(n)
            shared = rng.standard_normal(n)                      # common error of the cluster
            E = np.empty((n, M))
            E[:, :c] = shared[:, None]                           # c perfectly redundant members
            E[:, c:] = rng.standard_normal((n, M - c))           # M-c independent members
            X = y[:, None] + sigma * E                           # each predictor = y + error
            single_mse = sigma ** 2                              # equal for every model (variance of error)
            idx = rng.permutation(n); ntr = int(train_frac * n)
            tr, te = idx[:ntr], idx[ntr:]
            avg_pred = X[te].mean(axis=1)
            av.append(np.mean((avg_pred - y[te]) ** 2) / single_mse)
            sn_pred, _, _ = fit_stackingnet(X[tr], y[tr], X[te], seed=seed)
            sn.append(np.mean((sn_pred - y[te]) ** 2) / single_mse)
            bs.append(1.0)                                       # best single = single (all equal)
        rows.append({"cluster_size": c, "n_independent_votes": M - c + 1,
                     "best_single_rel": 1.0,
                     "averaging_rel_mean": float(np.mean(av)), "averaging_rel_std": float(np.std(av)),
                     "stackingnet_rel_mean": float(np.mean(sn)), "stackingnet_rel_std": float(np.std(sn))})
    return {"M": M, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(REPO, 'results', 'analysis', 'expB2_degradation.json'))
    args = ap.parse_args()

    syn = synthetic_curve()
    print("== Part 1: synthetic equicorrelated ensemble (averaging) ==")
    print("  rho   empirical/theory MSE-reduction factor (1/M at rho=0 -> 1 at rho=1)")
    for r in syn:
        print(f"  {r['rho']:.1f}   emp {r['empirical_reduction']:.3f}  theory {r['theory_reduction']:.3f}")

    red = redundancy_experiment()
    print(f"\n== Part 2: grow a redundant cluster among M={red['M']} equal-quality regressors ==")
    print("  c  indep_votes  best_single  averaging(MSE/single)  StackingNet(MSE/single)")
    for r in red['rows']:
        print(f"  {r['cluster_size']:2d}      {r['n_independent_votes']:2d}        {r['best_single_rel']:.3f}     "
              f"{r['averaging_rel_mean']:.3f} +/- {r['averaging_rel_std']:.3f}     "
              f"{r['stackingnet_rel_mean']:.3f} +/- {r['stackingnet_rel_std']:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"synthetic": syn, "redundancy": red}, open(args.out, 'w'), indent=1)
    print(f"\nSaved -> {args.out}")


if __name__ == '__main__':
    main()
