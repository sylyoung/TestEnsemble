"""
Exp A -- Fairness metric battery for the CFD attribute-rating task.
==================================================================
Answers Reviewer 1 comments 1 & 2 (Advanced Science revision of StackingNet).

For every method (each of the 10 base VLMs, simple Averaging, and StackingNet) and
every CFD attribute, we report a battery of *regression* group-fairness metrics,
disaggregated by gender (2 groups) and race/ethnicity (6 groups), with nonparametric
bootstrap 95% CIs on the disparity statistics. All quantities are in the normalised
[0,1] label space (predictions and labels were min-max scaled in the main pipeline),
so MAE values are comparable across attributes.

Metric battery (literature-grounded):
  (a) worst-group MAE  ............ minimax / Rawlsian        [Martinez 2020; Diana 2021]
  (b) max-min group-MAE gap ....... reviewer's required stat
  (c) bounded group loss (BGL) .... max group loss vs a bound [Agarwal 2019]  (== (a))
  (d) group-MAE coeff. of variation  dispersion robust to one group [Castelnovo 2022]
  (e) mean-prediction gap ......... statistical-parity proxy  [Agarwal 2019]  (context only:
                                    true ratings legitimately differ by group, so this is
                                    reported for transparency, not as a target)
  (f) Wasserstein-1 disparity ..... max pairwise W1 between group SIGNED-error dists [Chzhen 2020]
  (g) per-group mean signed error . directional over/under-rating + its max-min gap
  (h) bias amplification .......... gap(method) - mean gap(base models)  [Zhao 2017; Wang&Russakovsky 2021]
  (i) bootstrap 95% CIs ........... on (a),(b) and on method-vs-baseline differences

Design notes / integrity:
  * Methods compared: base models, "Averaging" (unweighted mean of the 10 bases), StackingNet.
  * "Best base" is the base model with lowest OVERALL test MAE (reported alongside, post-hoc).
  * StackingNet has ONE global weight vector + scalar bias and NO group-conditional
    parameters, so any gap reduction is incidental to overall error reduction. We report
    the battery transparently rather than claiming a fairness mechanism.
  * Bootstrap resamples test rows with replacement (paired across methods within a rep).
Usage:
  python analysis/exp_A_fairness_battery.py --smoke      # one attribute, quick check
  python analysis/exp_A_fairness_battery.py              # all 13 attributes
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# 13 attributes reported in the paper (age is collected but NOT reported -> excluded)
ATTRS = ['afraid', 'angry', 'attractive', 'babyfaced', 'disgusted', 'feminine',
         'happy', 'masculine', 'sad', 'surprised', 'threatening', 'trustworthy', 'unusual']
BASE_MODELS = ['BLIP', 'DeepSeek-VL', 'H2OVL', 'InternVL-2', 'LLaVA',
               'Molmo', 'Paligemma', 'Phi-3.5', 'SAIL-VL', 'SmolVLM']


# --------------------------- metric helpers ---------------------------
def group_maes(err_abs, group_ids, groups):
    """Mean absolute error within each group. Returns dict group->MAE (NaN if empty)."""
    out = {}
    for g in groups:
        m = group_ids == g
        out[g] = float(np.mean(err_abs[m])) if m.sum() > 0 else float('nan')
    return out


def disparity_stats(err_signed, group_ids, groups):
    """Compute the full battery from signed errors for one method on one group axis."""
    err_abs = np.abs(err_signed)
    gm = group_maes(err_abs, group_ids, groups)
    vals = np.array([gm[g] for g in groups if not np.isnan(gm[g])])
    worst = float(np.max(vals))                       # (a)/(c)
    gap = float(np.max(vals) - np.min(vals))          # (b)
    cov = float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else 0.0  # (d)
    # (g) per-group mean signed error + its spread
    gs = {g: (float(np.mean(err_signed[group_ids == g])) if (group_ids == g).sum() else float('nan'))
          for g in groups}
    gs_vals = np.array([gs[g] for g in groups if not np.isnan(gs[g])])
    signed_gap = float(np.max(gs_vals) - np.min(gs_vals))
    # (f) max pairwise Wasserstein-1 between group signed-error distributions
    present = [g for g in groups if (group_ids == g).sum() > 0]
    w1 = 0.0
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a = err_signed[group_ids == present[i]]
            b = err_signed[group_ids == present[j]]
            w1 = max(w1, float(wasserstein_distance(a, b)))
    return {"group_mae": gm, "worst_group_mae": worst, "gap": gap, "cov": cov,
            "group_signed_err": gs, "signed_gap": signed_gap, "wasserstein_max": w1,
            "overall_mae": float(np.mean(err_abs))}


def _fast_worst_gap(err_abs, group_ids, groups):
    """Cheap (worst_group_mae, gap) for bootstrap inner loop (no Wasserstein)."""
    vals = []
    for g in groups:
        m = group_ids == g
        if m.sum() > 0:
            vals.append(err_abs[m].mean())
    vals = np.asarray(vals)
    return float(vals.max()), float(vals.max() - vals.min())


def boot_ci(err_signed, group_ids, groups, stat_key, nboot, rng):
    """Bootstrap 95% CI for worst_group_mae or gap (resample rows with replacement)."""
    n = len(err_signed)
    ea = np.abs(err_signed)
    j = 0 if stat_key == "worst_group_mae" else 1
    vals = np.empty(nboot)
    for b in range(nboot):
        idx = rng.integers(0, n, n)
        vals[b] = _fast_worst_gap(ea[idx], group_ids[idx], groups)[j]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_diff_ci(err_a, err_b, group_ids, groups, stat_key, nboot, rng):
    """Bootstrap 95% CI for (stat_a - stat_b) using the SAME resampled rows (paired)."""
    n = len(err_a)
    ea, eb = np.abs(err_a), np.abs(err_b)
    j = 0 if stat_key == "worst_group_mae" else 1
    d = np.empty(nboot)
    for b in range(nboot):
        idx = rng.integers(0, n, n)
        gi = group_ids[idx]
        d[b] = _fast_worst_gap(ea[idx], gi, groups)[j] - _fast_worst_gap(eb[idx], gi, groups)[j]
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# --------------------------- per-attribute ---------------------------
def analyse_attribute(attr, log, nboot, seed):
    fp = os.path.join(REPO, 'results', 'CFD', log, f'{attr}_predictions.xlsx')
    df = pd.read_excel(fp)
    df['Gender'] = df['Gender'].astype(str).str.strip()          # fix stray 'F '
    df['Ethnicity'] = df['Ethnicity'].astype(str).str.strip()
    y = df['label'].to_numpy(float)
    df['Averaging'] = df[BASE_MODELS].mean(axis=1)
    methods = BASE_MODELS + ['Averaging', 'StackingNet']
    err = {m: df[m].to_numpy(float) - y for m in methods}        # signed errors
    overall_mae = {m: float(np.mean(np.abs(err[m]))) for m in methods}
    best_base = min(BASE_MODELS, key=lambda m: overall_mae[m])

    axes = {'gender': (df['Gender'].to_numpy(), sorted(df['Gender'].unique())),
            'ethnicity': (df['Ethnicity'].to_numpy(), sorted(df['Ethnicity'].unique()))}

    res = {"attr": attr, "n": int(len(y)), "best_base": best_base,
           "overall_mae": overall_mae, "axes": {}}
    rng = np.random.default_rng(seed)
    for axis, (gids, groups) in axes.items():
        per_method = {}
        for m in methods:
            s = disparity_stats(err[m], gids, groups)
            per_method[m] = s
        # bootstrap CIs for the headline methods on worst-group MAE and gap
        ci = {}
        for m in ['Averaging', 'StackingNet', best_base]:
            ci[m] = {
                "worst_group_mae": boot_ci(err[m], gids, groups, "worst_group_mae", nboot, rng),
                "gap": boot_ci(err[m], gids, groups, "gap", nboot, rng),
            }
        # paired differences StackingNet - {Averaging, best_base} on worst-group MAE & gap
        diffs = {}
        for ref in ['Averaging', best_base]:
            diffs[ref] = {
                "worst_group_mae": paired_diff_ci(err['StackingNet'], err[ref], gids, groups,
                                                  "worst_group_mae", nboot, rng),
                "gap": paired_diff_ci(err['StackingNet'], err[ref], gids, groups,
                                      "gap", nboot, rng),
            }
        # (h) bias amplification: gap(method) - mean gap(base models)
        base_gap = np.mean([per_method[m]["gap"] for m in BASE_MODELS])
        base_worst = np.mean([per_method[m]["worst_group_mae"] for m in BASE_MODELS])
        ba = {m: {"gap": per_method[m]["gap"] - float(base_gap),
                  "worst": per_method[m]["worst_group_mae"] - float(base_worst)}
              for m in ['Averaging', 'StackingNet']}
        res["axes"][axis] = {"groups": groups, "per_method": per_method,
                             "ci": ci, "stackingnet_minus": diffs,
                             "bias_amplification": ba,
                             "base_mean_gap": float(base_gap),
                             "base_mean_worst": float(base_worst)}
    return res


# --------------------------- validation ---------------------------
def self_test():
    """Sanity checks on the metric implementations (must pass before trusting numbers)."""
    g = np.array(['A'] * 50 + ['B'] * 50)
    groups = ['A', 'B']
    # identical groups -> zero disparities
    e0 = np.r_[np.full(50, 0.2), np.full(50, 0.2)]
    s0 = disparity_stats(e0, g, groups)
    assert abs(s0['gap']) < 1e-9 and abs(s0['cov']) < 1e-9 and s0['wasserstein_max'] < 1e-9, s0
    # known gap: group A error 0.1, group B error 0.3 -> worst=0.3, gap=0.2
    e1 = np.r_[np.full(50, 0.1), np.full(50, 0.3)]
    s1 = disparity_stats(e1, g, groups)
    assert abs(s1['worst_group_mae'] - 0.3) < 1e-9 and abs(s1['gap'] - 0.2) < 1e-9, s1
    # overall MAE within [min group, max group]
    assert s1['group_mae']['A'] <= s1['overall_mae'] <= s1['group_mae']['B'] + 1e-9, s1
    # wasserstein between two constants = |difference|
    assert abs(wasserstein_distance(np.full(50, 0.1), np.full(50, 0.3)) - 0.2) < 1e-9
    print("[self_test] all metric sanity checks passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default='fairrun', help='CFD per-sample preds subdir under results/CFD/')
    ap.add_argument('--attrs', nargs='*', default=None)
    ap.add_argument('--smoke', action='store_true', help='one attribute only')
    ap.add_argument('--nboot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(REPO, 'results', 'analysis', 'expA_fairness_battery.json'))
    args = ap.parse_args()

    self_test()
    attrs = args.attrs or (['trustworthy'] if args.smoke else ATTRS)
    if args.smoke:
        args.nboot = 500

    all_res = {}
    for a in attrs:
        r = analyse_attribute(a, args.log, args.nboot, args.seed)
        all_res[a] = r
        e = r["axes"]["ethnicity"]
        sn, av, bb = e["per_method"]["StackingNet"], e["per_method"]["Averaging"], e["per_method"][r["best_base"]]
        d = e["stackingnet_minus"][r["best_base"]]["worst_group_mae"]
        print(f"[{a:11s}] ethnicity worst-group MAE: "
              f"StackingNet {sn['worst_group_mae']:.3f}  Averaging {av['worst_group_mae']:.3f}  "
              f"best-base({r['best_base']}) {bb['worst_group_mae']:.3f} | "
              f"gap SN {sn['gap']:.3f}/Avg {av['gap']:.3f}/base {bb['gap']:.3f} | "
              f"BA(gap) SN {e['bias_amplification']['StackingNet']['gap']:+.3f} | "
              f"SN-bestbase worst Δ {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]")

    if not args.smoke:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(all_res, f, indent=1)
        print(f"\nSaved -> {args.out}")


if __name__ == '__main__':
    main()
