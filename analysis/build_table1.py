"""
Build our own reproduced Table 1 (HELM classification) into Excel, and a comparison
sheet against the manuscript's Table 1 values (hardcoded from the .tex).

Inputs (produced by the original script, the RNG-faithful path):
  results/repro_unsup_experiment_results.csv   (golden_num=0 -> unsupervised block + U-StackingNet)
  results/repro_sup_experiment_results.csv     (golden_num=10 -> supervised: StackingClassifier=LogReg, Stacking=S-StackingNet)

Outputs:
  results/analysis/Table1_reproduced.xlsx  (sheets: Reproduced, Paper, Diff)
"""
import os, re
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AN = os.path.join(REPO, 'results', 'analysis')
os.makedirs(AN, exist_ok=True)

DATASETS = ['boolq', 'civil_comments', 'entity_matching', 'imdb',
            'legal_support', 'lsat_qa', 'mmlu', 'raft']
# unsupervised-block methods (csv column -> table row label)
UNSUP = [('Voting', 'Voting'), ('WAwA', 'WAwA'), ('Dawid-Skene', 'Dawid-Skene'),
         ('M-MSR', 'M-MSR'), ('MACE', 'MACE'), ('GLAD', 'GLAD'), ('KOS', 'KOS'),
         ('SML', 'SML'), ('LA', 'LA'), ('LAA', 'LAA'), ('EBCC', 'EBCC'),
         ('PM', 'PM'), ('ZenCrowd', 'ZenCrowd'), ('Stacking', 'U-StackingNet')]

ROWS = (['Base-Worst', 'Base-Best'] + [lab for _, lab in UNSUP]
        + ['Logistic Regression', 'S-StackingNet'])

# Manuscript Table 1 values (BCA %), from 2026_StackingNet_AS_Li.tex
PAPER = {
 'boolq':          [64.13,89.21,87.29,88.26,88.01,88.41,87.74,88.23,88.24,88.24,88.25,87.73,88.26,88.76,88.16,89.24,89.13,89.75],
 'civil_comments': [50.91,65.12,64.39,64.79,65.24,64.98,64.13,64.92,64.84,64.84,64.86,65.46,65.21,64.30,64.75,65.80,50.00,65.81],
 'entity_matching':[61.74,95.76,92.01,93.95,90.33,92.86,91.83,94.70,92.96,89.91,93.25,90.20,88.17,95.19,94.41,95.22,91.71,96.21],
 'imdb':           [93.37,96.25,97.17,96.87,96.87,96.99,96.99,96.99,96.99,96.99,96.94,65.89,96.99,96.84,96.99,96.97,97.59,97.09],
 'legal_support':  [53.77,65.26,64.64,65.00,64.80,65.12,63.74,64.37,65.00,64.94,65.07,51.77,64.58,64.77,64.88,64.94,63.14,65.08],
 'lsat_qa':        [19.07,24.38,21.59,21.44,21.90,19.28,19.81,19.76,np.nan,21.59,19.98,20.55,21.11,20.10,19.72,21.38,22.62,19.85],
 'mmlu':           [35.51,59.48,54.60,54.82,53.90,54.58,55.10,54.74,np.nan,54.90,54.97,54.98,54.97,54.99,54.97,54.42,57.73,58.80],
 'raft':           [76.77,88.58,88.06,88.10,89.62,87.17,89.75,87.30,87.17,89.05,87.17,89.56,88.56,87.13,87.30,88.33,89.11,89.18],
}


def mean_of(cell):
    if not isinstance(cell, str):
        return np.nan
    m = re.match(r"([\d.]+)", cell)
    return float(m.group(1)) if m else np.nan


def load_block(csv):
    if not os.path.exists(csv):
        return None
    df = pd.read_csv(csv).set_index('Dataset')
    return df


def build():
    unsup = load_block(os.path.join(REPO, 'results', 'repro_unsup_experiment_results.csv'))
    sup = load_block(os.path.join(REPO, 'results', 'repro_sup_experiment_results.csv'))

    repro = pd.DataFrame(index=ROWS, columns=DATASETS, dtype=float)
    for ds in DATASETS:
        if unsup is not None and ds in unsup.index:
            w, a, b = [float(x) for x in str(unsup.loc[ds, 'Single']).split('-')]
            repro.loc['Base-Worst', ds] = w
            repro.loc['Base-Best', ds] = b
            for col, lab in UNSUP:
                repro.loc[lab, ds] = mean_of(unsup.loc[ds, col])
        if sup is not None and ds in sup.index:
            repro.loc['Logistic Regression', ds] = mean_of(sup.loc[ds, 'StackingClassifier'])
            repro.loc['S-StackingNet', ds] = mean_of(sup.loc[ds, 'Stacking'])

    paper = pd.DataFrame({ds: PAPER[ds] for ds in DATASETS}, index=ROWS)
    diff = repro - paper

    out = os.path.join(AN, 'Table1_reproduced.xlsx')
    with pd.ExcelWriter(out, engine='openpyxl') as xl:
        repro.round(2).to_excel(xl, sheet_name='Reproduced')
        paper.round(2).to_excel(xl, sheet_name='Paper')
        diff.round(2).to_excel(xl, sheet_name='Diff(repro-paper)')
    print(f"Saved -> {out}")

    # console: flag |diff| > 0.5 (beyond seed noise)
    print("\n== |reproduced - paper| > 0.5 (potential discrepancies) ==")
    flagged = 0
    for r in ROWS:
        for ds in DATASETS:
            d = diff.loc[r, ds]
            if pd.notna(d) and abs(d) > 0.5:
                print(f"  {r:20s} {ds:16s} repro {repro.loc[r,ds]:.2f}  paper {paper.loc[r,ds]:.2f}  diff {d:+.2f}")
                flagged += 1
    if not flagged:
        print("  none (all within 0.5)")
    # overall agreement
    mask = diff.notna()
    print(f"\nCells compared: {int(mask.values.sum())}  | mean |diff| = {diff.abs().mean().mean():.3f}")
    return repro, paper, diff


if __name__ == '__main__':
    build()
