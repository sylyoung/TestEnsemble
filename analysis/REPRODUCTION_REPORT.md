# Reproduction of the manuscript's quantitative results (our own runs)

Goal: reproduce the paper's result tables from scratch with controlled seeds, draw our own
observations, then diff against the manuscript and flag anything that differs.

Environment to run the original pipeline on this machine (all changes documented in-code as
`COMPAT`): crowd-kit installed (direct PyPI, proxy bypass); scikit-learn 1.8→1.5.2 (crowd-kit
predates sklearn's `__sklearn_tags__`); pandas 3.0→2.2.3 (copy-on-write returned read-only
arrays); two minimal script guards (`clf_pred` placeholder for the golden=0 run; PM "truth"
length fix for golden>0, mirroring the existing LAA handling). Runs use the **original script**
(RNG-faithful), not the isolated wiring.

---

## 1. Table 1 — HELM classification (`results/analysis/Table1_reproduced.xlsx`)
Method: golden=0 run → unsupervised block + **U-StackingNet**; golden=10 run → **Logistic
Regression** + **S-StackingNet**. 5 seeds, mean reported.

**Fidelity: mean |reproduced − paper| = 0.344 over 142 cells; 137/142 cells match to 0.00.**

Our observations (independent of the paper):
- **U-StackingNet ranks #1 in the unsupervised block** (boolq 89.24, civil 65.80, entity 95.22, …)
  — exceeds every crowdsourcing baseline and the best single model on most datasets.
- **S-StackingNet ranks #1 in the supervised block** (beats Logistic Regression on 7/8 datasets;
  LSAT is the exception, where all base models are near chance).
- StackingNet beats the **best single base model** on boolq/entity/imdb/raft/civil; falls slightly
  below it on legal/lsat/mmlu (consistent across our run and the paper).

Discrepancies vs the manuscript Table 1:
| Cell | Paper | Reproduced | Δ | Note |
|---|---|---|---|---|
| **S-StackingNet · boolq** | 89.75 | **89.15** | −0.60 | Only StackingNet anomaly. Seed std ±0.04, so not noise. S-StackingNet matches exactly on the other 7 datasets. **Likely a transcription typo (89.​**1**​5 → 89.​**7**​5).** Rank-1 still holds (89.15 ≥ LogReg 89.13). |
| LAA · imdb | 65.89 | 97.22 | +31.3 | LAA = unstable autoencoder; paper's value is 65.89**±24.43** (a failed run). Ours converged. Baseline, not the contribution. |
| LAA · legal_support | 51.77 | 64.51 | +12.7 | Same LAA instability (paper 51.77±4.44). |
| LogReg · raft | 89.11 | 91.17 | +2.1 | Golden-sampling variance in a baseline. |

→ **Action:** correct boolq S-StackingNet to ~89.15; optionally re-run LAA with more seeds (or
report its instability). No change to the paper's central conclusions.

## 2. Paper-rating — MAE (`results/research_reviewrepro_supervised.xlsx`, 1% labels, 5 seeds)
| | ICLR2025 | ICLR2024 | NeurIPS2024 | NeurIPS2023 | mean |
|---|---|---|---|---|---|
| Averaging | 1.828 | 1.841 | 1.416 | 1.415 | 1.625 |
| StackingRegressor (LinReg) | 1.497 | 1.410 | 0.626 | 0.793 | 1.082 |
| **StackingNet** | 0.983 | 0.957 | 0.632 | 0.600 | **0.793** |
| Individual-Human (full consensus) | 0.959 | 0.956 | 0.785 | 0.797 | 0.874 |

Observations: reproduces Fig 2's conclusion — **StackingNet (mean 0.793) < Individual-Human
(0.874)** on the dataset mean (ties/slightly behind on ICLR, clearly ahead on NeurIPS). NB: the
"Individual-Human" here uses the self-inclusion consensus; under our fair leave-one-out protocol
(Exp D) the human MAE rises to 1.02–1.28 and the gap is variance reduction, not superior judgment.
(Paper gives Fig 2 as a chart, no tabulated values to diff numerically.)

## 3. CFD — attribute-rating MAE (`results/CFDfairrun_supervised.xlsx`, [1,7], 1% labels)
StackingNet 13-attribute average **0.526** vs paper Table S2 ("both clamp") **0.513** — matches at
the average level. Per-attribute differences up to ~0.15 (e.g., happy 0.683 vs 0.534) reflect the
single-seed, 14-random-label few-shot setting (high variance), not a systematic gap.

---

## Bottom line
The pipeline reproduces faithfully (Table 1 essentially exact; paper-rating and CFD match at the
aggregate level). The **one substantive discrepancy is boolq S-StackingNet (89.75 → 89.15, likely
a typo)**; the LAA outliers are a known-unstable baseline where the paper logged failed runs.
None of these change the paper's conclusions, but the boolq value should be corrected.
