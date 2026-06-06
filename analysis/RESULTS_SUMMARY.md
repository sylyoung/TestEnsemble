# StackingNet revision — new experiments: results summary

Analyses answering the *Advanced Science* reviewer comments. All run locally on CPU from
the existing base-model predictions (no GPU, no re-querying of models). Each experiment is
a self-contained script in `analysis/`; raw outputs in `results/analysis/*.json`; figures
in `results/analysis/figures/`.

**Posture:** appreciate → clarify → defend with data → improve writing. Numbers were
computed *before* claims were written; results that go against the original claims are
reported honestly (and mostly land on framings the reviewers asked for).

---

## Integrity (research-experiment-integrity gate)
- New analyses reuse the **real** code paths: HELM loaders (`ReadLabel`/`ProcessData`)
  and the supervised combiner (`Stacking`/`Stacking_Classification`) are imported, not
  re-implemented (crowdkit, needed only for unrelated baselines, is stubbed).
- **Base-model BCAs reproduce Table 1** (e.g., boolq 64.11/89.19 vs paper 64.13/89.21;
  sub-1-pt diffs come from the pipeline's seeded handling of missing `-1` predictions).
- **Metric implementations self-tested** (Exp A `self_test()` passes: zero-disparity,
  known-gap, Wasserstein-of-constants cases).
- **Discrepancy flagged:** boolq S-StackingNet is 89.75 in the manuscript Table 1 but the
  repo's *own* recorded run gives 89.14 and a faithful re-run gives 88.99. The manuscript
  number looks optimistic for boolq; do **not** defend 89.75. (Other datasets reproduce the
  Table-1 *pattern*: StackingNet beats best base on entity/imdb/raft/civil + regression,
  and is below it on legal/lsat/mmlu — as in the paper.)

---

## Exp A — fairness metric battery (R1.1, R1.2)  ·  `exp_A_fairness_battery.py` → `expA_fairness_battery.json` → `fig_A_fairness.png`
Battery (regression group-fairness, ethnicity 6 groups + gender 2, bootstrap 95% CIs):
worst-group MAE, max–min gap, bounded group loss, group-MAE CoV, mean-prediction gap,
Wasserstein-1 error disparity, per-group signed error, **directional bias amplification**.

**Findings (ethnicity, vs oracle best base, normalised [0,1] MAE):**
- StackingNet **reduces worst-group MAE on 10/13** attributes (significant, CI excludes 0),
  ties on 2 (feminine, happy), **significantly worse on exactly 1: trustworthy** (+0.026 [+0.006,+0.048]).
- **Bias amplification (gap) is negative on 12/13** — StackingNet *shrinks* between-group
  disparity everywhere except **trustworthy** (+0.011, amplifies).

**Use in reply:** StackingNet improves group-level error/fairness almost everywhere; the one
exception is the socially-sensitive **trustworthy** attribute — exactly where the
human-consensus "ground truth" encodes shared social perception (Oosterhof & Todorov 2008;
Caliskan 2017) and base models share directional bias that aggregation cannot cancel. R1.1+R1.2
become a *demonstrated* point. Reframe "enhances fairness" → "reduces group-wise error and,
where base errors are not directionally shared, narrows disparities; it has no group-conditional
mechanism, so we report disparities transparently rather than claiming normative debiasing."

## Exp B — inter-model dependence & diversity (R1.3)  ·  `exp_B_intermodel_dependence.py` → `expB_dependence.json` → `fig_B_dependence.png`
**Findings (errors ARE correlated — Assumption 2 is an idealization):**
- CFD residual-error correlation mean **+0.39** (trustworthy +0.395 → shared directional bias,
  links to Exp A exception).
- Paper-review residual correlation **+0.36 … +0.52** (the 6 LLMs share errors).
- HELM Kuncheva Q-statistic **+0.32 … +0.74**; error-indicator correlation +0.13 … +0.41.

**Use in reply:** Quantifies dependence per task (the matrix R1.3 requested). Key argument:
only **U-StackingNet** relies on conditional independence; **S-StackingNet is fit by ERM and
does not** — it absorbs correlation through supervision. Independence governs the *theory*,
not the *learned* method; we now state this and report the dependence.

## Exp B2 — controlled degradation (R1.3b)  ·  `exp_B2_controlled_degradation.py` → `expB2_degradation.json` → `fig_B2_degradation.png`
- **Synthetic:** averaging's MSE-reduction matches theory `(1+(M-1)ρ)/M` **exactly** (1/M at
  ρ=0 → 1 at ρ=1) — the paper's 1/M benefit holds only under independence and degrades smoothly.
- **Real injection (ICLR2025):** as duplicate correlated models flood the pool, **Averaging
  collapses** (gain over best base −0.61 → −1.32) while **StackingNet stays flat** (+0.255 →
  +0.256), holding ~86% of its weight on the original models.

**Use in reply:** Empirical proof of graceful degradation; shows supervised StackingNet is
*more* robust to violated independence than naive averaging (it down-weights redundancy).

## Exp C — dispersion–gain & strong-subset (R1.5)  ·  `exp_C_dispersion_gain.py` → `expC_dispersion_gain.json` → `fig_C_dispersion.png`
- **Dispersion–gain:** regression Spearman **ρ=+0.78 (p<0.001, n=17)** — gains rise with base
  dispersion (part of the gain *is* robust down-weighting of weak models — a feature).
  Classification: no clean trend (IMDB gains at near-zero dispersion; LSAT fails because all
  bases are near-chance) → gain is **not** merely weak-model pruning.
- **Strong-subset:** restricting paper-rating to the **top-3 strongest** models retains
  **80–95% of StackingNet's gain** over the best base.

**Use in reply:** Directly answers "relate gains to dispersion" and "how much remains when the
pool is uniformly strong." Also reframe the two settings (modern proprietary LLMs / subjective
regression vs. established HELM pool / objective classification) as complementary, not one
"cross-domain" pool.

## Exp D — fair leave-one-out human comparison (R1.4, R2.1)  ·  `exp_D_human_loo.py` → `expD_human_loo.json` → `fig_D_human.png`
The manuscript scores each human against a consensus that *includes* them; self-inclusion
understates single-human deviation. Fair protocol: score the held-out human AND the AI against
the consensus of the **other** reviewers.
- Manuscript **HUMAN_full 0.78–0.95** → fair **HUMAN_loo 1.02–1.28**.
- Under fair LOO, StackingNet (0.64–1.08) is still closer to consensus, **but a single best LLM
  (1.09–1.25) ≈ a single human (1.02–1.28)**.

**Use in reply:** The ensemble's edge over a single human is **variance reduction from
aggregation**, not superior scientific judgment (a single LLM is comparable to a single human).
Soften "approximate, and even surpass, expert-level judgment" → "matches individual reviewers in
consensus alignment via variance reduction; not a substitute for expert reasoning."

---

## Reproduce
```sh
python analysis/exp_A_fairness_battery.py          # needs results/CFD/fairrun/*.xlsx (see below)
python analysis/exp_B_intermodel_dependence.py
python analysis/exp_B2_controlled_degradation.py
python analysis/exp_C_dispersion_gain.py
python analysis/exp_D_human_loo.py
python analysis/make_figures.py
```
CFD per-sample predictions are regenerated by the existing pipeline:
`python regression_stackingnet_cfd.py --golden_num 1 --log fairrun`.
HELM data is loaded on demand via `analysis/helm_io.py` (stubs crowdkit, reuses real loaders).
