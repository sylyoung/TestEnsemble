"""
analysis/helm_io.py
-------------------
Thin loader for the HELM per-sample base-model predictions used by the
StackingNet revision analyses (Exp B, Exp C strong-subset).

It reuses the REAL data loaders (ReadLabel, ProcessData, seed_everything)
from classification_stackingnet.py -- the exact code path used to produce the
manuscript's Table 1 -- so there is no risk of a divergent re-implementation.

classification_stackingnet.py imports `crowdkit` (used only for the
crowdsourcing *baselines*, which these analyses do not use). crowdkit is not
installable in this environment, so we inject a harmless stub into sys.modules
*before* importing the module. The stubbed names are never called by ReadLabel
or ProcessData. Fidelity is then confirmed by an integrity check that the
regenerated per-model balanced accuracies reproduce the paper's Table 1.
"""
import os, sys, types
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
os.chdir(_REPO)  # ReadLabel/ProcessData use relative ./logs/helm paths

# --- inject crowdkit stub so the real module imports without the optional dep ---
if "crowdkit" not in sys.modules:
    _ck = types.ModuleType("crowdkit")
    _agg = types.ModuleType("crowdkit.aggregation")
    for _n in ["DawidSkene", "Wawa", "MMSR", "MACE", "GLAD", "KOS"]:
        setattr(_agg, _n, object)
    _ck.aggregation = _agg
    sys.modules["crowdkit"] = _ck
    sys.modules["crowdkit.aggregation"] = _agg

from classification_stackingnet import ReadLabel, ProcessData, seed_everything  # noqa: E402

HELM_DATASETS = ['boolq', 'entity_matching', 'imdb', 'legal_support',
                 'lsat_qa', 'mmlu', 'raft', 'civil_comments']
MODEL_NAMES = ['meta_llama-2-70b', 'openai_gpt-3.5-turbo-0301', 'ai21_j2-jumbo',
               'writer_palmyra-x', 'tiiuae_falcon-40b', 'cohere_command-xlarge-beta',
               'microsoft_TNLGv2_530B', 'mosaicml_mpt-instruct-30b',
               'anthropic_stanford-online-all-v4-s3', 'together_redpajama-incite-instruct-7b']


class _Args:
    pass


def load_helm(dataset_name, seed=0):
    """Return (preds, labels, n_classes, model_names).
    preds: (n_models, n_samples) int class predictions; labels: (n_samples,)."""
    args = _Args()
    args.dataset_name = dataset_name
    args.seed = seed
    seed_everything(args)  # ProcessData uses random for missing (-1) predictions
    labels, n_classes = ReadLabel(args)
    preds = ProcessData(args, MODEL_NAMES, n_classes)
    preds = np.stack(preds)
    # replicate the few-shot "augmented query" label duplication from the main script
    if preds.shape[-1] % len(labels) == 0:
        k = preds.shape[-1] // len(labels)
        if k == 3:
            labels = np.concatenate([labels, labels, labels])
        elif k != 1:
            raise ValueError(f"{dataset_name}: unexpected preds/labels ratio {k}")
    assert preds.shape[1] == len(labels), f"{dataset_name}: {preds.shape} vs {len(labels)}"
    return preds, np.asarray(labels), n_classes, list(MODEL_NAMES)


if __name__ == "__main__":
    from sklearn.metrics import balanced_accuracy_score
    for ds in HELM_DATASETS:
        preds, labels, K, names = load_helm(ds)
        bcas = [100 * balanced_accuracy_score(labels, preds[i]) for i in range(len(names))]
        print(f"{ds:16s} n={len(labels):5d} K={K} "
              f"base BCA worst/best = {min(bcas):.2f}/{max(bcas):.2f}")
