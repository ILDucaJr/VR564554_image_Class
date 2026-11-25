"""
Approach 3: Feature extraction with frozen MobileNetV3 + scikit-learn classifiers.
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common import (DEFAULT_WORKERS, RESULTS_DIR, SEED, build_model, compute_metrics, env_info, get_datasets,
                    get_device, get_loader, save_json, set_seed)

FEAT_DIR = RESULTS_DIR / "features"

# Models and hyperparameter search grids
CLASSIFIERS = {
    "logreg": ("Features + Logistic Regression",
               lambda c: make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000)),
               [0.01, 0.1, 1.0, 10.0]),
    "linsvm": ("Features + Linear SVM",
               lambda c: make_pipeline(StandardScaler(), LinearSVC(C=c, max_iter=20000)),
               [0.001, 0.01, 0.1, 1.0]),
    "rf": ("Features + Random Forest",
           lambda n: RandomForestClassifier(n_estimators=n, n_jobs=-1, random_state=SEED),
           [200, 500]),
}


@torch.no_grad()
def extract(model, loader, device):
    """Map every image to the 576-D vector after global average pooling (before the classifier)."""
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        f = model.avgpool(model.features(x.to(device)))
        feats.append(torch.flatten(f, 1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def get_features(args, device):
    """Extract train/val/test features once and cache them in results/features/features.npz.

    The training features use the non-augmented images (train_eval), so every
    approach-3 classifier sees exactly the same deterministic inputs.
    """
    cache = FEAT_DIR / "features.npz"
    if args.reuse_features and cache.exists():
        import json
        data = np.load(cache)
        times = json.load(open(FEAT_DIR / "extraction.json"))["extraction_time_s"]
        return {k: (data[f"X_{k}"], data[f"y_{k}"]) for k in ["train", "val", "test"]}, times

    datasets, _ = get_datasets(download=not args.no_download)
    model = build_model("frozen").to(device)
    out, times = {}, {}
    for split, key in [("train", "train_eval"), ("val", "val"), ("test", "test")]:
        loader = get_loader(datasets[key], args.batch_size, False, args.num_workers, args.seed)
        t0 = time.perf_counter()
        out[split] = extract(model, loader, device)
        times[split] = time.perf_counter() - t0
        print(f"{split}: features {out[split][0].shape} in {times[split]:.1f}s")
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **{f"X_{k}": v[0] for k, v in out.items()}, **{f"y_{k}": v[1] for k, v in out.items()})
    save_json({"extraction_time_s": times, "feature_dim": int(out["train"][0].shape[1]),
               "env": env_info(device)}, FEAT_DIR / "extraction.json")
    return out, times


def n_classifier_params(clf):
    """Number of learned weights of a linear classifier (coef_ + intercept_).

    Returns None for the Random Forest, which is non-parametric (its size is the
    number of tree nodes, not a fixed set of weights).
    """
    est = clf[-1] if hasattr(clf, "steps") else clf
    if hasattr(est, "coef_"):
        return int(est.coef_.size + est.intercept_.size)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classifiers", nargs="+", default=list(CLASSIFIERS), choices=list(CLASSIFIERS))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--reuse-features", action="store_true")
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    feats, ext_times = get_features(args, device)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = feats["train"], feats["val"], feats["test"]

    for name in args.classifiers:
        # Grid search: fit on training features, choose the value with the best
        # validation macro-F1. The StandardScaler is inside the pipeline, so its
        # mean/std are computed on the training features only.
        display, factory, grid = CLASSIFIERS[name]
        print(f"\n== {display} ==")
        rows, best = [], None
        for h in grid:
            t0 = time.perf_counter()
            clf = factory(h).fit(X_tr, y_tr)
            fit_time = time.perf_counter() - t0
            val_f1 = f1_score(y_va, clf.predict(X_va), average="macro")
            rows.append({"hyperparam": h, "val_macro_f1": val_f1, "fit_time_s": fit_time})
            print(f"  param={h}: val macro-F1 {val_f1:.4f} ({fit_time:.1f}s)")
            if best is None or val_f1 > best[1]:
                best = (clf, val_f1, h, fit_time)
        clf, _, best_h, fit_time = best

        out_dir = RESULTS_DIR / f"feat_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / "grid.csv", index=False)
        y_pred_test = clf.predict(X_te)
        np.savez(out_dir / "test_predictions.npz", y_true=y_te, y_pred=y_pred_test)
        metrics = {
            "method": display,
            "trainable_params": n_classifier_params(clf),
            "total_params": None,
            # Total time = extraction + fitting
            "train_time_s": ext_times["train"] + fit_time,
            "feature_extraction_time_s": ext_times,
            "classifier_fit_time_s": fit_time,
            "grid_search_total_time_s": sum(r["fit_time_s"] for r in rows),
            "selected_hyperparam": best_h,
            "selection_rule": "max validation macro-F1",
            "train": compute_metrics(y_tr, clf.predict(X_tr)),
            "val": compute_metrics(y_va, clf.predict(X_va)),
            "test": compute_metrics(y_te, y_pred_test),
        }
        save_json(metrics, out_dir / "metrics.json")
        print(f"  selected {best_h} | train {metrics['train']} | test {metrics['test']}")


if __name__ == "__main__":
    main()
