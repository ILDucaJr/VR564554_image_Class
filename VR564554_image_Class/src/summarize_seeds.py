"""
Summarises approaches 1 and 2 over several seeds (results/head, results/head_seed1, ...).

Writes results/seeds_runs.csv (one row per run) and results/seeds_summary.csv
(mean and standard deviation of test accuracy and macro-F1 per approach, and the
head - full difference computed seed by seed).
"""
import json
import re

import pandas as pd

from common import RESULTS_DIR, SEED

MODES = {"full": "Full fine-tuning", "head": "Classifier-head training"}


def load_runs():
    """One row per (approach, seed); the main run in results/<mode>/ uses the default seed."""
    rows = []
    for d in sorted(RESULTS_DIR.iterdir()):
        m = re.fullmatch(r"(full|head)(?:_seed(\d+))?", d.name)
        if not m or not (d / "metrics.json").exists():
            continue
        metrics = json.load(open(d / "metrics.json"))
        rows.append({"approach": MODES[m.group(1)], "mode": m.group(1),
                     "seed": int(m.group(2)) if m.group(2) else SEED,
                     "best_epoch": metrics["best_epoch"],
                     "val_acc": metrics["val"]["accuracy"],
                     "test_acc": metrics["test"]["accuracy"],
                     "test_macro_f1": metrics["test"]["macro_f1"]})
    return pd.DataFrame(rows).sort_values(["mode", "seed"])


def main():
    runs = load_runs()
    runs.to_csv(RESULTS_DIR / "seeds_runs.csv", index=False)

    # Mean and sample standard deviation over seeds for each approach
    summary = (runs.groupby("approach")[["test_acc", "test_macro_f1"]]
               .agg(["mean", "std", "count"]))
    summary.columns = ["_".join(c) for c in summary.columns]

    # Paired difference: same seed, head minus full
    wide = runs.pivot(index="seed", columns="mode", values="test_acc").dropna()
    diff = wide["head"] - wide["full"]
    summary.loc["head - full (paired by seed)", ["test_acc_mean", "test_acc_std", "test_acc_count"]] = \
        [diff.mean(), diff.std(), len(diff)]
    summary.to_csv(RESULTS_DIR / "seeds_summary.csv")

    print(runs.to_string(index=False))
    print()
    print(summary.round(4).to_string())
    print(f"\nhead better than full in {(diff > 0).sum()} of {len(diff)} seeds")


if __name__ == "__main__":
    main()
