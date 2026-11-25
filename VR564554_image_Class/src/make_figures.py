"""
Generates comparison tables, plots, and confusion matrices from experiment results.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score

import re

from common import RESULTS_DIR, SEED, load_pets

N_BOOT = 1000
# Extra runs with other seeds (folders like head_seed1) are summarised separately
# by summarize_seeds.py and are not part of the main table and figures
SEED_RUN = re.compile(r"_seed\d+$")


def bootstrap_ci(y_true, y_pred, n=N_BOOT, seed=SEED):
    # 95% bootstrap CI for accuracy and macro-F1
    rng = np.random.default_rng(seed)
    N = len(y_true)
    accs, f1s = [], []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        accs.append((y_true[idx] == y_pred[idx]).mean())
        f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    return np.percentile(accs, [2.5, 97.5]), np.percentile(f1s, [2.5, 97.5])


def plot_confusion(cm, classes, title, path):
    """Save a row-normalised confusion matrix: cell (i, j) = fraction of breed i predicted as j."""
    cmn = cm / cm.sum(axis=1, keepdims=True)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted breed")
    ax.set_ylabel("True breed")
    ax.set_title(f"{title} (row-normalised)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def top_confusions(cm, classes, method, k=10):
    """The k most frequent (true breed, predicted breed) errors, off-diagonal cells only."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.dstack(np.unravel_index(np.argsort(-off, axis=None), off.shape))[0][:k]
    return [{"method": method, "true": classes[i], "predicted": classes[j],
             "count": int(off[i, j]), "fraction_of_true_class": off[i, j] / cm[i].sum()}
            for i, j in order if off[i, j] > 0]


def plot_examples(test_ds, y_true, y_pred, classes, title, path, n_each=8, seed=SEED):
    """Grid of randomly sampled correct and incorrect test predictions (fixed seed)."""
    rng = np.random.default_rng(seed)
    correct = np.flatnonzero(y_true == y_pred)
    wrong = np.flatnonzero(y_true != y_pred)
    picks = [(i, True) for i in rng.choice(correct, min(n_each, len(correct)), replace=False)] + \
            [(i, False) for i in rng.choice(wrong, min(n_each, len(wrong)), replace=False)]
    cols = 4
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.3))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (i, ok) in zip(axes.flat, picks):
        ax.imshow(Image.open(test_ds._images[i]).convert("RGB"))
        ax.set_title(f"T: {classes[y_true[i]]}\nP: {classes[y_pred[i]]}",
                     fontsize=8, color="green" if ok else "red")
    fig.suptitle(f"{title}: correct (green) and incorrect (red) test predictions")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_learning_curves(method_dirs):
    """Training/validation loss and accuracy per epoch for the CNN approaches (1 and 2)."""
    runs = [(d, json.load(open(d / "metrics.json"))["method"]) for d in method_dirs
            if (d / "history.csv").exists()]
    if not runs:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for d, name in runs:
        h = pd.read_csv(d / "history.csv")
        l1, = ax1.plot(h.epoch, h.train_loss, label=f"{name} train")
        ax1.plot(h.epoch, h.val_loss, "--", color=l1.get_color(), label=f"{name} val")
        l2, = ax2.plot(h.epoch, h.train_acc, label=f"{name} train (augmented)")
        ax2.plot(h.epoch, h.val_acc, "--", color=l2.get_color(), label=f"{name} val")
    ax1.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax2.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    ax1.legend(fontsize=7)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "learning_curves.png", dpi=200)
    plt.close(fig)


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def mcnemar_table(preds):
    """Paired exact McNemar test for every pair of methods on the same test images.

    Only the images where exactly one of the two methods is correct matter: under the
    null hypothesis (equal accuracy) each of them is equally likely to favour either
    method, so the count is Binomial(n, 0.5) and the p-value is an exact binomial test.
    """
    names = list(preds)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ok_a = preds[a]["y_pred"] == preds[a]["y_true"]
            ok_b = preds[b]["y_pred"] == preds[b]["y_true"]
            only_a, only_b = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
            p = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
            rows.append({"method_a": a, "method_b": b, "only_a_correct": only_a,
                         "only_b_correct": only_b, "p_value": p})
    return rows


def main():
    """Build results table, confusion matrices, examples, curves and statistical tests."""
    info = json.load(open(RESULTS_DIR / "classes.json"))
    classes, species = info["classes"], np.array(info["species"])
    test_ds = load_pets("test", download=False)

    method_dirs = sorted(d for d in RESULTS_DIR.iterdir()
                         if d.is_dir() and (d / "metrics.json").exists()
                         and not SEED_RUN.search(d.name))
    rows, confusions, per_class = [], [], {"breed": classes}
    for d in method_dirs:
        m = json.load(open(d / "metrics.json"))
        pred = np.load(d / "test_predictions.npz")
        y_true, y_pred = pred["y_true"], pred["y_pred"]
        # 95% CI over test images: resample the test set with replacement
        acc_ci, f1_ci = bootstrap_ci(y_true, y_pred)
        wrong = y_true != y_pred
        same_species = (species[y_true[wrong]] == species[y_pred[wrong]]).mean() if wrong.any() else np.nan
        rows.append({
            "method": m["method"],
            "trainable_params": m["trainable_params"],
            "train_time_s": m["train_time_s"],
            "train_acc": m["train"]["accuracy"],
            "val_acc": m["val"]["accuracy"],
            "test_acc": m["test"]["accuracy"],
            "test_acc_ci95": f"[{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]",
            "test_macro_f1": m["test"]["macro_f1"],
            "test_f1_ci95": f"[{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]",
            "train_test_gap": m["train"]["accuracy"] - m["test"]["accuracy"],
            "errors_within_same_species": same_species,
        })
        cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
        plot_confusion(cm, classes, m["method"], RESULTS_DIR / f"confusion_{d.name}.png")
        confusions += top_confusions(cm, classes, m["method"])
        per_class[m["method"]] = f1_score(y_true, y_pred, average=None, labels=range(len(classes)))
        plot_examples(test_ds, y_true, y_pred, classes, m["method"], RESULTS_DIR / f"examples_{d.name}.png")

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "results_table.csv", index=False)
    pd.DataFrame(confusions).to_csv(RESULTS_DIR / "top_confusions.csv", index=False)
    pd.DataFrame(per_class).to_csv(RESULTS_DIR / "per_class_f1.csv", index=False)
    plot_learning_curves(method_dirs)
    preds = {json.load(open(d / "metrics.json"))["method"]: np.load(d / "test_predictions.npz")
             for d in method_dirs}
    pd.DataFrame(mcnemar_table(preds)).to_csv(RESULTS_DIR / "mcnemar_tests.csv", index=False)

    # Export summary tables in Markdown and LaTeX
    header = ["Method", "Trainable params", "Time (s)", "Train acc", "Test acc [95% CI]",
              "Test macro-F1 [95% CI]", "Train-test gap"]
    lines = []
    for r in rows:
        params = "-" if r["trainable_params"] is None else f"{r['trainable_params']:,}"
        lines.append([r["method"], params, f"{r['train_time_s']:.0f}", fmt(r["train_acc"]),
                      f"{fmt(r['test_acc'])} {r['test_acc_ci95']}",
                      f"{fmt(r['test_macro_f1'])} {r['test_f1_ci95']}", fmt(r["train_test_gap"])])
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(l) + " |" for l in lines]
    (RESULTS_DIR / "results_table.md").write_text("\n".join(md) + "\n")
    tex = ["\\begin{tabular}{l" + "r" * (len(header) - 1) + "}", "\\toprule",
           " & ".join(h.replace("%", "\\%") for h in header) + " \\\\", "\\midrule"]
    tex += [" & ".join(c.replace(",", "{,}") if i == 1 else c for i, c in enumerate(l)) + " \\\\" for l in lines]
    tex += ["\\bottomrule", "\\end{tabular}"]
    (RESULTS_DIR / "results_table.tex").write_text("\n".join(tex) + "\n")

    print("\n".join(md))
    print("\nTop confusions:")
    print(pd.DataFrame(confusions).to_string(index=False))


if __name__ == "__main__":
    main()
