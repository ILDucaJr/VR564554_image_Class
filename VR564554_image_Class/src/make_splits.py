"""
Downloads Oxford-IIIT Pet and creates a stratified 80/20 train-val split.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import RESULTS_DIR, SEED, SPLIT_DIR, load_pets, save_json


def main():
    """Download the dataset, split trainval 80/20 by breed and save split files and class counts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-frac", type=float, default=0.2)
    args = parser.parse_args()

    trainval = load_pets("trainval", download=True)
    test = load_pets("test", download=True)
    classes = trainval.classes
    labels = np.array(trainval._labels)

    # stratify=labels keeps the same breed proportions in training and validation
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)), test_size=args.val_frac,
        stratify=labels, random_state=args.seed)
    train_idx.sort()
    val_idx.sort()

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(SPLIT_DIR / "train_idx.txt", train_idx, fmt="%d")
    np.savetxt(SPLIT_DIR / "val_idx.txt", val_idx, fmt="%d")

    # Save split metadata for each image
    rows = []
    for split, ids, ds in [("train", train_idx, trainval),
                           ("val", val_idx, trainval),
                           ("test", np.arange(len(test)), test)]:
        for i in ids:
            lab = int(ds._labels[i])
            rows.append({"split": split, "index": int(i),
                         "image": Path(ds._images[i]).name,
                         "label": lab, "breed": classes[lab]})
    df = pd.DataFrame(rows)
    df.to_csv(SPLIT_DIR / "split_files.csv", index=False)

    # In Oxford-IIIT Pet, cat breeds start with an uppercase letter
    species = {}
    for _, r in df.iterrows():
        species.setdefault(r["breed"], "cat" if r["image"][0].isupper() else "dog")
    save_json({"classes": classes, "species": [species[c] for c in classes]},
              RESULTS_DIR / "classes.json")

    counts = (df.pivot_table(index="breed", columns="split", values="image", aggfunc="count")
                .reindex(classes)[["train", "val", "test"]])
    counts["total"] = counts.sum(axis=1)
    counts.loc["TOTAL"] = counts.sum()
    counts.to_csv(RESULTS_DIR / "class_counts.csv")

    print(counts.to_string())
    per_class = counts.drop(index="TOTAL")
    print("\nmin/max images per breed:")
    for col in ["train", "val", "test"]:
        print(f"  {col:5s}: {per_class[col].min()} - {per_class[col].max()}")


if __name__ == "__main__":
    main()
