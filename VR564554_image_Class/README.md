# Homework 2 – Transfer learning on Oxford-IIIT Pet with MobileNetV3-Small

Comparison of three strategies for adapting an ImageNet-pre-trained
MobileNetV3-Small to the 37 breed classes of the Oxford-IIIT Pet dataset:

1. **Full fine-tuning**: every layer is trained.
2. **Classifier-head training**: `model.features` (convolutional feature extractor,
   including its BatchNorm statistics) is frozen; only the two fully connected
   layers `classifier.0` (Linear 576→1024) and `classifier.3` (Linear 1024→37) are trained.
   `classifier.1` (Hardswish) and `classifier.2` (Dropout 0.2) have no parameters.
3. **Feature extraction + scikit-learn**: the whole network is frozen; 576-D
   vectors are taken after `model.avgpool` (before the classifier) and used to
   train Logistic Regression, Linear SVM and Random Forest classifiers.

## Environment

The exact versions used for the reported results are saved automatically in
`results/*/config.json` and `results/features/extraction.json` (`env` field).
`requirements-lock.txt` pins the exact version of every package used; `requirements.txt`
lists only the direct dependencies with minimum versions.

| | Version |
|---|---|
| Python | 3.13.3 |
| PyTorch | 2.15.0 |
| TorchVision | 0.29.2 |
| scikit-learn | 1.9.1 |
| Device | Apple M4 Pro, PyTorch MPS backend (macOS 26.6.2) |

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

## Dataset

**Oxford-IIIT Pet** – O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar,
*Cats and Dogs*, CVPR 2012. <https://www.robots.ox.ac.uk/~vgg/data/pets/>.
License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).

The images are **not** included in this repository. They are downloaded
automatically by `torchvision.datasets.OxfordIIITPet` into `data/` the first
time `src/make_splits.py` is run (about 800 MB).

**Splits.** The official `trainval` (3,680 images) and `test` (3,669 images)
partitions are used. `trainval` is split 80/20 into training and validation,
stratified by breed (`sklearn.model_selection.train_test_split`, seed 42).
The indices are stored in `splits/train_idx.txt` and `splits/val_idx.txt`, and
`splits/split_files.csv` lists every image with its split. Images per breed:
`results/class_counts.csv`. The test partition is used only once per method,
for the final evaluation.

**Preprocessing.** Evaluation, validation and feature extraction use
`MobileNet_V3_Small_Weights.DEFAULT.transforms()` (resize 256, center crop 224,
ImageNet mean/std normalisation). Training images (approaches 1 and 2 only)
use RandomResizedCrop(224, scale 0.6–1.0), horizontal flip and mild colour
jitter, followed by the same normalisation. No class re-balancing is applied
(the classes are nearly balanced: see `class_counts.csv`).

## Reproducing the results

```bash
bash run_all.sh
```

or step by step:

```bash
python src/make_splits.py
python src/features_sklearn.py
python src/train_cnn.py --mode head
python src/train_cnn.py --mode full
python src/make_figures.py
```

To compare other hyper-parameters on the validation set:

```bash
python src/train_cnn.py --mode full --backbone-lr 3e-4 --tag _blr3e-4
python src/features_sklearn.py --reuse-features --classifiers logreg
```

To measure the effect of the random seed, `run_seeds.sh` repeats approaches 1 and 2
with seeds 1 and 2 (outputs in `results/head_seed1/`, `results/full_seed1/`, ...) and
summarises the three seeds (1, 2 and the main seed 42) in `results/seeds_runs.csv` and
`results/seeds_summary.csv`. These extra runs are not used by `make_figures.py`.

```bash
bash run_seeds.sh
```

## Training setup

| | Full fine-tuning | Head training | Feature extraction |
|---|---|---|---|
| Loss | Cross-entropy, label smoothing 0.1 | same | classifier-specific |
| Optimiser | AdamW, weight decay 1e-4 | AdamW, weight decay 1e-4 | – |
| Learning rate | 1e-4 (features), 1e-3 (classifier), cosine | 1e-3, cosine | – |
| Batch size | 32 | 32 | 64 (extraction) |
| Epochs | 15 | 15 | – |
| Hyper-parameter grid | – | – | LogReg C ∈ {0.01, 0.1, 1, 10}; LinearSVC C ∈ {0.001, 0.01, 0.1, 1}; RF trees ∈ {200, 500} |
| Feature scaling | – | – | StandardScaler fitted on training features (in the pipeline) |
| Model selection | epoch with best validation macro-F1 | same | hyper-parameter with best validation macro-F1 |

Selected epochs / hyper-parameters are written to `results/<method>/metrics.json`.

## Reproducibility

- Seed **42** for Python `random`, NumPy, PyTorch and the DataLoader generator
  (`src/common.py: set_seed`). DataLoader workers are seeded with `seed_worker`.
- `torch.use_deterministic_algorithms(True, warn_only=True)` and
  `cudnn.deterministic = True` are enabled.
- Remaining sources of variation: some GPU kernels (MPS/CUDA) have no
  deterministic implementation, so results may differ slightly across hardware,
  drivers and library versions.
- The test set is never used for early stopping, hyper-parameter tuning or
  feature scaling.

## Repository layout

```
src/common.py            seeds, transforms, datasets, model construction, metrics
src/make_splits.py       download data, stratified split, class counts
src/train_cnn.py         approaches 1 (full) and 2 (head)
src/features_sklearn.py  approach 3 (frozen features + scikit-learn)
src/make_figures.py      results table, confusion matrices, examples, curves
src/summarize_seeds.py   mean/std over seeds for approaches 1 and 2 (run_seeds.sh)
splits/                  split index files
results/                 metrics, predictions, figures, checkpoints (best.pt)
```

Checkpoints (`results/full/best.pt`, `results/head/best.pt`) are about 6 MB each.
The pre-trained weights are those distributed by TorchVision
(`MobileNet_V3_Small_Weights.IMAGENET1K_V1`).
