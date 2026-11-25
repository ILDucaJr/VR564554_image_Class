# Homework 2 – Image Classification (Computer Vision, Fall 2026)

**Tommaso Bellè – VR564554**

Transfer learning on the **Oxford-IIIT Pet** dataset (37 breeds) with an ImageNet-pre-trained
**MobileNetV3-Small**, comparing three strategies:

1. **Full fine-tuning** – the whole network is trained.
2. **Classifier-head training** – the convolutional feature extractor is frozen; only the two fully
   connected layers of the classifier are trained.
3. **Feature extraction + scikit-learn** – frozen 576-D features after global average pooling,
   classified with Logistic Regression, Linear SVM and Random Forest.

## Contents

| Path | Description |
|---|---|
| `report.pdf` | Final report (`report.tex` is its LaTeX source) |
| `VR564554_image_Classification/src/` | Python source code (data, training, feature extraction, evaluation, figures) |
| `VR564554_image_Classification/splits/` | Train/validation split indices (seed 42) and the split of every image |
| `VR564554_image_Classification/results/` | Saved metrics, predictions, tables, figures and model checkpoints |
| `VR564554_image_Classification/README.md` | Detailed documentation: environment, dataset, hyper-parameters, reproducibility |

## Results (held-out test set, 3,669 images)

| Method | Trainable params | Test accuracy | Test macro-F1 |
|---|---|---|---|
| Full fine-tuning | 1,555,781 | 0.856 | 0.854 |
| Classifier-head training | 628,773 | **0.870** | **0.869** |
| Features + Logistic Regression | 21,349 | 0.857 | 0.855 |
| Features + Linear SVM | 21,349 | 0.833 | 0.831 |
| Features + Random Forest | – | 0.826 | 0.822 |

Repeating approaches 1 and 2 with three seeds (42, 1, 2) gives 0.869 ± 0.002 test accuracy for
classifier-head training and 0.859 ± 0.003 for full fine-tuning; head training is better for every
seed (`VR564554_image_Classification/results/seeds_summary.csv`, reproduced by `run_seeds.sh`).

Confidence intervals, training times, McNemar tests and the full discussion are in the report
and in `VR564554_image_Classification/results/`.

## How to reproduce

Requirements: Python 3.10+ (tested with Python 3.13.3, PyTorch 2.15.0, TorchVision 0.29.2,
scikit-learn 1.9.1 on an Apple M4 Pro). The code uses a CUDA GPU or Apple MPS if available,
otherwise the CPU.

```bash
cd VR564554_image_Classification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
bash run_all.sh
```

On Windows, activate the environment with `.venv\Scripts\activate` and run the five
commands listed below instead of `run_all.sh`.

`run_all.sh` executes, in order:

```bash
python src/make_splits.py
python src/features_sklearn.py
python src/train_cnn.py --mode head
python src/train_cnn.py --mode full
python src/make_figures.py
```

| Step | What it does | Output |
|---|---|---|
| `make_splits.py` | Downloads the dataset (~800 MB) into `data/` and creates the stratified 80/20 train/validation split of the official `trainval` partition | `splits/`, `results/class_counts.csv` |
| `features_sklearn.py` | Approach 3: extracts frozen features and trains the scikit-learn classifiers | `results/features/`, `results/feat_*/` |
| `train_cnn.py --mode head` | Approach 2: classifier-head training | `results/head/` |
| `train_cnn.py --mode full` | Approach 1: full fine-tuning | `results/full/` |
| `make_figures.py` | Results table, confusion matrices, examples, learning curves, statistical tests | `results/*.png`, `results/*.csv`, `results/results_table.*` |

Total running time is about 20 minutes on an Apple M4 Pro, most of it for the dataset download;
it is longer on a CPU-only machine.

Random seeds are fixed (42) and deterministic algorithms are enabled; small differences can
still occur across hardware and library versions (see `VR564554_image_Classification/README.md`).
The test set is used only for the final evaluation of each selected model.

## Dataset

O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, *Cats and Dogs*, CVPR 2012 –
<https://www.robots.ox.ac.uk/~vgg/data/pets/>, licensed under CC BY-SA 4.0.
The images are not included in this repository; they are downloaded automatically by
`make_splits.py`.
