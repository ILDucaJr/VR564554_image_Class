#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python src/make_splits.py
python src/features_sklearn.py
python src/train_cnn.py --mode head
python src/train_cnn.py --mode full
python src/make_figures.py
