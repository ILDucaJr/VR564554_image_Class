#!/usr/bin/env bash
# Repeats approaches 1 and 2 with two extra seeds (the main runs use seed 42)
# to measure how much the results depend on the random seed.
set -e
cd "$(dirname "$0")"
for s in 1 2; do
    python src/train_cnn.py --mode head --seed "$s" --tag "_seed$s"
    python src/train_cnn.py --mode full --seed "$s" --tag "_seed$s"
done
python src/summarize_seeds.py
