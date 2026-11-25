"""
Approaches 1 & 2: Full fine-tuning and classifier-head training on MobileNetV3.
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from common import (DEFAULT_WORKERS, RESULTS_DIR, SEED, build_model, compute_metrics, count_parameters,
                    env_info, get_datasets, get_device, get_loader, predict, save_json,
                    set_seed, trainable_layer_names)

DEFAULTS = {
    "full": {"epochs": 15, "lr": 1e-3, "backbone_lr": 1e-4},
    "head": {"epochs": 15, "lr": 1e-3, "backbone_lr": None},
}
METHOD_NAMES = {"full": "Full fine-tuning", "head": "Classifier-head training"}


def parse_args():
    """Command-line options; unspecified epochs/learning rates take the per-mode DEFAULTS."""
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["full", "head"], required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float, help="learning rate of the classifier layers")
    p.add_argument("--backbone-lr", type=float, help="learning rate of model.features (full mode)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--tag", default="", help="suffix of the output folder, e.g. _lr3e-4")
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args()
    for k, v in DEFAULTS[args.mode].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


def train_one_epoch(model, loader, criterion, optimizer, device, mode):
    """One pass over the (augmented) training set. Returns (mean loss, accuracy)."""
    model.train()
    if mode == "head":
        # requires_grad=False does not stop BatchNorm from updating its running
        # mean/variance in train mode; eval mode keeps the backbone truly frozen
        model.features.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Loss, accuracy and macro-F1 on the validation set (no gradient, eval mode)."""
    model.eval()
    total_loss, n, y_true, y_pred = 0.0, 0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        n += x.size(0)
        y_true.append(y.cpu().numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    return total_loss / n, float((y_true == y_pred).mean()), float(f1_score(y_true, y_pred, average="macro"))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    out_dir = RESULTS_DIR / f"{args.mode}{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  output={out_dir.relative_to(RESULTS_DIR.parent)}")

    datasets, classes = get_datasets(download=not args.no_download)
    train_loader = get_loader(datasets["train"], args.batch_size, True, args.num_workers, args.seed, persistent=True)
    val_loader = get_loader(datasets["val"], args.batch_size, False, args.num_workers, args.seed, persistent=True)
    train_eval_loader = get_loader(datasets["train_eval"], args.batch_size, False, args.num_workers, args.seed)
    test_loader = get_loader(datasets["test"], args.batch_size, False, args.num_workers, args.seed)

    model = build_model(args.mode).to(device)
    n_trainable, n_total = count_parameters(model)
    layers = trainable_layer_names(model)
    print(f"trainable params: {n_trainable:,} / {n_total:,}")
    if args.mode == "head":
        print("trainable layers:", layers)

    if args.mode == "full":
        # Smaller learning rate for the pre-trained backbone, so that its filters are
        # adapted gently; larger one for the classifier, whose last layer is new
        param_groups = [
            {"params": model.features.parameters(), "lr": args.backbone_lr},
            {"params": model.classifier.parameters(), "lr": args.lr},
        ]
    else:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr}]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # Label smoothing (0.1) discourages over-confident predictions; it also means the
    # training loss cannot reach 0
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    history, best_f1, best_epoch = [], -1.0, -1
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, args.mode)
        va_loss, va_acc, va_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "val_macro_f1": va_f1,
                        "epoch_time_s": time.perf_counter() - t0})
        print(f"[{epoch:02d}/{args.epochs}] train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} F1 {va_f1:.3f} | {history[-1]['epoch_time_s']:.0f}s")
        # Model selection: keep the checkpoint with the best validation macro-F1
        # (the test set is never looked at during training)
        if va_f1 > best_f1:
            best_f1, best_epoch = va_f1, epoch
            torch.save(model.state_dict(), out_dir / "best.pt")
    train_time = time.perf_counter() - t_start
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    # Reload the selected checkpoint and evaluate it once on each split; training
    # accuracy is computed without augmentation (train_eval loader)
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    results = {}
    for split, loader in [("train", train_eval_loader), ("val", val_loader), ("test", test_loader)]:
        y_true, y_pred = predict(model, loader, device)
        results[split] = compute_metrics(y_true, y_pred)
        if split == "test":
            np.savez(out_dir / "test_predictions.npz", y_true=y_true, y_pred=y_pred)

    metrics = {
        "method": METHOD_NAMES[args.mode] + args.tag,
        "trainable_params": n_trainable,
        "total_params": n_total,
        "train_time_s": train_time,
        "best_epoch": best_epoch,
        "selection_rule": "max validation macro-F1",
        **results,
    }
    save_json(metrics, out_dir / "metrics.json")
    save_json({"args": vars(args), "loss": "CrossEntropyLoss", "optimizer": "AdamW",
               "scheduler": "CosineAnnealingLR", "trainable_layers": layers,
               "selected_checkpoint": str((out_dir / "best.pt").relative_to(RESULTS_DIR.parent)), "env": env_info(device)},
              out_dir / "config.json")
    print(f"\nbest epoch {best_epoch} | train {results['train']} | val {results['val']} | test {results['test']}")
    print(f"training time {train_time:.0f}s")


if __name__ == "__main__":
    main()
