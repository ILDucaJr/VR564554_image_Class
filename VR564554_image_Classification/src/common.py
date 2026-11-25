"""
Shared utilities for data loading, model setup, and evaluation.
"""
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLIT_DIR = ROOT / "splits"
RESULTS_DIR = ROOT / "results"

SEED = 42
NUM_CLASSES = 37
WEIGHTS = MobileNet_V3_Small_Weights.DEFAULT
DEFAULT_WORKERS = min(6, os.cpu_count() or 2)


# --- Reproducibility ---

def set_seed(seed: int = SEED, deterministic: bool = True) -> None:
    # Set seed across python, numpy and torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def seed_worker(worker_id: int) -> None:
    # Seed worker for reproducible augmentations
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    """Return the best available device: CUDA GPU, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def env_info(device=None) -> dict:
    """Collect library versions, device and seed.

    Saved in every config.json / extraction.json so that each result file records
    the exact environment that produced it.
    """
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "device": str(device) if device is not None else str(get_device()),
        "seed": SEED,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "weights": str(WEIGHTS),
    }


# --- Data ---

def get_transforms():
    """Return (train_transform, eval_transform).

    The evaluation transform is the one bundled with the pre-trained weights
    (resize 256, center crop 224, ImageNet mean/std), as required by the assignment.
    The training transform adds augmentation but ends with the same normalisation,
    so the network always sees inputs with the statistics it was pre-trained on.
    """
    eval_tf = WEIGHTS.transforms()
    size = eval_tf.crop_size[0]
    # Augmentation only for training
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=eval_tf.mean, std=eval_tf.std),
    ])
    return train_tf, eval_tf


def load_pets(split: str, transform=None, download: bool = True) -> OxfordIIITPet:
    """Load an official Oxford-IIIT Pet partition ("trainval" or "test") with breed labels."""
    return OxfordIIITPet(DATA_DIR, split=split, target_types="category",
                         transform=transform, download=download)


def load_split_indices():
    """Read the saved train/validation indices (positions inside the official trainval set)."""
    train_idx = np.loadtxt(SPLIT_DIR / "train_idx.txt", dtype=int)
    val_idx = np.loadtxt(SPLIT_DIR / "val_idx.txt", dtype=int)
    assert not set(train_idx) & set(val_idx), "train and validation overlap!"
    return train_idx, val_idx


def get_datasets(download: bool = True):
    """Build the four datasets used by every approach.

    "train"      training images with augmentation (used for gradient updates)
    "train_eval" the same training images without augmentation (to measure training accuracy)
    "val"        validation images, used only for model selection
    "test"       official test partition, used once per selected model
    """
    train_tf, eval_tf = get_transforms()
    # Two copies of trainval that differ only in the transform: augmentation must be
    # applied to the training images only, never to validation images
    trainval_aug = load_pets("trainval", train_tf, download)
    trainval_eval = load_pets("trainval", eval_tf, download=False)
    test = load_pets("test", eval_tf, download)
    train_idx, val_idx = load_split_indices()
    datasets = {
        "train": Subset(trainval_aug, train_idx),
        "train_eval": Subset(trainval_eval, train_idx),
        "val": Subset(trainval_eval, val_idx),
        "test": test,
    }
    return datasets, trainval_eval.classes


def get_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = DEFAULT_WORKERS,
               seed: int = SEED, persistent: bool = False):
    """DataLoader with a fixed random generator, so the shuffling order is reproducible."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, worker_init_fn=seed_worker,
                      generator=generator, pin_memory=torch.cuda.is_available(),
                      persistent_workers=persistent and num_workers > 0)


# --- Model ---

def build_model(mode: str) -> nn.Module:
    """MobileNetV3-Small pre-trained on ImageNet-1K, with 37 output classes.

    The classifier is Linear(576,1024) -> Hardswish -> Dropout -> Linear(1024,1000);
    only its last layer (classifier[3]) is replaced by Linear(1024,37).
    mode = "full":   every parameter is trainable (approach 1)
    mode = "head":   model.features is frozen, classifier.0 and classifier.3 are trained (approach 2)
    mode = "frozen": nothing is trainable, the model is only used to extract features (approach 3)
    """
    model = mobilenet_v3_small(weights=WEIGHTS)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)

    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif mode == "head":
        # Freeze the convolutional feature extractor; its BatchNorm layers are also
        # kept in eval mode during training (see train_cnn.train_one_epoch)
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif mode == "frozen":
        # Freeze all weights for feature extraction
        for p in model.parameters():
            p.requires_grad = False
    else:
        raise ValueError(f"unknown mode {mode}")
    return model


def count_parameters(model: nn.Module):
    """Return (number of trainable parameters, total number of parameters)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def trainable_layer_names(model: nn.Module):
    """Names of the layers that have at least one trainable parameter (e.g. classifier.0)."""
    names = {n.rsplit(".", 1)[0] for n, p in model.named_parameters() if p.requires_grad}
    return sorted(names)


# --- Evaluation ---

@torch.no_grad()
def predict(model: nn.Module, loader, device):
    """Return (true labels, predicted labels) for all images of a loader."""
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        logits = model(x.to(device))
        y_pred.append(logits.argmax(1).cpu().numpy())
        y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy and macro-averaged F1 (unweighted mean of the 37 per-class F1 scores)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def save_json(obj, path) -> None:
    """Write obj to a JSON file, creating the parent folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
