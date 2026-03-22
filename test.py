import os
import torch
import numpy as np
from collections import Counter

from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from train import (
    build_sequences,
    ROIDataset,
    CNNLSTM,
    CNNTransformer,
    FINAL_LABELS,
    DEVICE,
    BATCH_SIZE
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_DIR = "checkpoints"

EVAL_LABELS = ["eat_drink", "sitting", "moving"]
EVAL_INDICES = [FINAL_LABELS.index(l) for l in EVAL_LABELS]



def get_dataloaders():
    samples = build_sequences()

    samples = [s for s in samples if FINAL_LABELS[s["label"]] != "standing"]

    for s in samples:
        orig_name = FINAL_LABELS[s["label"]]
        s["label"] = EVAL_LABELS.index(orig_name)

    dataset = ROIDataset(samples)

    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    _, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    return val_loader


def evaluate(model, val_loader, model_name):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            logits = model(x)

            logits_filtered = logits[:, EVAL_INDICES]
            preds = torch.argmax(logits_filtered, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)

    print("\n==============================")
    print(f"{model_name.upper()} — VALIDATION RESULTS")
    print("==============================")
    print(f"Accuracy:    {acc:.4f}")
    print(f"F1 (macro):  {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=EVAL_LABELS))
    print("Confusion Matrix:")
    print(cm)

    plot_confusion_matrix(cm, model_name)

    return cm, acc, f1_macro


def plot_confusion_matrix(cm, model_name):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix — {model_name.upper()}", fontsize=14, fontweight="bold")

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=EVAL_LABELS,
        yticklabels=EVAL_LABELS,
        ax=axes[0],
        linewidths=0.5
    )
    axes[0].set_title("Raw counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=EVAL_LABELS,
        yticklabels=EVAL_LABELS,
        ax=axes[1],
        linewidths=0.5,
        vmin=0,
        vmax=1
    )
    axes[1].set_title("Normalised (recall per class)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()

    save_path = os.path.join(MODEL_DIR, f"confusion_{model_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")



def load_model(name):
    model = CNNLSTM() if name == "lstm" else CNNTransformer()
    path = os.path.join(MODEL_DIR, f"{name}.pt")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model = model.to(DEVICE)
    return model


def main():
    val_loader = get_dataloaders()

    for model_name in ["lstm", "transformer"]:
        print(f"\nEvaluating {model_name.upper()}")
        model = load_model(model_name)
        evaluate(model, val_loader, model_name)

if __name__ == "__main__":
    main()