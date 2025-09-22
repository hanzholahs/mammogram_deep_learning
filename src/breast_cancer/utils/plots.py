import os
from typing import List
import math
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from PIL import Image
from tensorflow.data import Dataset
from tqdm import tqdm


def plot_image_samples(
    ds: Dataset,
    label_mapping: dict[str, str],
    nrows: int = 3,
    ncols: int = 4,
    figsize: tuple[int, int] = (12, 8),
):
    n_images = nrows * ncols
    img_batch, lbl_batch = next(iter(ds))

    assert len(img_batch) >= n_images, (
        "`ds` batch size must be bigger than `ncols` times `nrows`."
    )

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = axs.flatten()

    for i in range(n_images):
        axs[i].imshow(img_batch[i], cmap="gray")
        axs[i].set_title(label_mapping[lbl_batch[i].numpy()])
        axs[i].axis("off")
    fig.tight_layout()
    plt.show()

def plot_history(history):
    hist = history.history if hasattr(history, "history") else history
    epochs = range(1, len(hist["loss"]) + 1)

    # Metrics to pair with their val_ counterparts
    base_metrics = {
        "loss": "Loss",
        "auc": "AUC",
        "binary_accuracy": "Binary Accuracy",
        "precision": "Precision",
        "recall": "Recall"
    }
    n_plots = len(base_metrics) + 1   # +1 for LR
    n_cols = 3
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()

    for i, (m, l) in enumerate(base_metrics.items()):
        ax = axes[i]
        ax.plot(epochs, hist[m], label=f"train_{m}")
        ax.plot(epochs, hist[f"val_{m}"], label=f"val_{m}")
        ax.set_title(l)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(l)
        ax.legend()
        ax.grid(True)

    # learning rate
    ax = axes[len(base_metrics)]
    ax.plot(epochs, hist["lr"], label="learning_rate")
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.grid(True)

    # Hide any unused subplots
    for j in range(len(base_metrics)+1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Training History")
    plt.tight_layout()
    plt.show()
