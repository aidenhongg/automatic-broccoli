"""Generate training visualization graphs for the README."""

import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Load training log
epochs, train_losses, test_losses, test_accs, decimal_maes, lrs = [], [], [], [], [], []
with open("training_log.txt", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 6:
            continue
        epochs.append(int(row[0]))
        train_losses.append(float(row[1]))
        test_losses.append(float(row[2]))
        test_accs.append(float(row[3]))
        decimal_maes.append(float(row[4]))
        lrs.append(float(row[5]))

epochs = np.array(epochs)
train_losses = np.array(train_losses)
test_losses = np.array(test_losses)
test_accs = np.array(test_accs)
decimal_maes = np.array(decimal_maes)
lrs = np.array(lrs)

BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#059669"
PURPLE = "#7c3aed"
ORANGE = "#ea580c"
GRAY = "#6b7280"

# --- Graph 1: Train vs Test Loss ---
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(epochs, train_losses, color=BLUE, linewidth=2, label="Train Loss", marker="o", markersize=3)
ax.plot(epochs, test_losses, color=RED, linewidth=2, label="Test Loss", marker="s", markersize=3)

# Shade the overfitting gap
ax.fill_between(epochs, train_losses, test_losses, alpha=0.1, color=RED, label="Generalization Gap")

# Mark LR decay steps
lr_changes = [i for i in range(1, len(lrs)) if lrs[i] != lrs[i-1]]
for idx in lr_changes:
    ax.axvline(x=epochs[idx], color=GRAY, linestyle=":", alpha=0.6)
    ax.annotate(f"LR={lrs[idx]:.1e}", xy=(epochs[idx], train_losses[idx]),
                xytext=(5, 15), textcoords="offset points", fontsize=8, color=GRAY,
                arrowprops=dict(arrowstyle="-", color=GRAY, alpha=0.4))

ax.set_xlabel("Epoch")
ax.set_ylabel("BCE Loss")
ax.set_title("Training vs Test Loss (BCEWithLogitsLoss)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("graphs/train_test_loss.png", bbox_inches="tight")
plt.close()
print("Saved: graphs/train_test_loss.png")

# --- Graph 2: Bit-Level Accuracy + Random Baseline ---
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(epochs, test_accs * 100, color=GREEN, linewidth=2, label="Bit Accuracy", marker="o", markersize=3)
ax.axhline(y=50, color=GRAY, linestyle="--", linewidth=1.5, label="Random Baseline (50%)", alpha=0.7)
ax.axhline(y=100, color=GRAY, linestyle=":", linewidth=1, alpha=0.3)

# Annotate peak accuracy
peak_idx = np.argmax(test_accs)
ax.annotate(f"Peak: {test_accs[peak_idx]*100:.2f}%",
            xy=(epochs[peak_idx], test_accs[peak_idx]*100),
            xytext=(15, -20), textcoords="offset points", fontsize=10,
            arrowprops=dict(arrowstyle="->", color=GREEN),
            color=GREEN, fontweight="bold")

ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Bit-Level Prediction Accuracy")
ax.set_ylim(45, 75)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("graphs/bit_accuracy.png", bbox_inches="tight")
plt.close()
print("Saved: graphs/bit_accuracy.png")

# --- Graph 3: Decimal MAE ---
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(epochs, decimal_maes, color=PURPLE, linewidth=2, label="Decimal MAE", marker="o", markersize=3)

# Random baseline for 16-bit: expected MAE ~ 21845 (1/3 of 65535)
ax.axhline(y=65535/3, color=GRAY, linestyle="--", linewidth=1.5, label="Random Baseline (~21845)", alpha=0.7)

ax.set_xlabel("Epoch")
ax.set_ylabel("Mean Absolute Error")
ax.set_title("Decimal-Level Prediction Error (16-bit MSB-first)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("graphs/decimal_mae.png", bbox_inches="tight")
plt.close()
print("Saved: graphs/decimal_mae.png")

# --- Graph 4: Generalization Gap (train-test divergence) ---
fig, ax = plt.subplots(figsize=(7, 4.5))
gap = test_losses - train_losses
ax.plot(epochs, gap, color=ORANGE, linewidth=2, marker="o", markersize=3)
ax.fill_between(epochs, 0, gap, alpha=0.15, color=ORANGE)
ax.axhline(y=0, color=GRAY, linestyle="-", linewidth=0.5)

# Mark LR decay steps
for idx in lr_changes:
    ax.axvline(x=epochs[idx], color=GRAY, linestyle=":", alpha=0.6)

ax.set_xlabel("Epoch")
ax.set_ylabel("Test Loss - Train Loss")
ax.set_title("Generalization Gap Over Training")
fig.tight_layout()
fig.savefig("graphs/generalization_gap.png", bbox_inches="tight")
plt.close()
print("Saved: graphs/generalization_gap.png")

print("\nAll graphs generated successfully.")
