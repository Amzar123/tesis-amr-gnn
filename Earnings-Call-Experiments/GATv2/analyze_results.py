"""
Analisis hasil training GATv2 dari folder mse-results/ dan training-logs/.

Output:
  results/
    01_best_models_summary.txt      -- tabel best model per plan+heads
    02_loss_curves.png              -- train/valid loss curve best model
    03_loss_curves_all_plans.png    -- loss curve per plan (best lr masing-masing)
    04_confusion_matrix.png         -- confusion matrix best model (directional)
    05_pred_vs_true_scatter.png     -- scatter predicted vs true
    06_mse_heatmap.png              -- heatmap test MSE: plan x heads
    07_directional_accuracy_bar.png -- bar chart directional accuracy semua config
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (aman untuk semua OS)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, mean_squared_error
)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MSE_DIR     = os.path.join(BASE_DIR, "mse-results")
LOG_DIR     = os.path.join(BASE_DIR, "training-logs")
OUT_DIR     = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. Parse semua mse_GATv2_*.txt
# ─────────────────────────────────────────────
print("[1/6] Scanning mse-results/ ...")

FNAME_RE = re.compile(
    r"mse_GATv2_4_layers_(\d+)_heads_[\w\-]+_"
    r"([\w_]+?)_eps(\d+)_lr=([\d.e+\-]+)_plan-([A-Z])_([\w\-]+)\.txt"
)

records = []
for fname in os.listdir(MSE_DIR):
    if not fname.startswith("mse_GATv2_") or not fname.endswith(".txt"):
        continue
    m = FNAME_RE.match(fname)
    if not m:
        continue
    heads, target, epochs, lr, plan, hist = m.groups()
    fpath = os.path.join(MSE_DIR, fname)
    with open(fpath) as f:
        lines = f.readlines()
    try:
        test_mse, valid_mse = lines[0].strip().split("---bare---")
        best_epoch_line     = lines[1].strip()          # "X.xxx after epoch: N"
        best_epoch          = int(best_epoch_line.split("after epoch:")[-1].strip())
        records.append({
            "heads":      int(heads),
            "target":     target,
            "epochs":     int(epochs),
            "lr":         float(lr),
            "plan":       plan,
            "hist":       hist,
            "test_mse":   float(test_mse),
            "valid_mse":  float(valid_mse),
            "best_epoch": best_epoch,
            "fname":      fname,
        })
    except Exception:
        continue

df = pd.DataFrame(records)
print(f"    {len(df)} eksperimen ditemukan.")

if df.empty:
    print("Tidak ada hasil training yang bisa diparse. Periksa folder mse-results/.")
    sys.exit(1)

# ─────────────────────────────────────────────
# 2. Best model overall & per plan+heads
# ─────────────────────────────────────────────
print("[2/6] Mencari best model ...")

df_sorted      = df.sort_values("test_mse")
best_overall   = df_sorted.iloc[0]

# Best per (plan, heads)
best_per_group = (
    df.sort_values("test_mse")
      .groupby(["plan", "heads"], sort=False)
      .first()
      .reset_index()
      .sort_values(["plan", "heads"])
)

# ─── simpan summary teks ───────────────────────────────────────────────────
summary_path = os.path.join(OUT_DIR, "01_best_models_summary.txt")
with open(summary_path, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("BEST MODEL OVERALL\n")
    f.write("=" * 80 + "\n")
    f.write(f"  Plan        : {best_overall['plan']}\n")
    f.write(f"  Heads       : {best_overall['heads']}\n")
    f.write(f"  LR          : {best_overall['lr']:.2e}\n")
    f.write(f"  Epochs      : {best_overall['epochs']}\n")
    f.write(f"  Target      : {best_overall['target']}\n")
    f.write(f"  Test  MSE   : {best_overall['test_mse']:.8f}\n")
    f.write(f"  Valid MSE   : {best_overall['valid_mse']:.8f}\n")
    f.write(f"  Best Epoch  : {best_overall['best_epoch']}\n")
    f.write(f"  File        : {best_overall['fname']}\n")
    f.write("\n" + "=" * 80 + "\n")
    f.write("BEST PER PLAN & HEADS\n")
    f.write("=" * 80 + "\n")
    f.write(best_per_group[
        ["plan","heads","lr","epochs","test_mse","valid_mse","best_epoch"]
    ].to_string(index=False))
    f.write("\n")

print(f"    Saved: {summary_path}")

# ─── print ke terminal juga ────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  BEST MODEL OVERALL")
print("─" * 60)
print(f"  Plan={best_overall['plan']}  Heads={best_overall['heads']}  "
      f"LR={best_overall['lr']:.2e}  Epochs={best_overall['epochs']}")
print(f"  Test MSE  : {best_overall['test_mse']:.8f}")
print(f"  Valid MSE : {best_overall['valid_mse']:.8f}")
print(f"  Best Epoch: {best_overall['best_epoch']}")
print("─" * 60 + "\n")

# ─────────────────────────────────────────────
# helper: parse training log
# ─────────────────────────────────────────────
LOG_LINE_RE = re.compile(
    r"Epoch\s+(\d+)\s*\|\s*Train Loss\s*([\d.]+)\s*\|\s*Valid Loss\s*([\d.]+)"
)

def parse_log(mse_fname):
    # Derive training log filename langsung dari mse filename
    # "mse_GATv2_..." → "training_log_GATv2_..."
    stem = mse_fname.replace("mse_GATv2_", "training_log_GATv2_")
    path = os.path.join(LOG_DIR, stem)
    if not os.path.exists(path):
        return None, None
    train_l, valid_l = [], []
    with open(path) as f:
        for line in f:
            m = LOG_LINE_RE.search(line)
            if m:
                train_l.append(float(m.group(2)))
                valid_l.append(float(m.group(3)))
    return train_l, valid_l

# ─────────────────────────────────────────────
# 3. Loss curve — best model
# ─────────────────────────────────────────────
print("[3/6] Plotting loss curves ...")

def plot_loss_curve(train_losses, valid_losses, title, out_path,
                    best_epoch=None, highlight_color="#e74c3c"):
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(len(train_losses))
    ax.plot(epochs, train_losses, label="Train Loss",  color="#2980b9", linewidth=2)
    ax.plot(epochs, valid_losses, label="Valid Loss",  color="#e67e22", linewidth=2)
    if best_epoch is not None and best_epoch < len(valid_losses):
        ax.axvline(best_epoch, color=highlight_color, linestyle="--",
                   linewidth=1.5, label=f"Best epoch ({best_epoch})")
        ax.scatter([best_epoch], [valid_losses[best_epoch]],
                   color=highlight_color, zorder=5, s=80)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"    Saved: {out_path}")

# best model loss curve
b = best_overall
train_l, valid_l = parse_log(b["fname"])
if train_l:
    plot_loss_curve(
        train_l, valid_l,
        title=f"Loss Curve — Best Model  "
              f"(Plan {b['plan']} | {b['heads']} heads | LR={b['lr']:.2e})",
        out_path=os.path.join(OUT_DIR, "02_loss_curves.png"),
        best_epoch=b["best_epoch"]
    )
else:
    print(f"    [skip] training log untuk best model tidak ditemukan.")

# all plans loss curve (best lr each plan, 4 heads)
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
axes = axes.flatten()
plans = sorted(df["plan"].unique())
for idx, plan in enumerate(plans):
    ax = axes[idx]
    sub = df[(df["plan"] == plan) & (df["heads"] == 4)].sort_values("test_mse")
    if sub.empty:
        ax.set_visible(False)
        continue
    row = sub.iloc[0]
    tl, vl = parse_log(row["fname"])
    if not tl:
        ax.set_visible(False)
        continue
    ep = range(len(tl))
    ax.plot(ep, tl, label="Train", color="#2980b9", linewidth=1.8)
    ax.plot(ep, vl, label="Valid", color="#e67e22", linewidth=1.8)
    if row["best_epoch"] < len(vl):
        ax.axvline(row["best_epoch"], color="#e74c3c", linestyle="--", linewidth=1.2)
    ax.set_title(f"Plan {plan}  (LR={row['lr']:.1e})", fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
for idx in range(len(plans), len(axes)):
    axes[idx].set_visible(False)
fig.suptitle("Loss Curve per Plan (4 heads, best LR each)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_loss_curves_all_plans.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved: {os.path.join(OUT_DIR, '03_loss_curves_all_plans.png')}")

# ─────────────────────────────────────────────
# 4. Confusion matrix + directional accuracy
# ─────────────────────────────────────────────
print("[4/6] Confusion matrix ...")

def load_preds_trues(fname_mse):
    stem = fname_mse.replace("mse_", "test_preds_").replace(".txt", ".csv")
    preds_path = os.path.join(MSE_DIR, stem)
    stem_true  = fname_mse.replace("mse_", "test_trues_").replace(".txt", ".csv")
    trues_path = os.path.join(MSE_DIR, stem_true)
    if not os.path.exists(preds_path) or not os.path.exists(trues_path):
        return None, None
    preds = np.loadtxt(preds_path)
    trues = np.loadtxt(trues_path)
    return preds, trues

preds, trues = load_preds_trues(best_overall["fname"])

if preds is not None:
    y_pred = (preds >= 0).astype(int)
    y_true = (trues >= 0).astype(int)
    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc    = accuracy_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Turun (↓)", "Naik (↑)"],
                yticklabels=["Turun (↓)", "Naik (↑)"],
                linewidths=0.5, linecolor="gray")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(
        f"Confusion Matrix — Best Model\n"
        f"Plan {best_overall['plan']} | {best_overall['heads']} heads | "
        f"LR={best_overall['lr']:.2e}\n"
        f"Directional Accuracy = {acc:.4f}  ({int(acc*len(y_true))}/{len(y_true)})",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"    Saved: {os.path.join(OUT_DIR, '04_confusion_matrix.png')}")
    print(f"    Directional Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred,
          labels=[0, 1], target_names=["Turun (↓)", "Naik (↑)"],
          digits=4, zero_division=0))
else:
    print("    [skip] preds/trues file tidak ditemukan untuk best model.")

# ─────────────────────────────────────────────
# 5. Scatter predicted vs true
# ─────────────────────────────────────────────
print("[5/6] Scatter predicted vs true ...")

if preds is not None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(trues, preds, alpha=0.15, s=8, color="#2980b9")
    lim = max(abs(trues).max(), abs(preds).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("True Value",      fontsize=12)
    ax.set_ylabel("Predicted Value", fontsize=12)
    ax.set_title(
        f"Predicted vs True — Best Model\n"
        f"Plan {best_overall['plan']} | {best_overall['heads']} heads | "
        f"Test MSE={best_overall['test_mse']:.6f}",
        fontsize=11
    )
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_pred_vs_true_scatter.png"), dpi=150)
    plt.close()
    print(f"    Saved: {os.path.join(OUT_DIR, '05_pred_vs_true_scatter.png')}")

# ─────────────────────────────────────────────
# 6. Heatmap test MSE: plan × heads
# ─────────────────────────────────────────────
print("[6/6] MSE heatmap & directional accuracy bar ...")

pivot = (
    df.groupby(["plan", "heads"])["test_mse"]
      .min()
      .reset_index()
      .pivot(index="plan", columns="heads", values="test_mse")
)

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".5f", cmap="YlOrRd_r", ax=ax,
            linewidths=0.4, linecolor="gray", cbar_kws={"label": "Test MSE (lower=better)"})
ax.set_title("Best Test MSE per Plan × Heads\n(minimum over all LRs & epochs)", fontsize=12)
ax.set_xlabel("Attention Heads"); ax.set_ylabel("Plan")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "06_mse_heatmap.png"), dpi=150)
plt.close()
print(f"    Saved: {os.path.join(OUT_DIR, '06_mse_heatmap.png')}")

# ─── directional accuracy bar across all configs ──────────────────────────
dir_records = []
print("    Computing directional accuracy for all experiments ...")
for _, row in df.iterrows():
    p, t = load_preds_trues(row["fname"])
    if p is None or len(p) == 0 or len(t) == 0 or len(p) != len(t):
        continue
    acc = accuracy_score((t >= 0).astype(int), (p >= 0).astype(int))
    dir_records.append({
        "plan":  row["plan"],
        "heads": row["heads"],
        "lr":    row["lr"],
        "epochs": row["epochs"],
        "dir_acc": acc,
        "test_mse": row["test_mse"],
        "label": f"P{row['plan']}-H{row['heads']}-{row['lr']:.0e}",
    })

if dir_records:
    df_dir = pd.DataFrame(dir_records).sort_values("dir_acc", ascending=False)

    # top-30 bar chart
    top_n = min(30, len(df_dir))
    top   = df_dir.head(top_n)
    colors = ["#e74c3c" if row["plan"] == best_overall["plan"]
                           and row["heads"] == best_overall["heads"]
              else "#3498db"
              for _, row in top.iterrows()]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.barh(top["label"][::-1], top["dir_acc"][::-1],
                   color=colors[::-1], edgecolor="white", height=0.7)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Random baseline (0.5)")
    ax.set_xlabel("Directional Accuracy", fontsize=12)
    ax.set_title(f"Top-{top_n} Directional Accuracy  (Predicted Naik/Turun)", fontsize=12)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "07_directional_accuracy_bar.png"), dpi=150)
    plt.close()
    print(f"    Saved: {os.path.join(OUT_DIR, '07_directional_accuracy_bar.png')}")

    # best directional accuracy
    best_dir = df_dir.iloc[0]
    print(f"\n    Best Directional Accuracy : {best_dir['dir_acc']:.4f}")
    print(f"    Config: Plan={best_dir['plan']} Heads={best_dir['heads']} "
          f"LR={best_dir['lr']:.2e} Epochs={best_dir['epochs']}")

# ─────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Analisis selesai. Semua output di: {OUT_DIR}/")
print(f"{'='*60}")
print(f"  01_best_models_summary.txt")
print(f"  02_loss_curves.png")
print(f"  03_loss_curves_all_plans.png")
print(f"  04_confusion_matrix.png")
print(f"  05_pred_vs_true_scatter.png")
print(f"  06_mse_heatmap.png")
print(f"  07_directional_accuracy_bar.png")
print(f"{'='*60}\n")
