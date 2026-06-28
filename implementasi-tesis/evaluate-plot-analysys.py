import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non-interaktif (aman di Kaggle/script)
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

# =====================================================================
# KONFIGURASI  (samakan dengan yang dipakai saat training!)
#
# TASK: KLASIFIKASI 3 KELAS YoY_Close_Category (rendah/netral/tinggi)
# =====================================================================
OUTPUT_DIR = "/kaggle/working"               # folder buat NULIS plot (writable)

# Root tempat mencari file hasil training. Karena tiap plan dilatih di notebook
# TERPISAH, output-nya ter-mount di /kaggle/input/<nama-notebook>/...
# Cukup arahkan ke "/kaggle/input" — script akan cari file secara rekursif di
# semua sub-folder (mse-results/, training-logs/) di mana pun output di-mount.
SEARCH_ROOTS = ["/kaggle/input", "/kaggle/working"]

PLANS              = ["A", "B", "C", "D", "E"]  # evaluate semua plan (yang belum ada output-nya otomatis di-skip)
sec                = "amd"
bv                 = "yoy_close_category"
hist               = "no-hist"
TEST_CSV           = "idx-yoy-test-available.csv"  # CSV label test (acuan kebenaran; -available = subset .pt yang sudah ada)
learning_rate      = 1e-4
total_epochs       = 5
num_of_attn_heads  = 4
num_of_layers      = 4

# Klasifikasi
CLASS_NAMES = ["rendah", "netral", "tinggi"]     # index = label integer
NUM_CLASSES = len(CLASS_NAMES)
LABELS = list(range(NUM_CLASSES))
TARGET_COL = "label"                             # kolom integer kelas di CSV test
# =====================================================================

PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def find_file(filename):
    """Cari file bernama `filename` secara rekursif di semua SEARCH_ROOTS.
    Kembalikan path pertama yang ditemukan, atau None."""
    for root in SEARCH_ROOTS:
        hits = glob.glob(os.path.join(root, "**", filename), recursive=True)
        if hits:
            return hits[0]
    return None


def suffix_for(plan):
    return (f"{num_of_layers}_layers_{num_of_attn_heads}_heads_{sec}_{bv}"
            f"_eps{total_epochs}_lr={learning_rate:.1e}_plan-{plan}_{hist}")


def parse_training_log(path):
    """Ambil (epochs, train_losses, valid_losses) dari training log."""
    epochs, tr, vl = [], [], []
    pat = re.compile(r"Epoch\s+(\d+)\s+\|\s+Train Loss\s+([\d.eE+-]+)\s+\|\s+Valid Loss\s+([\d.eE+-]+)")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                epochs.append(int(m.group(1)))
                tr.append(float(m.group(2)))
                vl.append(float(m.group(3)))
    return epochs, tr, vl


def plot_confusion(cm, plan):
    """Heatmap confusion matrix 3 kelas."""
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    ax.set_title(f"Confusion Matrix — Plan {plan}")
    thr = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/confusion_matrix_plan-{plan}.png", dpi=150)
    plt.close(fig)


# Label test asli dari CSV (acuan kebenaran).
csv_path = find_file(TEST_CSV)
if csv_path is None:
    raise FileNotFoundError(
        TEST_CSV + " tidak ditemukan di " + str(SEARCH_ROOTS) +
        ". Pastikan dataset CSV sudah ditambahkan sebagai Input notebook.")
print(f"[found] test CSV: {csv_path}")
df_test = pd.read_csv(csv_path)

summary = []   # baris ringkasan per plan
histories = {} # plan -> (epochs, train, valid)

for plan in PLANS:
    sfx = suffix_for(plan)
    preds_path = find_file(f"test_preds_GATv2-dim-16x_{sfx}.csv")
    trues_path = find_file(f"test_trues_GATv2-dim-16x_{sfx}.csv")
    log_path   = find_file(f"training_log_GATv2-dim-16x_{sfx}.txt")

    if preds_path is None:
        print(f"[skip] plan {plan}: file prediksi 'test_preds_GATv2-dim-16x_{sfx}.csv' "
              f"tidak ditemukan di {SEARCH_ROOTS}")
        continue
    print(f"[found] plan {plan}: {preds_path}")

    y_pred = np.atleast_1d(np.loadtxt(preds_path, delimiter=",", dtype=int))

    # Kebenaran: pakai test_trues yang ditulis training (urutannya pasti sinkron
    # dengan prediksi). Fallback ke kolom label CSV bila file trues tak ada.
    if trues_path is not None:
        y_true = np.atleast_1d(np.loadtxt(trues_path, delimiter=",", dtype=int))
    else:
        y_true = np.asarray(df_test[TARGET_COL], dtype=int)

    n = min(len(y_pred), len(y_true))
    y_pred, y_true = y_pred[:n], y_true[:n]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    print(f"\n========== PLAN {plan} ==========")
    print(f"n_test     : {n}")
    print(f"accuracy   : {acc:.4f}")
    print(f"macro F1   : {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, labels=LABELS,
                                target_names=CLASS_NAMES, digits=4, zero_division=0))

    plot_confusion(cm, plan)
    summary.append({"plan": plan, "accuracy": acc, "macro_f1": macro_f1, "n_test": n})

    if log_path is not None and os.path.isfile(log_path):
        histories[plan] = parse_training_log(log_path)

# ---------------------------------------------------------------------
# Ringkasan tabel
# ---------------------------------------------------------------------
if not summary:
    print("\n[!] Tidak ada hasil yang bisa dievaluasi. Cek path & apakah training sudah selesai.")
    raise SystemExit

summary_df = pd.DataFrame(summary).set_index("plan")
print("\n================ RINGKASAN ================")
print(summary_df)
summary_df.to_csv(f"{PLOT_DIR}/summary_per_plan.csv")

plans_done = summary_df.index.tolist()

# ---------------------------------------------------------------------
# Grafik 1: Accuracy & macro-F1 per plan (grouped bar)
# ---------------------------------------------------------------------
x = np.arange(len(plans_done))
w = 0.38
plt.figure(figsize=(8, 4.5))
b1 = plt.bar(x - w/2, summary_df["accuracy"], w, label="Accuracy", color="steelblue")
b2 = plt.bar(x + w/2, summary_df["macro_f1"], w, label="Macro F1", color="darkorange")
for bars in (b1, b2):
    for b in bars:
        plt.text(b.get_x() + b.get_width()/2, b.get_height(),
                 f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
plt.xticks(x, plans_done)
plt.ylim(0, 1.05)
plt.ylabel("Skor")
plt.xlabel("Plan")
plt.title(f"Accuracy & Macro-F1 per Plan  |  {bv} (3 kelas)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/accuracy_f1_per_plan.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# Grafik 2: Kurva train/valid loss per epoch (semua plan overlaid)
# ---------------------------------------------------------------------
if histories:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for plan, (ep, tr, vl) in histories.items():
        ax1.plot(ep, tr, marker="o", label=f"plan {plan}")
        ax2.plot(ep, vl, marker="o", label=f"plan {plan}")
    ax1.set_title("Train Loss (cross-entropy) per Epoch")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_title("Valid Loss (cross-entropy) per Epoch")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Valid Loss"); ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.suptitle(f"History Training  |  {bv} (3 kelas)")
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/loss_curves.png", dpi=150)
    plt.close(fig)

print(f"\n[done] grafik & ringkasan tersimpan di: {PLOT_DIR}/")
print("  - accuracy_f1_per_plan.png")
print("  - confusion_matrix_plan-*.png")
print("  - loss_curves.png")
print("  - summary_per_plan.csv")
