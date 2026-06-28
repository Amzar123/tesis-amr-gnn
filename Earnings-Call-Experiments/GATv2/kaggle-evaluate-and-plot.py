import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non-interaktif (aman di Kaggle/script)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

# =====================================================================
# KONFIGURASI  (samakan dengan yang dipakai saat training!)
# =====================================================================
OUTPUT_DIR = "/kaggle/working"               # folder buat NULIS plot (writable)

# Root tempat mencari file hasil training. Karena tiap plan dilatih di notebook
# TERPISAH, output-nya ter-mount di /kaggle/input/<nama-notebook>/...
# Cukup arahkan ke "/kaggle/input" — script akan cari file secara rekursif di
# semua sub-folder (mse-results/, training-logs/) di mana pun output di-mount.
# Bisa juga tambahkan "/kaggle/working" kalau ada plan yang dilatih di notebook ini.
SEARCH_ROOTS = ["/kaggle/input", "/kaggle/working"]

PLANS              = ["A", "B", "C", "D", "E"]  # plan yang mau dibandingkan
sec                = "tech-earnings-calls"
bv                 = "daily_price_change"
hist               = "no-hist"
learning_rate      = 1e-4
total_epochs       = 5
num_of_attn_heads  = 4
num_of_layers      = 4
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


def get_label(x):
    # Naik (>=0) -> 1, Turun (<0) -> 0
    return 1 if x >= 0 else 0


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


def read_test_mse(path):
    """Baca test_mse dari file mse_*.txt (format: test---bare---valid)."""
    try:
        with open(path) as f:
            first = f.readline().strip()
        return float(first.split("---bare---")[0])
    except Exception:
        return None


# Label test asli dari CSV (acuan kebenaran).
# Cari CSV-nya secara rekursif supaya tidak tergantung nama folder dataset.
csv_path = find_file("new-tech-2019-result.csv")
if csv_path is None:
    raise FileNotFoundError(
        "new-tech-2019-result.csv tidak ditemukan di " + str(SEARCH_ROOTS) +
        ". Pastikan dataset CSV sudah ditambahkan sebagai Input notebook.")
print(f"[found] test CSV: {csv_path}")
df_test = pd.read_csv(csv_path)
true_vals = np.asarray(df_test[bv], dtype=float)

summary = []   # baris ringkasan per plan
histories = {} # plan -> (epochs, train, valid)

for plan in PLANS:
    sfx = suffix_for(plan)
    preds_path = find_file(f"test_preds_GATv2-dim-16x_{sfx}.csv")
    mse_path   = find_file(f"mse_GATv2-dim-16x_{sfx}.txt")
    log_path   = find_file(f"training_log_GATv2-dim-16x_{sfx}.txt")

    if preds_path is None:
        print(f"[skip] plan {plan}: file prediksi 'test_preds_GATv2-dim-16x_{sfx}.csv' "
              f"tidak ditemukan di {SEARCH_ROOTS}")
        continue
    print(f"[found] plan {plan}: {preds_path}")

    preds = np.loadtxt(preds_path, delimiter=",", dtype=float)
    preds = np.atleast_1d(preds)

    n = min(len(preds), len(true_vals))
    y_pred = [get_label(preds[i])     for i in range(n)]
    y_true = [get_label(true_vals[i]) for i in range(n)]

    acc      = accuracy_score(y_true, y_pred)
    test_mse = read_test_mse(mse_path)

    print(f"\n========== PLAN {plan} ==========")
    print(f"n_test     : {n}")
    print(f"accuracy   : {acc:.4f}")
    print(f"test_mse   : {test_mse}")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    summary.append({"plan": plan, "accuracy": acc, "test_mse": test_mse, "n_test": n})

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
# Grafik 1: Akurasi per plan (bar)
# ---------------------------------------------------------------------
plt.figure(figsize=(7, 4))
bars = plt.bar(plans_done, summary_df["accuracy"], color="steelblue")
for b, v in zip(bars, summary_df["accuracy"]):
    plt.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}", ha="center", va="bottom")
plt.ylim(0, 1.05)
plt.ylabel("Accuracy (arah naik/turun)")
plt.xlabel("Plan")
plt.title(f"Perbandingan Akurasi per Plan  |  {bv}")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/accuracy_per_plan.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# Grafik 2: Test MSE per plan (bar) — makin kecil makin baik
# ---------------------------------------------------------------------
if summary_df["test_mse"].notna().any():
    plt.figure(figsize=(7, 4))
    vals = summary_df["test_mse"].fillna(0)
    bars = plt.bar(plans_done, vals, color="indianred")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width()/2, v, f"{v:.4f}", ha="center", va="bottom")
    plt.ylabel("Test MSE (makin kecil makin baik)")
    plt.xlabel("Plan")
    plt.title(f"Perbandingan Test MSE per Plan  |  {bv}")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/test_mse_per_plan.png", dpi=150)
    plt.close()

# ---------------------------------------------------------------------
# Grafik 3: Kurva train/valid loss per epoch (semua plan overlaid)
# ---------------------------------------------------------------------
if histories:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for plan, (ep, tr, vl) in histories.items():
        ax1.plot(ep, tr, marker="o", label=f"plan {plan}")
        ax2.plot(ep, vl, marker="o", label=f"plan {plan}")
    ax1.set_title("Train Loss per Epoch")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_title("Valid Loss per Epoch")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Valid Loss"); ax2.legend(); ax2.grid(True, alpha=0.3)
    # epoch 0 sering punya loss sangat besar -> pakai skala log biar kebaca
    ax1.set_yscale("log"); ax2.set_yscale("log")
    fig.suptitle(f"History Training  |  {bv}")
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/loss_curves.png", dpi=150)
    plt.close(fig)

print(f"\n[done] grafik & ringkasan tersimpan di: {PLOT_DIR}/")
print("  - accuracy_per_plan.png")
print("  - test_mse_per_plan.png")
print("  - loss_curves.png")
print("  - summary_per_plan.csv")
