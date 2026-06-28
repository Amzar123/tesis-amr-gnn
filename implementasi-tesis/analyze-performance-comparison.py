"""
Analisis performa & perbandingan hasil — KLASIFIKASI 3 KELAS YoY_Close_Category
(rendah=0 / netral=1 / tinggi=2).

Beda dengan `evaluate-plot-analysys.py` (plot dasar per plan), script ini fokus
ke ANALISIS MENDALAM & PERBANDINGAN antar-plan:
  - metrik lengkap: accuracy, balanced accuracy, macro/weighted precision-recall-F1,
    Cohen's kappa, MCC
  - rincian per-kelas (precision/recall/F1/support) tiap plan
  - perbandingan vs baseline trivial (majority-class & stratified-random) supaya
    ketahuan model benar-benar belajar atau cuma menebak kelas mayoritas
  - tabel ranking antar-plan (urut macro-F1)
  - confusion matrix ternormalisasi (proporsi per baris/kelas aktual)
  - heatmap F1 per-kelas (plan x kelas)
  - laporan tersimpan: summary_comparison.csv, per_class_metrics.csv, report.md

Sumber kebenaran: file `test_trues_*.csv` (ditulis training, urutannya sinkron
dengan prediksi). Prediksi: `test_preds_*.csv`. Keduanya berisi integer kelas.

SEARCH_ROOTS bisa di-override via env (mis. untuk tes lokal):
    SEARCH_ROOTS=/path/a,/path/b python analyze-performance-comparison.py
"""
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score,
    matthews_corrcoef, precision_recall_fscore_support, confusion_matrix,
    classification_report,
)

# =====================================================================
# KONFIGURASI  (samakan dengan saat training!)
# =====================================================================
OUTPUT_DIR   = os.environ.get("OUTPUT_DIR", "/kaggle/working")
SEARCH_ROOTS = os.environ.get("SEARCH_ROOTS", "/kaggle/input,/kaggle/working").split(",")

PLANS              = ["A", "B", "C", "D", "E"]
sec                = "amd"
bv                 = "yoy_close_category"
hist               = "no-hist"
learning_rate      = 1e-4
total_epochs       = 5
num_of_attn_heads  = 4
num_of_layers      = 4

CLASS_NAMES = ["rendah", "netral", "tinggi"]   # index = label integer
NUM_CLASSES = len(CLASS_NAMES)
LABELS = list(range(NUM_CLASSES))
# =====================================================================

ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)


def find_file(filename):
    for root in SEARCH_ROOTS:
        hits = glob.glob(os.path.join(root, "**", filename), recursive=True)
        if hits:
            return hits[0]
    return None


def suffix_for(plan):
    return (f"{num_of_layers}_layers_{num_of_attn_heads}_heads_{sec}_{bv}"
            f"_eps{total_epochs}_lr={learning_rate:.1e}_plan-{plan}_{hist}")


def parse_training_log(path):
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


# ---------------------------------------------------------------------
# Metrik
# ---------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    """Kumpulan metrik agregat untuk satu set prediksi."""
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    return {
        "accuracy":      accuracy_score(y_true, y_pred),
        "balanced_acc":  balanced_accuracy_score(y_true, y_pred),
        "precision_macro": p_macro,
        "recall_macro":    r_macro,
        "f1_macro":        f_macro,
        "precision_weighted": p_w,
        "recall_weighted":    r_w,
        "f1_weighted":        f_w,
        "cohen_kappa":   cohen_kappa_score(y_true, y_pred, labels=LABELS),
        "mcc":           matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0,
    }


def per_class_table(y_true, y_pred, plan):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    return pd.DataFrame({
        "plan": plan,
        "kelas": CLASS_NAMES,
        "precision": p, "recall": r, "f1": f, "support": s,
    })


def baseline_metrics(y_true):
    """Baseline trivial sebagai pembanding: selalu tebak kelas mayoritas."""
    y_true = np.asarray(y_true)
    counts = np.bincount(y_true, minlength=NUM_CLASSES)
    majority = int(counts.argmax())
    y_major = np.full_like(y_true, majority)
    m = compute_metrics(y_true, y_major)
    m["keterangan"] = f"selalu '{CLASS_NAMES[majority]}'"
    return m


def plot_confusion_normalized(cm, plan):
    cmn = cm.astype(float)
    row = cmn.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    cmn = cmn / row
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    ax.set_title(f"Confusion Matrix (ternormalisasi) — Plan {plan}")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(f"{ANALYSIS_DIR}/confusion_norm_plan-{plan}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Kumpulkan hasil tiap plan
# ---------------------------------------------------------------------
rows = []                 # metrik agregat per plan
per_class_frames = []     # rincian per kelas per plan
histories = {}            # plan -> (epochs, train, valid)
baseline_row = None

for plan in PLANS:
    sfx = suffix_for(plan)
    preds_path = find_file(f"test_preds_GATv2-dim-16x_{sfx}.csv")
    trues_path = find_file(f"test_trues_GATv2-dim-16x_{sfx}.csv")
    log_path   = find_file(f"training_log_GATv2-dim-16x_{sfx}.txt")

    if preds_path is None or trues_path is None:
        print(f"[skip] plan {plan}: file preds/trues belum ada "
              f"(preds={preds_path is not None}, trues={trues_path is not None})")
        continue

    y_pred = np.atleast_1d(np.loadtxt(preds_path, delimiter=",", dtype=int))
    y_true = np.atleast_1d(np.loadtxt(trues_path, delimiter=",", dtype=int))
    n = min(len(y_pred), len(y_true))
    y_pred, y_true = y_pred[:n], y_true[:n]
    print(f"[found] plan {plan}: n_test={n}  ({preds_path})")

    m = compute_metrics(y_true, y_pred)
    m["plan"] = plan
    m["n_test"] = n
    rows.append(m)
    per_class_frames.append(per_class_table(y_true, y_pred, plan))
    plot_confusion_normalized(confusion_matrix(y_true, y_pred, labels=LABELS), plan)

    if baseline_row is None:   # baseline dihitung sekali dari y_true (sama tiap plan)
        baseline_row = baseline_metrics(y_true)
        baseline_row["plan"] = "BASELINE"
        baseline_row["n_test"] = n

    if log_path is not None and os.path.isfile(log_path):
        histories[plan] = parse_training_log(log_path)

    print(classification_report(y_true, y_pred, labels=LABELS,
                                target_names=CLASS_NAMES, digits=4, zero_division=0))

if not rows:
    print("\n[!] Tidak ada hasil. Pastikan training sudah menulis test_preds_*/test_trues_* "
          f"dan SEARCH_ROOTS benar: {SEARCH_ROOTS}")
    raise SystemExit

# ---------------------------------------------------------------------
# Tabel perbandingan (urut macro-F1, terbaik di atas)
# ---------------------------------------------------------------------
metric_cols = ["accuracy", "balanced_acc", "precision_macro", "recall_macro",
               "f1_macro", "f1_weighted", "cohen_kappa", "mcc", "n_test"]
summary = pd.DataFrame(rows).set_index("plan")[metric_cols].sort_values("f1_macro", ascending=False)

# Sisipkan baris baseline untuk konteks
base_df = pd.DataFrame([baseline_row]).set_index("plan")[
    [c for c in metric_cols if c in baseline_row]]
summary_with_base = pd.concat([summary, base_df])

print("\n================ PERBANDINGAN ANTAR-PLAN (urut macro-F1) ================")
with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
    print(summary_with_base)

summary_with_base.to_csv(f"{ANALYSIS_DIR}/summary_comparison.csv")

per_class_df = pd.concat(per_class_frames, ignore_index=True)
per_class_df.to_csv(f"{ANALYSIS_DIR}/per_class_metrics.csv", index=False)

best_plan = summary.index[0]
best_f1 = summary.loc[best_plan, "f1_macro"]
base_f1 = baseline_row["f1_macro"]
best_acc = summary.loc[best_plan, "accuracy"]
base_acc = baseline_row["accuracy"]

# ---------------------------------------------------------------------
# Grafik 1: Perbandingan metrik utama per plan (grouped bar) + garis baseline
# ---------------------------------------------------------------------
plot_metrics = ["accuracy", "f1_macro", "f1_weighted", "balanced_acc"]
plans_done = summary.index.tolist()
x = np.arange(len(plans_done))
w = 0.8 / len(plot_metrics)
plt.figure(figsize=(max(8, 1.6 * len(plans_done)), 5))
for k, mname in enumerate(plot_metrics):
    plt.bar(x + (k - (len(plot_metrics) - 1) / 2) * w, summary[mname], w, label=mname)
plt.axhline(base_acc, ls="--", color="gray", lw=1, label=f"baseline acc ({base_acc:.3f})")
plt.xticks(x, plans_done)
plt.ylim(0, 1.05)
plt.ylabel("Skor"); plt.xlabel("Plan")
plt.title(f"Perbandingan Metrik per Plan  |  {bv} (3 kelas)")
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(f"{ANALYSIS_DIR}/metrics_comparison.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# Grafik 2: Heatmap F1 per-kelas (baris=plan, kolom=kelas)
# ---------------------------------------------------------------------
f1_pivot = per_class_df.pivot(index="plan", columns="kelas", values="f1").reindex(plans_done)[CLASS_NAMES]
fig, ax = plt.subplots(figsize=(1.4 * NUM_CLASSES + 2, 0.7 * len(plans_done) + 2))
im = ax.imshow(f1_pivot.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES)
ax.set_yticks(range(len(plans_done))); ax.set_yticklabels(plans_done)
ax.set_xlabel("Kelas"); ax.set_ylabel("Plan")
ax.set_title("F1 per Kelas per Plan")
for i in range(len(plans_done)):
    for j in range(NUM_CLASSES):
        v = f1_pivot.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if v < 0.5 else "black", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(f"{ANALYSIS_DIR}/f1_per_class_heatmap.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------
# Grafik 3: Kurva train/valid loss (kalau ada log)
# ---------------------------------------------------------------------
if histories:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for plan, (ep, tr, vl) in histories.items():
        ax1.plot(ep, tr, marker="o", label=f"plan {plan}")
        ax2.plot(ep, vl, marker="o", label=f"plan {plan}")
    ax1.set_title("Train Loss (cross-entropy)"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax2.set_title("Valid Loss (cross-entropy)"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3); ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.suptitle(f"History Training  |  {bv}")
    fig.tight_layout()
    fig.savefig(f"{ANALYSIS_DIR}/loss_curves.png", dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------
# Laporan markdown
# ---------------------------------------------------------------------
lines = []
lines.append(f"# Laporan Performa — {bv} (klasifikasi 3 kelas)\n")
lines.append(f"Kelas: {', '.join(f'{i}={c}' for i, c in enumerate(CLASS_NAMES))}\n")
lines.append(f"Plan dievaluasi: {', '.join(plans_done)} | n_test = {int(summary['n_test'].iloc[0])}\n")
lines.append("\n## Ringkasan perbandingan (urut macro-F1)\n")
lines.append(summary_with_base.round(4).to_markdown())
lines.append("\n\n## Rincian per kelas\n")
lines.append(per_class_df.round(4).to_markdown(index=False))
lines.append("\n\n## Kesimpulan\n")
lines.append(f"- Plan terbaik: **{best_plan}** (macro-F1 {best_f1:.4f}, accuracy {best_acc:.4f}).\n")
lines.append(f"- Baseline majority-class: accuracy {base_acc:.4f}, macro-F1 {base_f1:.4f} "
             f"({baseline_row['keterangan']}).\n")
verdict = ("MENGALAHKAN" if best_f1 > base_f1 + 1e-9 else "TIDAK mengalahkan")
lines.append(f"- Plan terbaik **{verdict}** baseline pada macro-F1 "
             f"(selisih {best_f1 - base_f1:+.4f}). Macro-F1 lebih adil dari accuracy "
             f"karena kelas tidak seimbang.\n")
with open(f"{ANALYSIS_DIR}/report.md", "w") as f:
    f.write("\n".join(lines))

print(f"\n[done] analisis tersimpan di: {ANALYSIS_DIR}/")
print("  - summary_comparison.csv   (tabel metrik per plan + baseline)")
print("  - per_class_metrics.csv    (precision/recall/F1/support per kelas)")
print("  - report.md                (laporan ringkas siap-tempel)")
print("  - metrics_comparison.png   (bar metrik per plan)")
print("  - f1_per_class_heatmap.png (F1 per kelas per plan)")
print("  - confusion_norm_plan-*.png")
if histories:
    print("  - loss_curves.png")
print(f"\nPlan terbaik: {best_plan} | macro-F1={best_f1:.4f} | acc={best_acc:.4f} "
      f"| baseline acc={base_acc:.4f}")
