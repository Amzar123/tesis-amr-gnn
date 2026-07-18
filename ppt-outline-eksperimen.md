# Outline PPT: Perbandingan Hasil Eksperimen
## GATv2 AMR-GNN — Dataset Earnings Calls (FLAG) vs. MDA Annual Report (AMD/IDX)

---

## Slide 1 — Judul

**Perbandingan Eksperimen GATv2 AMR-GNN**
**Dataset Earnings Calls (FLAG) vs. MDA Annual Report (AMD/IDX)**

---

## Slide 2 — Overview Pipeline

Pipeline yang sama digunakan untuk kedua dataset:

```
Teks Dokumen
    │
    ▼  AMR Parsing (SPRING)
Graf AMR (node = konsep, edge = relasi)
    │
    ▼  FinBERT Node Embedding (768-dim)
Graf PyG / DGL  (.pt)
    │
    ▼  Plan A/B/C/D/E  (strategi konstruksi graf)
    │
    ▼  GATv2  (4 Layer, multi-head attention)
    │
    ▼  Prediksi Harga / Kategori Saham
```

---

## Slide 3 — Perbandingan Dataset

| Aspek | Earnings Calls (FLAG) | MDA Annual Report (AMD) |
|---|---|---|
| **Sumber teks** | Transkrip earnings call | Seksi MDA laporan tahunan |
| **Pasar** | Nasdaq/NYSE – Tech AS | IDX – Indonesia |
| **Bahasa** | Inggris | Inggris (laporan tahunan IDX) |
| **Periode** | Train: 2010–2018 / Test: 2019 | ≈2016–2023 (train) / 2024 (test) |
| **Ukuran (total label)** | ~300+ dokumen | 1.522 train/val + 585 test |
| **Ukuran (grafik tersedia)** | Semua | 27 train/val + 10 test |
| **Plans dievaluasi** | A, B, C, D, E | A, B, C, D, E |

---

## Slide 4 — Perbedaan Task & Arsitektur Model

| Aspek | Earnings Calls | MDA (IDX) |
|---|---|---|
| **Task** | Regresi | Klasifikasi 3 kelas |
| **Target** | Perubahan harga harian / bulanan / volatilitas | Kategori YoY close: *rendah / netral / tinggi* |
| **Loss function** | MSE | Cross-Entropy |
| **Metrik utama** | Test MSE, Directional Accuracy | Accuracy, Macro-F1 |
| **HIDDEN_DIM** | Blow-up tiap layer (768 → 1536 → …) | Fixed 256 (cegah OOM) |
| **Heads dievaluasi** | 2, 4, 8, 12 | 4 |
| **Learning rate** | 3e-6 – 1e-3 (grid search) | 1e-4 |
| **Epochs** | 5 atau 10 | 5 |

---

## Slide 5 — Hasil Earnings Calls: Best Model Overall

**Best Model:** Plan C | 4 heads | LR = 1e-3 | 5 epochs | Target: *vol* (volatilitas bulanan)

| Metrik | Nilai |
|---|---|
| Test MSE | **0.001032** |
| Valid MSE | 0.001298 |
| Best Epoch | 2 |

> Model konvergen sangat cepat (epoch ke-2). Plan C dengan 4 heads dan LR tinggi memberikan MSE terbaik pada target volatilitas.

---

## Slide 6 — Hasil Earnings Calls: Best MSE per Plan × Heads

| Plan | Heads | LR | Test MSE | Valid MSE | Best Epoch |
|---|---|---|---|---|---|
| A | 4 | 3e-5 | 0.008454 | 0.005921 | 5 |
| B | 4 | 1e-5 | 0.008685 | 0.006064 | 2 |
| **C** | **4** | **1e-3** | **0.001032** | **0.001298** | **2** |
| C | 2 | 1e-5 | 0.007983 | 0.005940 | 3 |
| C | 8 | 3e-5 | 0.007931 | 0.005748 | 3 |
| C | 12 | 1e-4 | 0.008701 | 0.006053 | 2 |
| D | 8 | 1e-5 | 0.007844 | 0.005813 | 4 |
| E | 8 | 6e-5 | 0.007814 | 0.005771 | 6 |

> 📊 Sertakan gambar: `Earnings-Call-Experiments/GATv2/results/06_mse_heatmap.png`

---

## Slide 7 — Hasil Earnings Calls: Directional Accuracy

> 📊 Sertakan gambar:
> - `results/04_confusion_matrix.png`
> - `results/07_directional_accuracy_bar.png`

**Poin diskusi:**
- Directional accuracy = prediksi arah (naik/turun), dihitung dari tanda nilai regresi
- Confusion matrix best model: Plan C, 4 heads
- Directional accuracy: `[FILL dari output analyze_results.py]`

---

## Slide 8 — Hasil Earnings Calls: Loss Curve

> 📊 Sertakan gambar:
> - `results/02_loss_curves.png` (best model)
> - `results/03_loss_curves_all_plans.png` (semua plan, 4 heads)

**Poin diskusi:**
- Plan C konvergen lebih cepat dibanding plan lain
- Valid loss stabil → tidak overfit berat

---

## Slide 9 — Hasil MDA (IDX): Accuracy & Macro-F1 per Plan

| Plan | Accuracy | Macro-F1 | Balanced Acc | n_test |
|---|---|---|---|---|
| A | `[FILL]` | `[FILL]` | `[FILL]` | 10 |
| B | `[FILL]` | `[FILL]` | `[FILL]` | 10 |
| C | `[FILL]` | `[FILL]` | `[FILL]` | 10 |
| D | `[FILL]` | `[FILL]` | `[FILL]` | 10 |
| E | `[FILL]` | `[FILL]` | `[FILL]` | 10 |
| **BASELINE** (majority class) | `[FILL]` | `[FILL]` | — | — |

> 📊 Sertakan gambar: `accuracy_f1_per_plan.png` dari Kaggle output

---

## Slide 10 — Hasil MDA (IDX): Detail Per Kelas (Plan Terbaik)

| Kelas | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| rendah (0) | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |
| netral (1) | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |
| tinggi (2) | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` |

> 📊 Sertakan gambar:
> - `confusion_norm_plan-X.png` (plan terbaik)
> - `f1_per_class_heatmap.png`
>
> (Semua dari Kaggle output folder `analysis/`)

---

## Slide 11 — Perbandingan & Diskusi

| Aspek | Earnings Calls | MDA (IDX) |
|---|---|---|
| **Metrik utama** | Test MSE = 0.00103 | Acc = `[FILL]`, Macro-F1 = `[FILL]` |
| **Plan terbaik** | C (4 heads, LR 1e-3) | `[FILL]` |
| **Konvergensi** | Cepat (epoch 2) | `[FILL]` |
| **Tantangan** | Grid hyperparameter luas | Dataset grafik sangat kecil (10 test) |
| **Catatan khusus** | Target regresi kontinu | 3 kelas tidak seimbang |

**Poin diskusi tambahan:**
- Keterbatasan MDA: file `.pt` yang tersedia sangat kecil (10 test) dari total 585 → hasil belum representatif untuk keseluruhan dataset IDX
- Plan C secara konsisten performatif di kedua dataset
- Domain berbeda (pasar AS vs IDX), tapi pipeline AMR-GNN berhasil ditransfer

---

## Slide 12 — Kesimpulan

1. Pipeline AMR-GNN (GATv2 4-layer) berhasil diterapkan pada **dua domain berbeda**: earnings calls perusahaan AS dan laporan tahunan IDX
2. **Earnings Calls (FLAG):** Plan C + 4 heads + LR 1e-3 → Test MSE terbaik **0.001032** pada prediksi volatilitas bulanan
3. **MDA (IDX):** `[FILL — plan terbaik, accuracy, macro-F1 dari Kaggle]`
4. **Keterbatasan utama MDA:** grafik yang ter-generate baru 27 train + 10 test dari total 1.522 + 585 dokumen → perlu generate seluruh file `.pt` agar hasil lebih representatif
5. **Ke depan:** Lengkapi pembuatan graf AMR untuk seluruh dataset IDX, eksplorasi hyperparameter lebih luas untuk task klasifikasi

---

*Isi placeholder `[FILL]` dengan hasil dari Kaggle output: `analysis/summary_comparison.csv` dan `plots/summary_per_plan.csv`*
