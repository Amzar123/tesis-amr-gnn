"""
Re-split 37 dokumen (klasifikasi YoY: rendah/netral/tinggi) secara STRATIFIED
per kelas, supaya komposisi kelas di train/valid dan test SAMA (proporsional)
-- bukan lagi murni per-tahun (2021-2023 vs 2024) seperti split lama, yang
komposisi kelasnya jomplang (lihat masukan dosen #2).

Tidak butuh sklearn/pandas -> murni Python stdlib, supaya bisa dicek siapa
saja saja (ticker_and_date) yang masuk train vs test sebelum dipakai training.

Cara pakai:
    python split-stratified-yoy.py
    (baca idx-yoy-trainvalid-available.csv + idx-yoy-test-available.csv,
     gabung jadi 37 baris, lalu tulis ulang jadi:
       idx-yoy-trainvalid-stratified.csv
       idx-yoy-test-stratified.csv)

Catatan penting (baca sebelum dipakai): split ini TIDAK LAGI temporal holdout.
Beberapa dokumen 2024 dipindah ke train dan beberapa dokumen 2021-2023
dipindah ke test supaya proporsi rendah/netral/tinggi merata. Untuk skenario
"prediksi tahun depan dari data tahun lalu" yang realistis, split temporal
lama tetap relevan -- gunakan split stratified ini sebagai eksperimen kedua
untuk menunjukkan performa model tidak semata-mata efek pergeseran distribusi
kelas antar tahun, BUKAN pengganti mutlak split temporal.
"""
import csv
import random

TRAINVALID_IN = "idx-yoy-trainvalid-available.csv"
TEST_IN = "idx-yoy-test-available.csv"
TRAINVALID_OUT = "idx-yoy-trainvalid-stratified.csv"
TEST_OUT = "idx-yoy-test-stratified.csv"
TEST_FRACTION = 10 / 37  # pertahankan ukuran test set spt semula (~27%)
SEED = 42


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    rows = read_rows(TRAINVALID_IN) + read_rows(TEST_IN)
    fieldnames = list(rows[0].keys())

    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    rng = random.Random(SEED)
    train_rows, test_rows = [], []
    for label, group in sorted(by_label.items()):
        group = group[:]
        rng.shuffle(group)
        n_test = round(len(group) * TEST_FRACTION)
        test_rows += group[:n_test]
        train_rows += group[n_test:]

    with open(TRAINVALID_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sorted(train_rows, key=lambda r: r["ticker_and_date"]))

    with open(TEST_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sorted(test_rows, key=lambda r: r["ticker_and_date"]))

    def counts(rs):
        c = {"0": 0, "1": 0, "2": 0}
        for r in rs:
            c[r["label"]] += 1
        return c

    print(f"train/valid -> {TRAINVALID_OUT} ({len(train_rows)} dok): {counts(train_rows)}")
    print(f"test        -> {TEST_OUT} ({len(test_rows)} dok): {counts(test_rows)}")


if __name__ == "__main__":
    main()
