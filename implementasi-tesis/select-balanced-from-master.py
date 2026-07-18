"""
Pilih dokumen dari MASTER data (idx-yoy-trainvalid.csv / idx-yoy-test.csv,
2.107 kandidat 2021-2024, BUKAN cuma 37 yang graf .pt-nya sudah jadi) supaya
tiap kelas (rendah/netral/tinggi) punya jumlah yang SAMA rata di train maupun
di test.

Strategi: dokumen yang KEBETULAN sudah termasuk 37 yang graf-nya sudah ada
dipakai ulang dulu (supaya tidak generate ulang AMR+FinBERT yang mahal) --
baru kalau kuota kelas tsb belum penuh, ambil dokumen BARU dari master pool
(diacak dgn seed tetap = reproducible) untuk menutup kekurangannya.

Output:
    balanced-selection-trainvalid.csv  (target: TARGET_PER_CLASS x 3 kelas)
    balanced-selection-test.csv        (target: TARGET_PER_CLASS x 3 kelas)
    kolom tambahan `sudah_ada_graph` (True/False) menandai mana yang TIDAK
    perlu digenerate ulang vs mana yang PERLU AMR+FinBERT graph baru.
"""
import csv
import os
import random

TARGET_PER_CLASS = 10
SEED = 42

MASTER_TRAINVALID = "idx-yoy-trainvalid.csv"
MASTER_TEST = "idx-yoy-test.csv"
DONE_TRAINVALID = "idx-yoy-trainvalid-available.csv"
DONE_TEST = "idx-yoy-test-available.csv"


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_split(master_rows, done_rows, target_per_class, seed, split_name):
    done_keys = {r["ticker_and_date"] for r in done_rows}
    by_class = {}
    for r in master_rows:
        by_class.setdefault(r["yoy_close_category"], []).append(r)

    rng = random.Random(seed)
    selected = []
    report_lines = []
    for cls in ["rendah", "netral", "tinggi"]:
        pool = by_class.get(cls, [])
        already_done = [r for r in pool if r["ticker_and_date"] in done_keys]
        not_done = [r for r in pool if r["ticker_and_date"] not in done_keys]
        rng.shuffle(already_done)
        rng.shuffle(not_done)

        chosen = already_done[:target_per_class]
        n_new_needed = target_per_class - len(chosen)
        new_picks = not_done[:n_new_needed]
        chosen += new_picks

        for r in chosen:
            r = dict(r)
            r["sudah_ada_graph"] = r["ticker_and_date"] in done_keys
            selected.append(r)

        report_lines.append(
            f"  [{split_name}] {cls:<7}: pakai {len(chosen)-len(new_picks)} yg sudah ada + "
            f"{len(new_picks)} baru = {len(chosen)} (kandidat master tersedia: {len(pool)})"
        )
    return selected, report_lines


def write_csv(rows, path):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["yoy_close_category"], r["ticker_and_date"])))


def main():
    master_tv = load(MASTER_TRAINVALID)
    master_te = load(MASTER_TEST)
    done_tv = load(DONE_TRAINVALID)
    done_te = load(DONE_TEST)

    sel_tv, rep_tv = select_split(master_tv, done_tv, TARGET_PER_CLASS, SEED, "train/valid")
    sel_te, rep_te = select_split(master_te, done_te, TARGET_PER_CLASS, SEED + 1, "test")

    write_csv(sel_tv, "balanced-selection-trainvalid.csv")
    write_csv(sel_te, "balanced-selection-test.csv")

    print("=== train/valid (target {} x 3 kelas = {}) ===".format(TARGET_PER_CLASS, TARGET_PER_CLASS * 3))
    print("\n".join(rep_tv))
    print("=== test (target {} x 3 kelas = {}) ===".format(TARGET_PER_CLASS, TARGET_PER_CLASS * 3))
    print("\n".join(rep_te))

    n_new = sum(1 for r in sel_tv + sel_te if not r["sudah_ada_graph"])
    n_reused = sum(1 for r in sel_tv + sel_te if r["sudah_ada_graph"])
    print(f"\nTotal dokumen terpilih: {len(sel_tv)+len(sel_te)} "
          f"({n_reused} pakai graph yg sudah ada, {n_new} PERLU digenerate AMR+FinBERT baru)")


if __name__ == "__main__":
    main()
