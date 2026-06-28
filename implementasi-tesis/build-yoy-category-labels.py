"""
Bangun label KLASIFIKASI untuk dataset laporan tahunan IDX dari file kategori
YoY (tabular): `mda_auto_from_2016_not_adj_merge_fold.csv`.

Target = YoY_Close_Category (3 kelas) -> dipakai APA ADANYA dari file mda.
    rendah = 0
    netral = 1
    tinggi = 2

Key dokumen `ticker_and_date` dibentuk dari kolom Ticker (buang sufiks .JK) + "_" +
Year supaya cocok dengan nama file graph .pt (mis. ACES.JK + 2018 -> ACES_2018).

Split TEMPORAL (bukan fold 99, karena fold 99 meng-hold-out ticker yang
mayoritas tak punya dokumen teks IDX):
    report_year <= 2023  -> TRAIN/VALID (pemisahan train vs valid di script training)
    report_year == 2024  -> TEST
Baris dengan report_year < MIN_YEAR dibuang (dokumen teks IDX hanya 2021-2024,
jadi tahun lebih lama mustahil punya graph .pt). Kolom Ticker_Fold tetap
disimpan untuk referensi.

Opsional: kalau DOC_INVENTORY_CSV ada, baris difilter ke dokumen yang BENAR-BENAR
ada (key ticker_and_date ada di inventaris) supaya hitungan train/test jujur.

Output:
    implementasi-tesis/idx-yoy-trainvalid.csv
    implementasi-tesis/idx-yoy-test.csv
Kolom: ticker_and_date, ticker, report_year, ticker_fold,
       yoy_close_category, label
"""
import os
import pandas as pd

# --- Konfigurasi path ---
MDA_CSV = os.environ.get(
    "MDA_CSV",
    "/Users/ajizapar/Downloads/mda_auto_from_2016_not_adj_merge_fold.csv",
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_TRAINVALID = os.path.join(OUT_DIR, "idx-yoy-trainvalid.csv")
OUT_TEST = os.path.join(OUT_DIR, "idx-yoy-test.csv")

# Daftar dokumen yang .pt-nya SUDAH ter-generate (untuk sanity-run). Union key
# dari file-file ini = subset yang punya .pt. Kalau ada, script juga menulis
# versi "-available" (idx-yoy-*-available.csv) yang sudah difilter ke subset itu.
AVAILABLE_PT_CSVS = [
    os.path.join(OUT_DIR, "idx-trainvalid-available.csv"),
    os.path.join(OUT_DIR, "idx-test-available.csv"),
]
OUT_TRAINVALID_AVAIL = os.path.join(OUT_DIR, "idx-yoy-trainvalid-available.csv")
OUT_TEST_AVAIL = os.path.join(OUT_DIR, "idx-yoy-test-available.csv")

# Inventaris dokumen IDX yang benar-benar ada (kolom 'ticker_and_date').
# Set ke "" untuk menonaktifkan filter inventaris.
DOC_INVENTORY_CSV = os.environ.get(
    "DOC_INVENTORY_CSV", os.path.join(OUT_DIR, "idx-annual-report-labels.csv")
)

# Split temporal
MIN_YEAR  = 2021   # buang dokumen lebih lama dari ini (tak ada teksnya)
TEST_YEAR = 2024   # tahun yang dijadikan test set

# Map kategori -> integer (urut rendah->tinggi)
CATEGORY_TO_LABEL = {"rendah": 0, "netral": 1, "tinggi": 2}

# --- 1. Baca file kategori ---
df = pd.read_csv(MDA_CSV)
print(f"[info] baca {MDA_CSV}")
print(f"[info] {len(df)} baris | kolom: {list(df.columns)}")

required = {"Ticker", "Year", "YoY_Close_Category", "Ticker_Fold"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Kolom wajib hilang: {missing}")

# --- 2. Bersihkan & bentuk kolom ---
df = df.dropna(subset=["Ticker", "Year", "YoY_Close_Category", "Ticker_Fold"]).copy()
df["YoY_Close_Category"] = df["YoY_Close_Category"].str.strip().str.lower()

unknown = set(df["YoY_Close_Category"].unique()) - set(CATEGORY_TO_LABEL)
if unknown:
    raise ValueError(f"Kategori tak dikenal (di luar {list(CATEGORY_TO_LABEL)}): {unknown}")

df["label"] = df["YoY_Close_Category"].map(CATEGORY_TO_LABEL).astype(int)
df["report_year"] = df["Year"].astype(int)
df["ticker_fold"] = df["Ticker_Fold"].astype(int)
# Buang sufiks .JK (atau bursa lain) untuk mencocokkan nama file .pt
df["ticker"] = df["Ticker"].str.replace(r"\.[A-Za-z]+$", "", regex=True)
df["ticker_and_date"] = df["ticker"] + "_" + df["report_year"].astype(str)

out_cols = [
    "ticker_and_date", "ticker", "report_year", "ticker_fold",
    "yoy_close_category", "label",
]
df = df.rename(columns={"YoY_Close_Category": "yoy_close_category"})[out_cols]

# Cek duplikat key (1 dokumen = 1 label)
dups = df["ticker_and_date"].duplicated().sum()
if dups:
    print(f"[warn] {dups} key duplikat ditemukan -> ambil baris pertama")
    df = df.drop_duplicates(subset=["ticker_and_date"], keep="first").reset_index(drop=True)

# --- 3a. Buang tahun yang mustahil punya dokumen ---
_before = len(df)
df = df[df["report_year"] >= MIN_YEAR].reset_index(drop=True)
print(f"[filter] report_year >= {MIN_YEAR}: {len(df)}/{_before} baris (sisanya dibuang, tak ada teks)")

# --- 3b. (opsional) Filter ke dokumen yang benar-benar ada ---
if DOC_INVENTORY_CSV and os.path.isfile(DOC_INVENTORY_CSV):
    inv = pd.read_csv(DOC_INVENTORY_CSV)
    have = set(inv["ticker_and_date"].astype(str))
    _before = len(df)
    df = df[df["ticker_and_date"].isin(have)].reset_index(drop=True)
    print(f"[filter] ada di inventaris dokumen ({DOC_INVENTORY_CSV}): {len(df)}/{_before} baris")
else:
    print(f"[filter] inventaris dokumen tidak dipakai (DOC_INVENTORY_CSV kosong/absen)")

# --- 4. Split temporal ---
df_test = df[df["report_year"] == TEST_YEAR].reset_index(drop=True)
df_trainvalid = df[df["report_year"] < TEST_YEAR].reset_index(drop=True)

df_trainvalid.to_csv(OUT_TRAINVALID, index=False)
df_test.to_csv(OUT_TEST, index=False)

# --- 5. Ringkasan ---
def dist(d):
    return d["yoy_close_category"].value_counts().to_dict()

print("\n================ RINGKASAN ================")
print(f"trainvalid (year {MIN_YEAR}..{TEST_YEAR-1}) : {len(df_trainvalid)} baris -> {OUT_TRAINVALID}")
print(f"  distribusi kelas : {dist(df_trainvalid)}")
print(f"test       (year {TEST_YEAR})       : {len(df_test)} baris -> {OUT_TEST}")
print(f"  distribusi kelas : {dist(df_test)}")

# --- 6. (opsional) Versi -available: hanya dokumen yang .pt-nya sudah ada ---
avail_keys = set()
for p in AVAILABLE_PT_CSVS:
    if os.path.isfile(p):
        avail_keys |= set(pd.read_csv(p)["ticker_and_date"].astype(str))
if avail_keys:
    df_tv_av = df_trainvalid[df_trainvalid["ticker_and_date"].isin(avail_keys)].reset_index(drop=True)
    df_te_av = df_test[df_test["ticker_and_date"].isin(avail_keys)].reset_index(drop=True)
    df_tv_av.to_csv(OUT_TRAINVALID_AVAIL, index=False)
    df_te_av.to_csv(OUT_TEST_AVAIL, index=False)
    print(f"\n[available] .pt sudah ada: {len(avail_keys)} key")
    print(f"trainvalid-available : {len(df_tv_av)} baris -> {OUT_TRAINVALID_AVAIL}")
    print(f"  distribusi kelas : {dist(df_tv_av)}")
    print(f"test-available       : {len(df_te_av)} baris -> {OUT_TEST_AVAIL}")
    print(f"  distribusi kelas : {dist(df_te_av)}")
else:
    print("\n[available] tak ada daftar .pt -> versi -available dilewati")

print(f"\nmap label: {CATEGORY_TO_LABEL}")
print("[done]")
