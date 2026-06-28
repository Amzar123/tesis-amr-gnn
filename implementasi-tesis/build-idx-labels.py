"""
Bangun label untuk dataset laporan tahunan IDX (Raw_English_2159).

Label = "return tahun berikutnya" (prediktif):
    dokumen KODE_N  ->  return saham sepanjang tahun (N+1)
                        = (close_akhir_(N+1) - close_akhir_N) / close_akhir_N

Output CSV: kolom 'ticker_and_date' (= nama dasar dokumen, mis. AALI_2021) dan
'next_year_return' (target regresi), plus kolom pendukung.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import yfinance as yf

RAW_DIR  = "/Users/ajizapar/Downloads/Raw_English_2159"
OUT_CSV  = "/Users/ajizapar/Documents/learning/amr-gnn/earnings-call-experiments/idx-annual-report-labels.csv"
BATCH    = 80   # jumlah ticker per panggilan download

# --- 1. Kumpulkan (ticker, tahun) dari nama file ---
docs = []
for fn in sorted(os.listdir(RAW_DIR)):
    if not fn.endswith(".txt"):
        continue
    base = fn[:-4]                 # AALI_2021
    if "_" not in base:
        continue
    ticker, year = base.rsplit("_", 1)
    if not year.isdigit():
        continue
    docs.append((base, ticker, int(year)))

tickers = sorted({t for _, t, _ in docs})
print(f"[info] {len(docs)} dokumen | {len(tickers)} ticker unik")

# --- 2. Download harga harian (auto-adjusted) per batch ---
yf_symbols = [t + ".JK" for t in tickers]
close = {}   # ticker -> Series(close) indexed by date

for i in range(0, len(yf_symbols), BATCH):
    chunk = yf_symbols[i:i+BATCH]
    print(f"[dl] batch {i//BATCH+1}: {len(chunk)} ticker ...", flush=True)
    data = yf.download(chunk, start="2021-01-01", end="2026-01-15",
                       progress=False, auto_adjust=True, group_by="ticker", threads=True)
    for sym in chunk:
        tk = sym[:-3]
        try:
            s = data[sym]["Close"].dropna() if len(chunk) > 1 else data["Close"].dropna()
        except Exception:
            continue
        if len(s) > 0:
            close[tk] = s
    time.sleep(1)  # jeda sopan ke server

print(f"[info] berhasil dapat harga untuk {len(close)}/{len(tickers)} ticker")

# --- 3. Hitung harga penutupan akhir tahun per ticker ---
def year_end_close(series, year):
    sub = series[series.index.year == year]
    return float(sub.iloc[-1]) if len(sub) else None

# --- 4. Bangun baris label ---
rows = []
n_missing_price, n_missing_year = 0, 0
for base, ticker, year in docs:
    if ticker not in close:
        n_missing_price += 1
        continue
    s = close[ticker]
    c_n  = year_end_close(s, year)
    c_n1 = year_end_close(s, year + 1)
    if c_n is None or c_n1 is None or c_n == 0:
        n_missing_year += 1
        continue
    ret = (c_n1 - c_n) / c_n
    rows.append({
        "ticker_and_date": base,
        "ticker": ticker,
        "report_year": year,
        "close_year": c_n,
        "close_next_year": c_n1,
        "next_year_return": ret,
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print("\n================ RINGKASAN ================")
print(f"dokumen berlabel       : {len(df)} / {len(docs)}")
print(f"drop (ticker tanpa harga): {n_missing_price}")
print(f"drop (tahun tak lengkap) : {n_missing_year}")
if len(df):
    print(f"next_year_return  mean={df['next_year_return'].mean():.4f}  std={df['next_year_return'].std():.4f}")
    print(f"  proporsi NAIK (>=0) : {(df['next_year_return']>=0).mean():.3f}")
    print(df.groupby('report_year')['next_year_return'].agg(['count','mean']))
print(f"\n[done] label disimpan ke: {OUT_CSV}")
