"""
Analisis perbandingan 5 skema retrofit graf AMR (Plan A-E) untuk tesis
AMR-GNN.

Setiap "Book(Plan X).csv" adalah statistik graf (jumlah node & relasi) dari
strategi penghubungan doc_cls/snt_cls yang berbeda-beda di
`retrofit-labels-plan-*.py`:

  Plan A: doc_cls <-> root kalimat sebelumnya saja (baseline, paling jarang)
  Plan B: Plan A + rantai root(sblm) <-> root(skrg)
  Plan C: tambah node virtual snt_cls per kalimat, snt_cls <-> semua node
          kalimat sebelumnya + doc_cls, plus rantai snt_cls antar-kalimat
  Plan D: Plan C + doc_cls disambungkan langsung ke SETIAP node individual
          di kalimat sebelumnya (bukan cuma lewat snt_cls)
  Plan E: tanpa snt_cls, doc_cls langsung ke SETIAP node kalimat sebelumnya

Script ini menghasilkan (di ./analysis_output/):
  - summary_per_plan.csv     : statistik agregat per plan
  - per_file_growth.csv      : node/relasi tiap file utk semua plan + growth %
  - fig1_avg_nodes_relations.png
  - fig2_avg_density.png
  - fig3_nodes_vs_relations_scatter.png
  - fig4_density_boxplot.png
  - fig5_growth_vs_doc_size.png
dan mencetak insight ringkas ke stdout (untuk bahan slide presentasi).
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "analysis_output")
os.makedirs(OUT_DIR, exist_ok=True)

PLANS = ["A", "B", "C", "D", "E"]
PLAN_DESC = {
    "A": "doc_cls<->root (baseline)",
    "B": "A + rantai root antar-kalimat",
    "C": "+snt_cls, snt_cls<->semua node kalimat sblm",
    "D": "C + doc_cls<->tiap node kalimat sblm",
    "E": "doc_cls<->tiap node kalimat sblm (tanpa snt_cls)",
}

# Palet kategorikal (validated, lihat dataviz skill) - urutan tetap A..E
PLAN_COLOR = {
    "A": "#2a78d6",  # blue
    "B": "#008300",  # green
    "C": "#e87ba4",  # magenta
    "D": "#eda100",  # yellow
    "E": "#1baf7a",  # aqua
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
})


def load_plan(letter):
    path = os.path.join(HERE, f"Book(Plan {letter}).csv")
    df = pd.read_csv(path)
    df["num_relations"] = pd.to_numeric(df["num_relations"], errors="coerce")
    df = df.dropna(subset=["num_relations"]).copy()
    df["num_relations"] = df["num_relations"].astype(int)
    m = df["filename"].str.extract(r"^([A-Z0-9]+)_(\d{4})\.pt$")
    df["ticker"] = m[0]
    df["year"] = m[1]
    df["plan"] = letter
    df["density"] = df["num_relations"] / df["num_nodes"]
    return df


def main():
    long_df = pd.concat([load_plan(p) for p in PLANS], ignore_index=True)

    # ---------- 1. Statistik agregat per plan ----------
    summary = (
        long_df.groupby("plan")
        .agg(
            n_files=("filename", "count"),
            avg_nodes=("num_nodes", "mean"),
            avg_relations=("num_relations", "mean"),
            median_relations=("num_relations", "median"),
            total_relations=("num_relations", "sum"),
            avg_density=("density", "mean"),
            std_density=("density", "std"),
        )
        .reindex(PLANS)
        .round(2)
    )
    summary["desc"] = [PLAN_DESC[p] for p in summary.index]
    summary.to_csv(os.path.join(OUT_DIR, "summary_per_plan.csv"))

    # ---------- 2. Wide format + growth vs Plan A ----------
    wide_nodes = long_df.pivot_table(index=["ticker", "year"], columns="plan", values="num_nodes")
    wide_nodes.columns = [f"nodes_{c}" for c in wide_nodes.columns]
    wide_rel = long_df.pivot_table(index=["ticker", "year"], columns="plan", values="num_relations")
    wide_rel.columns = [f"rel_{c}" for c in wide_rel.columns]
    wide = wide_nodes.join(wide_rel).reset_index()

    for p in PLANS:
        wide[f"node_growth_{p}_vs_A_pct"] = (wide[f"nodes_{p}"] - wide["nodes_A"]) / wide["nodes_A"] * 100
        wide[f"rel_growth_{p}_vs_A_pct"] = (wide[f"rel_{p}"] - wide["rel_A"]) / wide["rel_A"] * 100

    wide.to_csv(os.path.join(OUT_DIR, "per_file_growth.csv"), index=False)

    # ---------- 3. Korelasi ukuran dokumen (Plan A) vs growth tiap plan ----------
    corr = {}
    for p in ["B", "C", "D", "E"]:
        corr[p] = np.corrcoef(wide["nodes_A"], wide[f"rel_growth_{p}_vs_A_pct"])[0, 1]

    # ---------- Fig 1: rata-rata nodes & relations per plan (2 subplot, 1 sumbu tiap panel) ----------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, col, title in zip(
        axes, ["avg_nodes", "avg_relations"], ["Rata-rata jumlah node", "Rata-rata jumlah relasi"]
    ):
        vals = summary[col]
        bars = ax.bar(PLANS, vals, color=[PLAN_COLOR[p] for p in PLANS], width=0.62)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Plan retrofit")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom",
                     fontsize=9, color=INK_SECONDARY)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Ukuran graf rata-rata per skema retrofit (n=60 laporan)", fontsize=12, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_avg_nodes_relations.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------- Fig 2: rata-rata density (relasi/node) ----------
    fig, ax = plt.subplots(figsize=(6, 4.2))
    vals = summary["avg_density"]
    bars = ax.bar(PLANS, vals, color=[PLAN_COLOR[p] for p in PLANS], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("Relasi per node (rata-rata)")
    ax.set_xlabel("Plan retrofit")
    ax.set_title("Kepadatan graf (num_relations / num_nodes) per plan", loc="left", color=INK)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_avg_density.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------- Fig 3: scatter num_nodes (Plan A) vs num_relations, per plan ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    for p in PLANS:
        sub = long_df[long_df["plan"] == p].sort_values("num_nodes")
        ax.scatter(sub["num_nodes"], sub["num_relations"], s=16, color=PLAN_COLOR[p],
                   label=f"Plan {p}", alpha=0.85, edgecolors="none")
    ax.set_xlabel("Jumlah node graf (skala plan masing-masing)")
    ax.set_ylabel("Jumlah relasi")
    ax.set_title("Node vs relasi per laporan, seluruh plan", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_nodes_vs_relations_scatter.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------- Fig 4: boxplot distribusi density per plan ----------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [long_df[long_df["plan"] == p]["density"].values for p in PLANS]
    bp = ax.boxplot(data, tick_labels=PLANS, patch_artist=True, widths=0.5,
                     medianprops=dict(color=INK, linewidth=1.5),
                     whiskerprops=dict(color=INK_MUTED), capprops=dict(color=INK_MUTED),
                     flierprops=dict(marker="o", markersize=3, markerfacecolor=INK_MUTED,
                                      markeredgecolor="none"))
    for patch, p in zip(bp["boxes"], PLANS):
        patch.set_facecolor(PLAN_COLOR[p])
        patch.set_alpha(0.55)
        patch.set_edgecolor(PLAN_COLOR[p])
    ax.set_ylabel("Relasi per node")
    ax.set_xlabel("Plan retrofit")
    ax.set_title("Sebaran kepadatan graf antar-63 laporan, per plan", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_density_boxplot.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------- Fig 5: growth relasi (%) vs ukuran dokumen (Plan A nodes) ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    for p in ["B", "C", "D", "E"]:
        ax.scatter(wide["nodes_A"], wide[f"rel_growth_{p}_vs_A_pct"], s=16,
                   color=PLAN_COLOR[p], label=f"Plan {p} (r={corr[p]:.2f})", alpha=0.85, edgecolors="none")
    ax.set_xlabel("Ukuran dokumen (jumlah node, Plan A)")
    ax.set_ylabel("Pertambahan relasi vs Plan A (%)")
    ax.set_title("Apakah retrofit membebani dokumen besar lebih berat?", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_growth_vs_doc_size.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ================= INSIGHT (dicetak utk bahan slide) =================
    print("=" * 78)
    print("RINGKASAN STATISTIK PER PLAN")
    print("=" * 78)
    print(summary.to_string())

    print("\n" + "=" * 78)
    print("INSIGHT UTAMA")
    print("=" * 78)

    base_rel = summary.loc["A", "avg_relations"]
    base_nodes = summary.loc["A", "avg_nodes"]
    for p in PLANS:
        rel_up = (summary.loc[p, "avg_relations"] - base_rel) / base_rel * 100
        node_up = (summary.loc[p, "avg_nodes"] - base_nodes) / base_nodes * 100
        print(f"- Plan {p}: rata2 {summary.loc[p,'avg_nodes']:,.0f} node "
              f"({node_up:+.1f}% vs A), {summary.loc[p,'avg_relations']:,.0f} relasi "
              f"({rel_up:+.1f}% vs A), density {summary.loc[p,'avg_density']:.2f} relasi/node.")

    print("\n- Korelasi ukuran dokumen (jumlah node Plan A) vs %pertambahan relasi tiap plan:")
    for p in ["B", "C", "D", "E"]:
        print(f"    Plan {p}: r = {corr[p]:.3f}")

    densest = summary["avg_density"].idxmax()
    sparsest = summary["avg_density"].idxmin()
    print(f"\n- Plan paling padat: {densest} ({summary.loc[densest,'avg_density']:.2f} relasi/node); "
          f"paling jarang: {sparsest} ({summary.loc[sparsest,'avg_density']:.2f} relasi/node).")

    # Plan C vs D vs E: C&D nambah node virtual, E tidak tapi paling padat setelah D
    print(f"\n- Plan C & D menambah node virtual 'snt_cls' (+{(summary.loc['C','avg_nodes']/base_nodes-1)*100:.1f}% node vs A),"
          f" sedangkan E mencapai densitas relasi yang mendekati C/D TANPA menambah node sama sekali"
          f" -> paling efisien kalau target-nya kaya-relasi tapi hemat memori node.")

    top5 = long_df[long_df["plan"] == "A"].nlargest(5, "num_nodes")[["filename", "num_nodes"]]
    print("\n- 5 laporan dgn graf terbesar (baseline Plan A):")
    for _, r in top5.iterrows():
        print(f"    {r['filename']}: {r['num_nodes']:,} node")

    print(f"\nOutput tersimpan di: {OUT_DIR}")


if __name__ == "__main__":
    main()
