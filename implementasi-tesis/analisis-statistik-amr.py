"""
Statistik deskriptif per dokumen AMR + kalkulator density Plan A-E.

Menjawab masukan dosen #1 (statistik data: jumlah kalimat/node/relasi per
dokumen) dan #4 (cara hitung density). Topologi Plan A-E di sini adalah
replikasi murni dari edge-wiring pada
earnings-call-experiments/amrs-to-graphs/amrs-to-graphs-plan-{A..E}-hk-finbert-new*.py,
tapi tanpa dependensi torch/amrlib/FinBERT -> bisa jalan di laptop biasa,
cukup baca file .amr (tidak perlu regenerate embedding).

Cara pakai:
    python analisis-statistik-amr.py path/ke/folder-amr/ -o statistik_38_dokumen.csv
    python analisis-statistik-amr.py path/ke/satu_file.amr

Density dihitung sebagai graf tak berarah sederhana:
    density = 2E / (N * (N-1))
di mana N = jumlah node, E = jumlah pasangan edge unik (u,v) u!=v setelah
graf disimetriskan (analog to_undirected() di PyG, minus duplikat).
"""
import argparse
import csv
import os
import statistics


def parse_amr(path):
    """Parse satu file .amr (format JAMR: # ::tok / # ::node / # ::edge / # ::root)
    menjadi list per-kalimat: {tokens, nodes(var->word), edges(list of (u,v)), root}."""
    sentences = []
    cur = None
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("# ::tok"):
            if cur is not None:
                sentences.append(cur)
            tok_str = line[8:].rstrip("\n")
            tokens = [t for t in tok_str.split(" ") if t != ""]
            cur = {"tokens": tokens, "nodes": {}, "edges": [], "root": None}
        elif line.startswith("# ::node"):
            parts = line[8:].split()
            cur["nodes"][parts[0]] = parts[1]
        elif line.startswith("# ::edge"):
            parts = line[8:].split()
            cur["edges"].append((parts[-2], parts[-1]))
        elif line.startswith("# ::root"):
            cur["root"] = line[8:].split()[0]
    if cur is not None:
        sentences.append(cur)
    return sentences


def build_plan_graph(sentences, plan):
    """Bangun (jumlah_node, list_edge_terarah) global untuk satu Plan (A/B/C/D/E),
    mengikuti persis logika wiring di amrs-to-graphs-plan-{plan}-hk-finbert-new*.py."""
    next_id = 0

    def new_node():
        nonlocal next_id
        i = next_id
        next_id += 1
        return i

    doc_cls = new_node()  # node 0, selalu ada di semua plan
    edges = []
    prev_root_id = None
    prev_snt_cls_id = None

    for s in sentences:
        local_ids = {var: new_node() for var in s["nodes"]}
        snt_cls_id = new_node() if plan in ("C", "D") else None

        for (k1, k2) in s["edges"]:
            edges.append((local_ids[k1], local_ids[k2]))

        root_id = local_ids.get(s["root"])

        if plan == "A":
            edges.append((doc_cls, root_id))
        elif plan == "B":
            edges.append((doc_cls, root_id))
            if prev_root_id is not None:
                edges.append((prev_root_id, root_id))
            prev_root_id = root_id
        elif plan == "C":
            for v in local_ids.values():
                edges.append((snt_cls_id, v))
            edges.append((doc_cls, snt_cls_id))
            if prev_snt_cls_id is not None:
                edges.append((prev_snt_cls_id, snt_cls_id))
            prev_snt_cls_id = snt_cls_id
        elif plan == "D":
            for v in local_ids.values():
                edges.append((snt_cls_id, v))
                edges.append((doc_cls, v))
            edges.append((doc_cls, snt_cls_id))
            if prev_snt_cls_id is not None:
                edges.append((prev_snt_cls_id, snt_cls_id))
            prev_snt_cls_id = snt_cls_id
        elif plan == "E":
            for v in local_ids.values():
                edges.append((doc_cls, v))

    return next_id, edges


def density_of(num_nodes, directed_edges):
    undirected = {(min(u, v), max(u, v)) for (u, v) in directed_edges if u != v}
    e = len(undirected)
    n = num_nodes
    d = (2 * e) / (n * (n - 1)) if n > 1 else 0.0
    return n, e, d


def analyze_file(path):
    sentences = parse_amr(path)
    n_sent = len(sentences)
    n_nodes = sum(len(s["nodes"]) for s in sentences)
    n_edges = sum(len(s["edges"]) for s in sentences)
    n_tokens = sum(len(s["tokens"]) for s in sentences)

    row = {
        "doc": os.path.splitext(os.path.basename(path))[0],
        "n_sentences": n_sent,
        "n_amr_nodes": n_nodes,
        "n_amr_relations": n_edges,
        "n_tokens": n_tokens,
        "avg_nodes_per_sentence": round(n_nodes / n_sent, 2) if n_sent else 0,
        "avg_relations_per_sentence": round(n_edges / n_sent, 2) if n_sent else 0,
    }
    for plan in ["A", "B", "C", "D", "E"]:
        num_nodes, directed_edges = build_plan_graph(sentences, plan)
        n, e, d = density_of(num_nodes, directed_edges)
        row[f"plan_{plan}_nodes"] = n
        row[f"plan_{plan}_edges"] = e
        row[f"plan_{plan}_density"] = round(d, 8)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="file .amr tunggal, atau folder berisi banyak file .amr (mis. 38 dokumen)")
    ap.add_argument("-o", "--out", default="statistik_amr.csv", help="path CSV output (default: statistik_amr.csv)")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        files = sorted(f for f in os.listdir(args.path) if f.endswith(".amr"))
        paths = [os.path.join(args.path, f) for f in files]
    else:
        paths = [args.path]

    rows = [analyze_file(p) for p in paths]

    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] {len(rows)} dokumen dianalisis -> {args.out}")

    for col in ["n_sentences", "n_amr_nodes", "n_amr_relations"]:
        vals = [r[col] for r in rows]
        print(f"{col:22s} min={min(vals):<8} max={max(vals):<8} mean={statistics.mean(vals):<10.1f} "
              f"stdev={statistics.pstdev(vals):.1f}" if len(vals) > 1 else f"{col}: {vals[0]}")


if __name__ == "__main__":
    main()
