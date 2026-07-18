"""
BONUS: nambahin label + jumlah relasi PERSIS ke file .pt PLAN C yang SUDAH
ke-generate, TANPA perlu re-run BERT/GPU sama sekali.

Kenapa ini butuh dibuat terpisah?
- File .pt kamu udah full ke-generate semua (tanpa label & tanpa angka
  relasi pasti), dan generate ulang lewat script utama bakal di-skip semua
  karena mekanisme resume-nya (aman, tapi jadinya label nggak nambah).
- Regenerasi ulang via BERT itu mahal (butuh GPU lagi). Padahal label
  (kata & entity tiap node) dan jumlah relasi PERSIS sebenarnya bisa
  didapat cuma dari file .amr aslinya, TANPA butuh embedding sama sekali.
  Jadi script ini cuma parsing teks (CPU, cepat), lalu nempelin
  word/entity/label/num_relations ke file .pt yang udah ada -- embedding
  yang udah dihitung sebelumnya (g.x) TIDAK disentuh/dihitung ulang.

Logika parsing node/edge di bawah ini adalah REPLIKA PERSIS dari
amrs-to-graphs-plan-C-hk-finbert-new-amd.py (bagian pembentukan graf, minus
pemanggilan BERT): setiap kalimat dapat 1 node ekstra "snt_cls", yang
disambungkan ke SEMUA node di kalimat SEBELUMNYA + ke doc_cls (index 0),
plus rantai snt_cls(sblm) <-> snt_cls(skrg). Plan C TIDAK memakai baris
::root sama sekali. JANGAN diubah agar jumlah relasi & jumlah node yang
dihasilkan tetap cocok dengan file .pt yang sudah ada.

CARA PAKAI
1. Sesuaikan sec_fname, SRC_DIR, dan DST_DIR di bawah.
   - SRC_DIR = folder yang BERISI .pt lama (boleh di /kaggle/input/..., yang
     read-only, kalau kamu attach hasil run sebelumnya sebagai dataset).
   - DST_DIR = folder TUJUAN buat nulis .pt yang sudah dikasih label. HARUS
     di /kaggle/working (pakai path relatif kayak default di bawah), karena
     /kaggle/input itu read-only dan nggak bisa ditulisi -- ini penyebab
     error "Read-only file system" kalau SRC_DIR dan DST_DIR sama-sama
     nunjuk ke /kaggle/input.
2. Jalankan (nggak butuh GPU, jalan di CPU aja).
3. Script cek tiap file di SRC_DIR: kalau file dgn nama sama SUDAH ada di
   DST_DIR, di-skip (resumable, aman di-run ulang kalau kena timeout).
   Kalau belum, file di-load dari SRC_DIR (read-only, aman), dicek jumlah
   node hasil parsing ulang .amr cocok dgn jumlah node di .pt (validasi
   keamanan), label & num_relations ditempel, lalu ditulis ke DST_DIR
   (bukan menimpa SRC_DIR).
4. Di akhir, CSV graph_stats_plan_C.csv ikut dibuat/diupdate (filename,
   num_nodes, num_relations, has_labels).
"""

import torch
from torch_geometric.data import Data
import re
import os
import csv
import traceback

sec_fname = "/kaggle/input/datasets/amzar11/idx-amrs-v4/idx-amrs"

# Baca dari sini (boleh read-only, misal dataset hasil run sebelumnya):
SRC_DIR = "/kaggle/input/datasets/amzar11/graf-plan-c/truly-all-results-graphs-hk-finbert-plan-C"

# Tulis hasil yang sudah dikasih label ke sini (WAJIB writable, di /kaggle/working):
DST_DIR = 'truly-all-results-graphs-hk-finbert-plan-C'
os.makedirs(DST_DIR, exist_ok=True)


def parse_amr_labels_and_relations(fname):
    """
    Parsing ulang 1 file .amr, PERSIS mengikuti urutan & logika construct
    graph di amrs-to-graphs-plan-C-hk-finbert-new-amd.py: tiap kalimat
    dikasih 1 node "snt_cls" ekstra yang disambungkan ke semua node di
    kalimat SEBELUMNYA + ke doc_cls, plus rantai snt_cls(sblm) <->
    snt_cls(skrg). Tanpa hitung embedding sama sekali. Return (words,
    entities, num_relations). Urutan words/entities sejajar dengan urutan
    node_index waktu graph dibuat aslinya (index 0 = doc_cls).
    """
    with open(fname, 'r') as ffile:
        flines = ffile.readlines()

    current_snt = ''
    current_snt_cls_doc_idx = None
    previous_snt_cls_doc_idx = None
    current_snt_nodes_dict = {}  # snt_node_id -> node_index, utk kalimat yg BELUM di-flush
    count = 0

    node_words = ["doc_cls"]
    node_entities = ["doc_cls"]
    node_index = 1

    edge_list1 = []
    edge_list2 = []

    while True:
        line = flines[count]
        if line[:7] == "# ::tok":
            # edges dari snt_cls KALIMAT SEBELUMNYA ke tiap node di kalimat itu,
            # plus edge doc_cls <-> snt_cls KALIMAT SEBELUMNYA
            if current_snt_cls_doc_idx is not None:
                for key in current_snt_nodes_dict:
                    edge_list1.append(current_snt_cls_doc_idx)
                    edge_list2.append(current_snt_nodes_dict[key])
                edge_list1.append(0)
                edge_list2.append(current_snt_cls_doc_idx)

            previous_snt_cls_doc_idx = current_snt_cls_doc_idx
            current_snt_nodes_dict = {}

            tok_str = re.sub(r'  +', '  ', line[8:].rstrip('\n'))
            current_snt = [re.sub(r'[^a-zA-Z0-9]', '', tok) or 'x' for tok in tok_str.split(' ')]

            node_words.append("snt_cls")
            node_entities.append("snt_cls")
            current_snt_cls_doc_idx = node_index
            node_index += 1

            # rantai: snt_cls kalimat SEBELUMNYA <-> snt_cls kalimat SEKARANG
            if previous_snt_cls_doc_idx is not None:
                edge_list1.append(previous_snt_cls_doc_idx)
                edge_list2.append(current_snt_cls_doc_idx)

            count += 1
            continue
        elif line[:8] == "# ::node":
            node_info_list = line[8:].split()
            snt_node_id = node_info_list[0]
            word_index = int(node_info_list[-1].split('-')[0])
            word = current_snt[word_index]
            entity = node_info_list[1]
            node_words.append(word)
            node_entities.append(entity)
            current_snt_nodes_dict[snt_node_id] = node_index
            node_index += 1
            count += 1
            continue
        elif line[:8] == "# ::edge":
            edge_info_list = line[8:].split()
            key1 = edge_info_list[-2]
            key2 = edge_info_list[-1]
            n1 = current_snt_nodes_dict[key1]
            n2 = current_snt_nodes_dict[key2]
            edge_list1.append(n1)
            edge_list2.append(n2)
            count += 1
            continue

        count += 1
        if count == len(flines):
            for key in current_snt_nodes_dict:
                edge_list1.append(current_snt_cls_doc_idx)
                edge_list2.append(current_snt_nodes_dict[key])
            edge_list1.append(0)
            edge_list2.append(current_snt_cls_doc_idx)
            break

    num_relations = len(edge_list1)
    return node_words, node_entities, num_relations


pt_files = sorted(fn for fn in os.listdir(SRC_DIR) if fn.endswith('.pt'))
print(f"Ketemu {len(pt_files)} file .pt di {SRC_DIR}", flush=True)
print(f"Hasil akan ditulis ke: {DST_DIR}", flush=True)

n_updated = 0
n_copied_as_is = 0
n_skipped_already_in_dst = 0
n_mismatch = 0
n_failed = 0
rows = []

for fn in pt_files:
    src_path = os.path.join(SRC_DIR, fn)
    dst_path = os.path.join(DST_DIR, fn)
    amr_path = os.path.join(sec_fname, fn[:-2] + "amr")

    # --- Resumable: kalau file ini udah pernah ditulis ke DST_DIR, skip ---
    if os.path.exists(dst_path):
        n_skipped_already_in_dst += 1
        continue

    try:
        g = torch.load(src_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"[SKIP] gagal load {fn}: {repr(e)}", flush=True)
        n_failed += 1
        continue

    already_has_labels = hasattr(g, 'word') and hasattr(g, 'entity')
    num_nodes_pt = g.num_nodes if g.num_nodes is not None else g.x.shape[0]

    if already_has_labels:
        # udah ada label dari sononya -> tinggal disalin ke DST_DIR apa adanya
        num_relations = g.num_relations if hasattr(g, 'num_relations') else g.edge_index.shape[1] // 2
        tmp_path = dst_path + ".tmp"
        torch.save(g, tmp_path)
        os.replace(tmp_path, dst_path)
        n_copied_as_is += 1
        rows.append({"filename": fn, "num_nodes": num_nodes_pt, "num_relations": num_relations, "has_labels": True})
        continue

    if not os.path.exists(amr_path):
        print(f"[SKIP] file .amr sumber tidak ketemu utk {fn} (dicari di {amr_path})", flush=True)
        n_failed += 1
        continue

    try:
        node_words, node_entities, num_relations = parse_amr_labels_and_relations(amr_path)
    except Exception as e:
        print(f"[SKIP] gagal parsing {amr_path}: {repr(e)}", flush=True)
        traceback.print_exc()
        n_failed += 1
        continue

    if len(node_words) != num_nodes_pt:
        print(f"[MISMATCH] {fn}: jumlah node hasil parsing ulang ({len(node_words)}) "
              f"!= jumlah node di .pt ({num_nodes_pt}). File ini DILEWATI, tidak diubah.", flush=True)
        n_mismatch += 1
        rows.append({"filename": fn, "num_nodes": num_nodes_pt, "num_relations": "MISMATCH", "has_labels": False})
        continue

    g.word = node_words
    g.entity = node_entities
    g.label = [f"{e}|{w}" for e, w in zip(node_entities, node_words)]
    g.num_relations = num_relations

    tmp_path = dst_path + ".tmp"
    torch.save(g, tmp_path)
    os.replace(tmp_path, dst_path)

    n_updated += 1
    rows.append({"filename": fn, "num_nodes": num_nodes_pt, "num_relations": num_relations, "has_labels": True})
    print(f"[OK] {fn}: {num_nodes_pt} nodes, {num_relations} relations -> label ditempel", flush=True)

summary = (f"Selesai. label_ditempel={n_updated}, sudah_ada_label(copy_as_is)={n_copied_as_is}, "
           f"sudah_ada_di_dst(skip)={n_skipped_already_in_dst}, mismatch(skip)={n_mismatch}, "
           f"gagal={n_failed}, total_di_src={len(pt_files)}")
print(summary, flush=True)

csv_path = 'graph_stats_plan_C.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "num_nodes", "num_relations", "has_labels"])
    writer.writeheader()
    writer.writerows(rows)
print(f"CSV statistik graf disimpan di: {csv_path} ({len(rows)} baris)", flush=True)
