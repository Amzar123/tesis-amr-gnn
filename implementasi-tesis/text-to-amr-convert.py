"""
Konversi file .txt (1 kalimat per baris) -> file .amr dalam format beralignment
JAMR yang dibutuhkan oleh amrs-to-graphs-plan-*-new.py:

    # ::tok  <kalimat ter-tokenisasi spasi>
    # ::node \t <var> \t <concept> \t <start>-<end>     (alignment indeks kata)
    # ::edge \t <role> \t <src_var> \t <dst_var>
    # ::root \t <var> \t <concept>

Pakai amrlib (parse) + RBWAligner (alignment node->token) + penman (ekstraksi).

PRASYARAT (jalankan sekali):
    pip install amrlib penman
    python -m spacy download en_core_web_sm
    python -c "import amrlib; amrlib.download('model_parse_xfm_bart_large')"
    # atau download manual model stog lalu set path-nya

GPU & PARALEL:
    Script menjalankan SATU proses per GPU (lihat GPU_IDS). Tiap proses di-pin
    ke satu GPU lewat CUDA_VISIBLE_DEVICES, load model-nya sendiri, dan
    menggarap subset dokumen (dibagi round-robin). Dengan 2x T4 -> ~2x lebih
    cepat. Parsing per-batch di GPU; align/ekstraksi di CPU (ringan).

RESUME (lanjut tanpa ngulang):
    Sebelum memproses sebuah dokumen, dicek apakah <base>.amr SUDAH ADA di
    OUT_DIR. Kalau ada -> dilewati. Penulisan file pakai tulis-ke-.tmp lalu
    rename atomik, jadi file yang ada PASTI lengkap (tidak ada sisa setengah
    jadi kalau proses ke-kill di tengah jalan). Aman dijalankan ulang kapan pun.
"""
import os
import subprocess
import multiprocessing as mp

# hindari warning fork dari HuggingFace tokenizers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# =====================================================================
# KONFIGURASI
# =====================================================================
RAW_DIR  = "/kaggle/input/datasets/amzar11/raw-english-2159"   # folder berisi *.txt (dicari rekursif)
OUT_DIR  = "/kaggle/working/idx-amrs"          # folder output *.amr
NAMES_OUT = "/kaggle/working/idx-amr-file-names.txt"  # daftar nama file (untuk builder)
# Folder model parser amrlib. Set "AUTO" -> cari sendiri di /kaggle/input
# (nama folder: model_parse_xfm_bart_large*). Isi path manual kalau mau spesifik,
# atau "" untuk pakai model default amrlib di site-packages.
#
# PENTING: taruh model di /kaggle/input (attach sebagai dataset / output versi
# lama), JANGAN di /kaggle/working — semua isi /kaggle/working jadi "Output"
# notebook, jadi model di sana ikut keangkut (GB-an) & bikin save version lambat.
MODEL_DIR = "/kaggle/input/models/amzar11/amrlib-model/other/default/1/model_parse_xfm_bart_large-v0_1_0"
# Folder berisi .amr dari run sebelumnya untuk di-seed ke OUT_DIR (RESUME lintas
# versi). "AUTO" -> cari folder bernama idx-amrs di /kaggle/input. "" = nonaktif.
SEED_DIR = "AUTO"
USE_GPU  = True
# Daftar id GPU yang dipakai (1 proses per GPU). None = auto-deteksi semua GPU.
# Untuk 2x T4 Kaggle ini jadi [0, 1]. Set [0] kalau mau 1 GPU saja.
GPU_IDS  = None
BATCH_SIZE = 32   # jumlah kalimat diparse sekaligus di GPU (naikkan jika VRAM cukup)
MIN_CHARS = 5     # lewati baris terlalu pendek (ikut logika text_to_amr.py asli)
LIMIT_DOCS = 100 # set angka (mis. 3) untuk MODE TES beberapa dokumen dulu
# =====================================================================


def resolve_raw_txt():
    """Kumpulkan path *.txt secara REKURSIF di bawah RAW_DIR (file .txt sering
    berada di subfolder, mis. raw-english-2159/Raw_English_2159/). RAW_DIR='AUTO'
    atau tidak ada -> cari di /kaggle/input. Return list path lengkap, urut nama."""
    def walk_txt(root):
        out = []
        for dp, _d, fns in os.walk(root, followlinks=True):
            for fn in fns:
                if fn.endswith(".txt"):
                    out.append(os.path.join(dp, fn))
        return out

    if RAW_DIR != "AUTO" and os.path.isdir(RAW_DIR):
        txt = walk_txt(RAW_DIR)
        if txt:
            return sorted(txt, key=os.path.basename)
    # fallback: cari di seluruh /kaggle/input, prioritaskan yang ber-"raw"
    allt = walk_txt("/kaggle/input")
    hinted = [p for p in allt if "raw" in p.lower()]
    return sorted(hinted if hinted else allt, key=os.path.basename)


def resolve_model_dir():
    """Tentukan folder model. Pakai MODEL_DIR kalau ada; 'AUTO' atau path tidak
    ada -> cari di /kaggle/input. '' -> model default amrlib."""
    if MODEL_DIR == "":
        return ""
    if MODEL_DIR != "AUTO" and os.path.isdir(MODEL_DIR):
        return MODEL_DIR
    import glob
    # cari folder model_parse_xfm_bart_large* yang berisi config.json
    cands = glob.glob("/kaggle/input/**/model_parse_xfm_bart_large*", recursive=True)
    for c in sorted(cands):
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "config.json")):
            return c
    print("[warn] model tidak ketemu di /kaggle/input, pakai model default amrlib",
          flush=True)
    return ""


def seed_existing_amrs():
    """Copy .amr dari run sebelumnya (SEED_DIR / auto di /kaggle/input) ke OUT_DIR.
    Hanya menyalin yang belum ada. Untuk RESUME lintas versi notebook."""
    if not SEED_DIR:
        return 0
    import glob
    import shutil
    if SEED_DIR == "AUTO":
        # kumpulkan semua *.amr di bawah folder bernama idx-amrs di /kaggle/input
        srcs = glob.glob("/kaggle/input/**/idx-amrs/*.amr", recursive=True)
    else:
        srcs = glob.glob(os.path.join(SEED_DIR, "*.amr"))
    n = 0
    for src in srcs:
        dst = os.path.join(OUT_DIR, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            n += 1
    if n:
        print(f"[setup] seed {n} file .amr dari run sebelumnya -> {OUT_DIR}",
              flush=True)
    return n


def block_from_penman(pen):
    """PENMAN str -> blok teks format JAMR beralignment. None jika gagal.
    Tahap ini CPU (align rule-based + ekstraksi)."""
    # import lokal supaya child process (spawn) yang baru set CUDA_VISIBLE_DEVICES
    import json
    from penman import surface
    from amrlib.alignments.rbw_aligner import RBWAligner
    from amrlib.graph_processing.annotator import add_lemmas

    if pen is None:
        return None
    try:
        pgraph = add_lemmas(pen, snt_key='snt')
        aligner = RBWAligner.from_penman_w_json(pgraph)
        g = aligner.get_penman_graph()
    except Exception:
        return None

    toks = g.metadata.get('tokens')
    if toks is None:
        return None
    tokens = json.loads(toks) if isinstance(toks, str) else toks
    if not tokens:
        return None

    align = surface.alignments(g)   # dict: triple -> Alignment(indices)
    n_tok = len(tokens)

    lines = ["# ::tok " + " ".join(tokens)]

    # node: instance triple (var, ':instance', concept)
    n_aligned = 0
    for inst in g.instances():
        a = align.get(inst)
        if a is not None and len(a.indices) > 0 and a.indices[0] < n_tok:
            start = a.indices[0]
            n_aligned += 1
        else:
            start = 0   # fallback: node abstrak tanpa alignment -> token 0
        lines.append(f"# ::node\t{inst.source}\t{inst.target}\t{start}-{start+1}")

    # edge: (src_var, role, tgt_var) -> 2 field terakhir = src, tgt
    for e in g.edges():
        lines.append(f"# ::edge\t{e.role.lstrip(':')}\t{e.source}\t{e.target}")

    # root
    if g.top is None:
        return None
    top_concept = next((i.target for i in g.instances() if i.source == g.top), g.top)
    lines.append(f"# ::root\t{g.top}\t{top_concept}")

    return "\n".join(lines), n_aligned, len(list(g.instances()))


def worker(gpu_id, files, tag):
    """Proses 1 GPU: load model lalu garap subset `files`.
    Melewati dokumen yang .amr-nya sudah ada (RESUME)."""
    # Pin proses ini ke 1 GPU. HARUS sebelum CUDA diinisialisasi (load model).
    if USE_GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import amrlib

    # Karena CUDA_VISIBLE_DEVICES sudah membatasi ke 1 GPU, device cukup "cuda".
    device = "cuda" if USE_GPU else "cpu"
    print(f"[{tag}] loading stog model on physical GPU {gpu_id} ({device}) ...",
          flush=True)
    model_dir = resolve_model_dir()
    kw = {"device": device}
    if model_dir:
        kw["model_dir"] = model_dir
    try:
        stog = amrlib.load_stog_model(batch_size=BATCH_SIZE, **kw)
    except TypeError:
        stog = amrlib.load_stog_model(**kw)

    def parse_batch(sentences):
        graphs = []
        for i in range(0, len(sentences), BATCH_SIZE):
            chunk = sentences[i:i + BATCH_SIZE]
            try:
                graphs.extend(stog.parse_sents(chunk))
            except Exception:
                graphs.extend([None] * len(chunk))   # batch gagal -> tandai None
        return graphs

    n_ok, n_skip, n_fail_doc = 0, 0, 0
    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(OUT_DIR, base + ".amr")

        # --- RESUME: sudah ada & lengkap -> lewati ---
        if os.path.exists(out_path):
            n_skip += 1
            continue

        with open(path) as f:
            sentences = [s.strip() for s in f if len(s.strip()) >= MIN_CHARS]

        # 1) parse semua kalimat dokumen ini per-batch di GPU
        penmans = parse_batch(sentences)

        # 2) align + konversi tiap graph (CPU)
        blocks, tot_nodes, tot_aligned = [], 0, 0
        for pen in penmans:
            res = block_from_penman(pen)
            if res is None:
                continue
            block, na, nn = res
            blocks.append(block)
            tot_aligned += na
            tot_nodes += nn

        if not blocks:
            n_fail_doc += 1
            print(f"[{tag}] {base}: tidak ada kalimat berhasil di-parse, dilewati",
                  flush=True)
            continue

        # tulis atomik: .tmp lalu rename, supaya file final selalu lengkap
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write("\n".join(blocks) + "\n")
        os.replace(tmp_path, out_path)
        n_ok += 1

        align_pct = (tot_aligned / tot_nodes * 100) if tot_nodes else 0
        print(f"[{tag}] {base}: {len(blocks)} kalimat, {tot_nodes} node, "
              f"alignment {align_pct:.0f}%", flush=True)

    print(f"[{tag}] selesai: {n_ok} baru, {n_skip} dilewati (sudah ada), "
          f"{n_fail_doc} gagal total", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- seed .amr lama (resume lintas versi notebook) ---
    seed_existing_amrs()

    # --- tentukan daftar GPU (pakai nvidia-smi, JANGAN import torch di induk
    #     supaya CUDA belum diinisialisasi -> aman untuk fork) ---
    if not USE_GPU:
        gpu_ids = [0]   # 1 proses CPU
    elif GPU_IDS is not None:
        gpu_ids = GPU_IDS
    else:
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
            n = sum(1 for ln in out.stdout.splitlines() if ln.strip().startswith("GPU "))
            gpu_ids = list(range(n)) if n > 0 else [0]
        except Exception:
            gpu_ids = [0]
    print(f"[setup] memakai GPU id: {gpu_ids}", flush=True)

    # --- daftar dokumen (path lengkap, rekursif) ---
    files = resolve_raw_txt()
    if LIMIT_DOCS:
        files = files[:LIMIT_DOCS]
    total = len(files)
    if total == 0:
        print("[error] tidak menemukan file .txt mentah. Cek RAW_DIR.", flush=True)
        return
    print(f"[setup] sumber .txt: {os.path.dirname(files[0])}", flush=True)

    # hitung yang sudah ada (info resume)
    n_already = sum(
        1 for p in files
        if os.path.exists(os.path.join(
            OUT_DIR, os.path.splitext(os.path.basename(p))[0] + ".amr"))
    )
    print(f"[setup] {total} dokumen total, {n_already} sudah ada (akan dilewati), "
          f"{total - n_already} akan diproses", flush=True)

    # --- bagi dokumen round-robin antar GPU (seimbang) ---
    n_gpu = len(gpu_ids)
    shards = [files[i::n_gpu] for i in range(n_gpu)]

    # --- jalankan 1 proses per GPU ---
    if n_gpu == 1:
        # 1 GPU: jalankan langsung tanpa multiprocessing
        worker(gpu_ids[0], shards[0], f"gpu{gpu_ids[0]}")
    else:
        # fork (bukan spawn): induk belum pernah import torch/amrlib jadi CUDA
        # belum aktif -> fork aman, dan child mewarisi fungsi worker dari memori
        # (tidak perlu import ulang __main__) -> jalan baik inline maupun via file.
        ctx = mp.get_context("fork")
        procs = []
        for gid, shard in zip(gpu_ids, shards):
            p = ctx.Process(target=worker, args=(gid, shard, f"gpu{gid}"))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    # --- daftar nama file: scan OUT_DIR (termasuk hasil run sebelumnya) ---
    names = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".amr"))
    with open(NAMES_OUT, "w") as f:
        f.write("\n".join(names) + "\n")

    print(f"\n[done] {len(names)} dokumen .amr di {OUT_DIR}/", flush=True)
    print(f"[done] daftar nama: {NAMES_OUT}", flush=True)


if __name__ == "__main__":
    main()
