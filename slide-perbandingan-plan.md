# Slide: Perbandingan Strategi Konstruksi Graf AMR (Plan A–E)

---

## Strategi Konstruksi Graf AMR: Plan A hingga E

### Latar Belakang
Setiap dokumen terdiri dari **N kalimat**, masing-masing di-parse AMR menjadi node (konsep) dan edge (relasi semantik). Perbedaan antar plan terletak pada **cara menghubungkan antar kalimat** dan **node spesial** yang ditambahkan.

---

### Node Spesial

| Node | Keterangan | Hadir di Plan |
|---|---|---|
| `doc_cls` | Representasi seluruh dokumen (embedding zero-vector) | **A, B, C, D, E** |
| `snt_cls` | Representasi per kalimat (embedding zero-vector) | **C, D** |
| `snt_root` | Root node AMR dari tiap kalimat | **A, B** |

---

### Perbandingan Strategi Edge Antar Kalimat

| Plan | Edge Intra-Kalimat | Edge Inter-Kalimat | Node Spesial Tambahan |
|---|---|---|---|
| **A** | AMR edges | `doc_cls → snt_root` tiap kalimat | `snt_root` |
| **B** | AMR edges | `doc_cls → snt_root` + **chain** `snt_rootₙ → snt_rootₙ₊₁` | `snt_root` |
| **C** | AMR edges + `snt_cls → semua node dalam kalimat` | `doc_cls → snt_cls` + **chain** `snt_clsₙ → snt_clsₙ₊₁` | `snt_cls` |
| **D** | AMR edges + `snt_cls → semua node` | Plan C + **`doc_cls → setiap node konten` langsung** | `snt_cls` |
| **E** | AMR edges | `doc_cls → setiap node konten` langsung (tanpa perantara) | — |

---

### Visualisasi Struktur Graf

```
PLAN A                    PLAN B                    PLAN C
                                                    
doc_cls                   doc_cls                   doc_cls
  │                         │                         │
  ├→ snt₁_root            ├→ snt₁_root ─→ snt₂_root  ├→ snt₁_cls ─→ snt₂_cls
  ├→ snt₂_root            ├→ snt₂_root              │      │              │
  └→ snt₃_root            └→ snt₃_root              │   ↓ nodes       ↓ nodes
       ↓ AMR                    ↓ AMR                └→ snt₃_cls
      nodes                    nodes                      │
                                                       ↓ nodes


PLAN D                              PLAN E
                                    
doc_cls                             doc_cls
  │  ╲                               │
  │   ╲──────────────────────┐       ├→ node₁
  ├→ snt₁_cls → snt₂_cls    │       ├→ node₂
  │       │          │       │       ├→ node₃
  │    ↓ nodes   ↓ nodes    │       └→ ... (semua node langsung)
  │       │          │       │            + AMR edges intra-kalimat
  └───────┴──────────┘ ←────┘
  (doc_cls juga langsung ke semua node konten)
```

---

### Ringkasan Karakteristik

| | **A** | **B** | **C** | **D** | **E** |
|---|:---:|:---:|:---:|:---:|:---:|
| `doc_cls` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `snt_cls` | ❌ | ❌ | ✅ | ✅ | ❌ |
| Chain antar kalimat | ❌ | ✅ root | ✅ cls | ✅ cls | ❌ |
| `doc_cls → node` langsung | ❌ | ❌ | ❌ | ✅ | ✅ |
| Hierarki | 2 level | 2 level | **3 level** | 3 level | 2 level (flat) |
| Density | ⬛ | ⬛⬛ | ⬛⬛⬛ | ⬛⬛⬛⬛⬛ | ⬛⬛⬛ |

---

### Mengapa Plan C Terbaik di Eksperimen?

> Plan C menghadirkan **hierarki tiga level** (`doc_cls → snt_cls → node`) yang cukup untuk menangkap informasi global dokumen tanpa over-connecting.
>
> - **Plan A/B** terlalu *sparse* — informasi inter-kalimat hanya lewat satu root node
> - **Plan D** terlalu *dense* — terlalu banyak edge ke `doc_cls` memperlemah sinyal lokal AMR
> - **Plan E** *flat* — tidak ada agregator level kalimat, `doc_cls` menjadi super-hub dengan degree sangat tinggi
> - **Plan C** seimbang: `snt_cls` sebagai agregator per kalimat + chain urutan kalimat terjaga
