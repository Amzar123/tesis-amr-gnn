import torch
from torch_geometric.data import Data

torch.serialization.add_safe_globals([Data])

# 1. Load file .pt kamu (masukkan path lengkap ke filenya)
path_file = "truly-all-results-graphs-hk-finbert-plan-A/AMEX-FSP_2010-02-24.pt"
graph = torch.load(path_file, weights_only=False)

# 2. Cetak rangkuman objek graf-nya
print(graph)
# Output-nya kurang lebih bakal seperti ini: Data(x=[Jumlah_Node, 768], edge_index=[2, Jumlah_Edge])

# 3. Kalau mau lihat matrix nilai embedding kata-katanya:
print("Isi Fitur Node (Vektor Kata):")
print(graph.x) 

# 4. Kalau mau lihat daftar koneksi indeks antar katanya:
print("Isi Struktur Edge (Koneksi Antar Kata):")
print(graph.edge_index)