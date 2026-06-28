import os
# Kurangi fragmentasi memori CUDA. HARUS di-set sebelum torch meng-inisialisasi
# CUDA. Di Kaggle: taruh ini di CELL PERTAMA dan restart session kalau torch
# sudah pernah di-import di cell sebelumnya.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time
import glob
import torch as th
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# =====================================================================
# KONFIGURASI KAGGLE - DATASET AMD (IDX annual-report, raw English)
# Versi SINGLE-GPU
#
# TASK: KLASIFIKASI 3 KELAS (YoY_Close_Category)
#   rendah = 0 | netral = 1 | tinggi = 2
# Target diambil APA ADANYA dari mda_auto_from_2016_not_adj_merge_fold.csv,
# disiapkan oleh build-yoy-category-labels.py -> idx-yoy-trainvalid/test.csv
# =====================================================================
# Path dataset Kaggle (read-only). Sesuaikan slug dengan dataset yang
# kamu upload. Isi dataset yang diharapkan:
#   <slug>/idx-yoy-trainvalid.csv
#   <slug>/idx-yoy-test.csv
#   <slug>/truly-all-results-graphs-hk-finbert-plan-{plan}/*.pt
# Cek path persisnya di panel kanan Kaggle ("Input") atau: !ls /kaggle/input
INPUT_DIR  = "/kaggle/input/amd-amr-gnn"     # folder berisi 2 CSV (idx-yoy-*.csv)
GRAPHS_DIR = "/kaggle/input/amd-amr-gnn"     # folder induk berisi truly-all-results-graphs-hk-finbert-plan-*/
OUTPUT_DIR = "/kaggle/working"               # folder output (writable)

# Nama file CSV label. Versi "-available" = sudah difilter ke dokumen yang .pt-nya
# SUDAH ter-generate (38 dok: 27 train/valid + 10 test) -> untuk SANITY-RUN.
# Kalau .pt sudah lengkap nanti, ganti ke "idx-yoy-trainvalid.csv"/"idx-yoy-test.csv"
# (baris tanpa .pt tetap di-skip otomatis saat runtime).
TRAINVALID_CSV = "idx-yoy-trainvalid-available.csv"
TEST_CSV       = "idx-yoy-test-available.csv"

# --- Klasifikasi ---
CLASS_NAMES = ["rendah", "netral", "tinggi"]   # index = label integer
NUM_CLASSES = len(CLASS_NAMES)

# Parameter run
sec                = "amd"                       # label dataset untuk nama file output
bv                 = "yoy_close_category"        # nama target (untuk penamaan file output)
TARGET_COL         = "label"                     # kolom integer kelas di CSV
hist               = "no-hist"
learning_rate      = 1e-4
total_epochs       = 5
plan               = "A"                          # pilih A/B/C/D/E (sesuai folder .pt)
start_epoch        = 0
end_epoch          = 5
num_of_attn_heads  = 4
# =====================================================================

num_of_layers = 4

# Folder output di /kaggle/working
TRACKING_DIR  = os.path.join(OUTPUT_DIR, "tracking-files")
MSE_DIR       = os.path.join(OUTPUT_DIR, "mse-results")
BEST_VL_DIR   = os.path.join(OUTPUT_DIR, "best-valid-losses")
WEIGHTS_DIR   = os.path.join(OUTPUT_DIR, "saved-model-weights")
TRAINLOG_DIR  = os.path.join(OUTPUT_DIR, "training-logs")

for d in [TRACKING_DIR, MSE_DIR, BEST_VL_DIR, WEIGHTS_DIR, TRAINLOG_DIR]:
    os.makedirs(d, exist_ok=True)

if th.cuda.is_available():
    device = th.device("cuda")
else:
    device = th.device("cpu")

print(f"[setup] device     : {device}")
print(f"[setup] torch      : {th.__version__}")

# Dim tersembunyi TETAP di semua layer (versi earnings-call melipatgandakan dim
# tiap layer -> 12288 dim/node -> OOM untuk graph IDX puluhan ribu node).
# Naikkan/turunkan HIDDEN_DIM kalau mau model lebih besar/kecil (harus habis
# dibagi num_of_attn_heads).
HIDDEN_DIM = 256
assert HIDDEN_DIM % num_of_attn_heads == 0, "HIDDEN_DIM harus habis dibagi num_of_attn_heads"

class Net(nn.Module):
    def __init__(self,num_heads,hidden=HIDDEN_DIM):
        super(Net, self).__init__()
        self.num_attn_heads = num_heads
        per_head = hidden // num_heads
        self.layer0 = GATv2Conv(768,    per_head, heads=num_heads, concat=True)
        self.layer1 = GATv2Conv(hidden, per_head, heads=num_heads, concat=True)
        self.layer2 = GATv2Conv(hidden, per_head, heads=num_heads, concat=True)
        self.layer3 = GATv2Conv(hidden, per_head, heads=num_heads, concat=True)
        self.dropout = nn.Dropout(0.1)
        self.leakyrelu =  nn.LeakyReLU()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, NUM_CLASSES)   # logits 3 kelas

    def forward(self, features, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=features.size(0))

        # Model sudah kecil -> tidak perlu gradient checkpointing (lebih cepat).
        x = self.layer0(features, edge_index)
        del features
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = self.layer3(x, edge_index)
        del edge_index

        x = x[0:1]                    # doc_cls = node pertama graph -> jaga dim batch (1, hidden)

        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)

        y = self.fc2(x)
        return x, y

print(f"[setup] building model  : GATv2 hidden={HIDDEN_DIM} | {num_of_layers} layers | {num_of_attn_heads} heads")
net = Net(num_of_attn_heads).to(device)
print(f"[setup] model params    : {sum(p.numel() for p in net.parameters()):,}")

def get_graph(graph_filename):
    return th.load(f'{GRAPHS_DIR}/truly-all-results-graphs-hk-finbert-plan-{plan}/{graph_filename}.pt', weights_only=False)

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
print(f"\n[data] loading train/valid CSV ...")
df = pd.read_csv(f'{INPUT_DIR}/{TRAINVALID_CSV}')

print(f"[data] loading test CSV ...")
df_test = pd.read_csv(f'{INPUT_DIR}/{TEST_CSV}')

# Hanya latih/uji dokumen yang graph .pt-nya SUDAH ter-generate.
# Baris CSV yang belum punya .pt di-skip otomatis (belum diproses dari .amr).
_graph_dir = f'{GRAPHS_DIR}/truly-all-results-graphs-hk-finbert-plan-{plan}'
available_graphs = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f'{_graph_dir}/*.pt')}
print(f"[data] available .pt graphs (plan {plan}): {len(available_graphs)} in {_graph_dir}")

_n_tv, _n_te = len(df), len(df_test)
df = df[df['ticker_and_date'].isin(available_graphs)].reset_index(drop=True)
df_test = df_test[df_test['ticker_and_date'].isin(available_graphs)].reset_index(drop=True)
print(f"[data] trainvalid kept {len(df)}/{_n_tv} | test kept {len(df_test)}/{_n_te} (sisanya di-skip: belum ada .pt)")

train_index, valid_index, train_labels, valid_labels = train_test_split(df['ticker_and_date'],
    df[TARGET_COL],
    shuffle=False,
    train_size=0.8)

valid_index = valid_index.reset_index(drop=True)

test_index = df_test['ticker_and_date']

test_labels = df_test[TARGET_COL]

# Label kelas -> tensor long (untuk CrossEntropyLoss)
train_y = th.tensor(train_labels.tolist(), dtype=th.long).to(device)
valid_y = th.tensor(valid_labels.tolist(), dtype=th.long).to(device)
test_y = th.tensor(test_labels.tolist(), dtype=th.long).to(device)

print(f"[data] train={len(train_index)} | valid={len(valid_index)} | test={len(test_index)}")
print(f"[data] target column : {TARGET_COL} ({bv}) | kelas : {CLASS_NAMES} | plan : {plan}")

# Bobot kelas (inverse-frequency) untuk imbangi distribusi kelas yang tak rata.
_counts = np.bincount(train_labels.to_numpy(), minlength=NUM_CLASSES).astype(float)
_counts[_counts == 0] = 1.0
class_weights = th.tensor(_counts.sum() / (NUM_CLASSES * _counts), dtype=th.float).to(device)
print(f"[data] count kelas (train): {_counts.astype(int).tolist()} | bobot: {class_weights.cpu().numpy().round(3).tolist()}\n")

criterion = nn.CrossEntropyLoss(weight=class_weights)

# Suffix nama file output (dipakai berulang)
suffix = f"{num_of_layers}_layers_{num_of_attn_heads}_heads_{sec}_{bv}_eps{total_epochs}_lr={learning_rate:.1e}_plan-{plan}_{hist}"

def train(model,epoch):

    memory_file = open(f'{TRACKING_DIR}/memory_GATv2-dim-16x_{num_of_layers}_layers_{num_of_attn_heads}_heads_{sec}_{bv}_ep_{epoch}_lr={learning_rate:.1e}_plan-{plan}_{hist}.txt', 'a+')
    model.train()

    total_loss, total_accuracy = 0, 0
    total_preds = []
    xs = []

    pbar = tqdm(range(len(train_index)), desc=f"  train ep{epoch:03d}", unit="doc", leave=False)
    for i in pbar:

        memory_file.write("doc num: "+str(i)+"\n")
        memory_file.flush()

        sent_id = get_graph(train_index[i])
        features = sent_id.x.to(device)
        edge_index = sent_id.edge_index.to(device)
        labels = train_y[i].unsqueeze(0)

        model.zero_grad()

        x, logits = model(features, edge_index)

        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1).detach().cpu().numpy()   # kelas prediksi
        x = x.detach().cpu().numpy().ravel()

        total_loss = total_loss + loss.item()

        loss.backward()

        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_preds.append(preds)
        xs.append(x)

        # Bebaskan memori GPU per dokumen (graph diproses 1 per 1)
        del sent_id, features, edge_index, labels, loss, preds, x
        if device.type == "cuda":
            th.cuda.empty_cache()

        pbar.set_postfix(loss=f"{total_loss/(i+1):.6f}")

    avg_loss = total_loss / len(train_index)

    xs = np.array(xs)

    total_preds = np.concatenate(total_preds, axis=0)
    memory_file.close()
    return avg_loss, total_preds, xs

def evaluate(model):

    model.eval()

    total_loss, total_accuracy = 0.0, 0.0
    total_preds = []
    total_xs = []

    for i in tqdm(range(len(valid_index)), desc="  evaluate", unit="doc", leave=False):

        sent_id = get_graph(valid_index[i])
        features = sent_id.x.to(device)
        edge_index = sent_id.edge_index.to(device)
        labels = valid_y[i].unsqueeze(0)

        with th.no_grad():

            x, logits = model(features, edge_index)

            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1).detach().cpu().numpy()

            total_loss = total_loss + loss.item()

            total_preds.append(preds)

            x = x.detach().cpu().numpy().ravel()

            total_xs.append(x)

    avg_loss = total_loss / len(valid_index)

    total_xs = np.array(total_xs)

    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds, total_xs

def test(model,test_index):

    model.eval()

    total_xs = []
    total_preds=[]
    total_loss = 0.0

    for i in tqdm(range(len(test_index)), desc="  test    ", unit="doc", leave=False):

        sent_id = get_graph(test_index[i])
        features = sent_id.x.to(device)
        edge_index = sent_id.edge_index.to(device)
        labels = test_y[i].unsqueeze(0)

        with th.no_grad():
            x, logits = model(features, edge_index)

            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            total_loss = total_loss + loss.item()

            x = x.detach().cpu().numpy().ravel()
            total_preds.append(preds)
            total_xs.append(x)

    total_xs = np.array(total_xs)
    avg_loss = total_loss / len(test_index)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds, total_xs

# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
optimizer = th.optim.Adam(net.parameters(), lr=learning_rate)

best_vl_path = f'{BEST_VL_DIR}/best_valid_loss_GATv2-dim-16x_{suffix}.txt'
weights_path = f'{WEIGHTS_DIR}/saved_weights_GATv2-dim-16x_{suffix}.pt'

if os.path.isfile(best_vl_path):
    with open(best_vl_path) as f:
        lines = f.readlines()
    best_valid_loss = float(lines[0])
    best_epoch = int(lines[1])
    print(f"[resume] loaded prev best_valid_loss={best_valid_loss:.6f} (epoch {best_epoch})")
else:
    best_valid_loss = float('inf')
    best_epoch = 0

if os.path.isfile(weights_path):
    net.load_state_dict(th.load(weights_path, weights_only=True))
    print(f"[resume] loaded saved weights from {weights_path}")

print(f"\n[train] starting epochs {start_epoch} → {end_epoch}  (total={total_epochs}  lr={learning_rate})\n")
training_log = open(f'{TRAINLOG_DIR}/training_log_GATv2-dim-16x_{suffix}.txt', 'a+')
for epoch in range(start_epoch,end_epoch):
    print(f"[epoch {epoch:03d}] training ...")
    start_time = time.time()
    train_loss, _,_ = train(net,epoch)
    print(f"[epoch {epoch:03d}] evaluating ...")
    valid_loss,_ ,_= evaluate(net)

    end_time = time.time()
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_epoch = epoch
        model_to_save = net.module if hasattr(net, 'module') else net
        th.save(model_to_save.state_dict(), weights_path)
        print(f"[epoch {epoch:03d}] ** new best model saved (valid_loss={valid_loss:.9f}) **")
    print("Target: "+bv+" | Layers {:03d}| Heads {:03d}| Epoch {:03d} | Train Loss {:.9f} | Valid Loss {:.9f}| lr {:.9f} | time {:.3f}".format(
            num_of_layers,num_of_attn_heads, epoch, train_loss, valid_loss, learning_rate, end_time-start_time))
    training_log.write("Target: "+bv+" | Layers {:03d}| Heads {:03d}| Epoch {:03d} | Train Loss {:.9f} | Valid Loss {:.9f}| lr {:.9f} | time {:.3f}\n".format(
            num_of_layers,num_of_attn_heads, epoch, train_loss, valid_loss, learning_rate, end_time-start_time))
    training_log.flush()
training_log.close()
print(f"\n[train] done. best_valid_loss={best_valid_loss:.9f} at epoch {best_epoch}")

valid_loss_file = open(best_vl_path, 'w')
valid_loss_file.write(str(best_valid_loss)+"\n")
valid_loss_file.write(str(best_epoch))
valid_loss_file.close()

if end_epoch == total_epochs:
    print(f"\n[final] loading best model weights ...")
    model = Net(num_of_attn_heads).to(device)

    model.load_state_dict(th.load(weights_path, weights_only=True))

    print(f"[final] running validation inference ...")
    _ , preds, xs_valid = evaluate(model)
    preds = np.asarray(preds)
    valid_y = valid_y.cpu().data.numpy()
    valid_acc = accuracy_score(valid_y, preds)
    valid_f1  = f1_score(valid_y, preds, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)

    if len(test_index) == 0:
        # Belum ada .pt untuk dokumen test (mis. semua .pt masih dari tahun trainvalid).
        print("[final] tidak ada graph test (.pt) — lewati test inference, simpan metrik valid saja")
        mse_file = open(f'{MSE_DIR}/mse_GATv2-dim-16x_{suffix}.txt', "w")
        mse_file.write("NO_TEST---bare---"+str(valid_acc)+"\n")
        mse_file.write(str(best_valid_loss)+" after epoch: "+str(best_epoch)+"\n")
        mse_file.write("valid_macro_f1="+str(valid_f1)+"\n")
        mse_file.close()
        print(f"[final] valid_acc={valid_acc:.4f} | valid_macro_f1={valid_f1:.4f}")
        print(f"[final] results saved to {MSE_DIR}/")
    else:
        print(f"[final] running test inference ...")
        _, preds,xs_test= test(model,test_index)
        preds = np.asarray(preds)
        np.savetxt(f'{MSE_DIR}/test_preds_GATv2-dim-16x_{suffix}.csv',preds, fmt='%d', delimiter=',')
        test_y = test_y.cpu().data.numpy()
        np.savetxt(f'{MSE_DIR}/test_trues_GATv2-dim-16x_{suffix}.csv',test_y, fmt='%d', delimiter=',')
        test_acc = accuracy_score(test_y, preds)
        test_f1  = f1_score(test_y, preds, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)

        print("GATv2 "+str(num_of_layers)+" layers "+str(num_of_attn_heads)+" heads acc: "+str(test_acc)+'---bare---'+str(valid_acc))

        mse_file = open(f'{MSE_DIR}/mse_GATv2-dim-16x_{suffix}.txt', "w")
        mse_file.write(str(test_acc)+'---bare---'+str(valid_acc)+"\n")
        mse_file.write(str(best_valid_loss)+" after epoch: "+str(best_epoch)+"\n")
        mse_file.write("test_macro_f1="+str(test_f1)+" valid_macro_f1="+str(valid_f1)+"\n")
        mse_file.close()
        print(f"[final] test_acc={test_acc:.4f} (macro_f1={test_f1:.4f}) | valid_acc={valid_acc:.4f} (macro_f1={valid_f1:.4f})")
        print(f"[final] results saved to {MSE_DIR}/")
