import os
import time
import torch as th
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.nn.functional import relu
import psutil
num_of_attn_heads = int(sys.argv[9])
num_of_layers = 4

try:
    os.mkdir('tracking-files')
    os.mkdir('mse-results')
    os.mkdir('best-valid-losses')
    os.mkdir('../saved-model-weights')
    os.mkdir('training-logs')
except OSError as error:
    print(error)

if th.cuda.is_available():
    device = th.device("cuda")
else:
    device = th.device("cpu")

print(f"[setup] device     : {device}")
print(f"[setup] torch      : {th.__version__}")

class Net(nn.Module):
    def __init__(self,num_heads):
        super(Net, self).__init__()
        self.num_attn_heads = num_heads
        # PyG GATv2Conv dengan concat=True (default) secara otomatis melakukan
        # head concatenation, setara dengan manual cat di versi DGL.
        # Dimensi output per layer identik: num_heads * out_channels_per_head.
        self.layer0 = GATv2Conv(768,         int(768/num_heads*2),         heads=num_heads, concat=True)
        self.layer1 = GATv2Conv(768*2,       int(768*2/num_heads*2),       heads=num_heads, concat=True)
        self.layer2 = GATv2Conv(768*2*2,     int(768*2*2/num_heads*2),     heads=num_heads, concat=True)
        self.layer3 = GATv2Conv(768*2*2*2,   int(768*2*2*2/num_heads*2),   heads=num_heads, concat=True)
        self.dropout = nn.Dropout(0.1)
        self.leakyrelu =  nn.LeakyReLU()
        self.fc1 = nn.Linear(768*2*2*2*2, 600*2*2*2*2)
        self.fc2 = nn.Linear(600*2*2*2*2, 1)

    def forward(self, features, edge_index):
        # Setara dengan dgl.add_self_loop(g) di versi DGL
        edge_index, _ = add_self_loops(edge_index, num_nodes=features.size(0))

        # PyG GATv2Conv (features, edge_index) menggantikan DGL GATv2Conv (g, features).
        # concat=True membuat output langsung [N, num_heads * out_channels],
        # setara dengan manual `th.cat(attn_heads, dim=1)` di versi DGL.
        x = self.layer0(features, edge_index)
        del features

        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = self.layer3(x, edge_index)
        del edge_index

        x = x[0]

        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)

        y = self.fc2(x)
        return x, y

print(f"[setup] building model  : GATv2-dim-16x | {num_of_layers} layers | {num_of_attn_heads} heads")
net = Net(num_of_attn_heads).to(device)
print(f"[setup] model params    : {sum(p.numel() for p in net.parameters()):,}")

def get_graph(graph_filename):
    # Menggantikan: dgl.load_graphs(f'.../{graph_filename}.graph', [0])[0][0]
    # File .pt adalah torch_geometric.data.Data object hasil amrs-to-graphs-...-new.py
    return th.load(f'../amrs-to-graphs/truly-all-results-graphs-hk-finbert-plan-{sys.argv[6]}/{graph_filename}.pt', weights_only=False)

sec = sys.argv[1]

print(f"\n[data] loading train/valid CSV ...")
df = pd.read_csv('../../new-tech-2010-to-2018-result.csv')

bv = sys.argv[2]

hist = sys.argv[3]

train_index, valid_index, train_labels, valid_labels = train_test_split(df['ticker_and_date'],
    df[bv],
    shuffle=False,
    train_size=0.8)

valid_index = valid_index.reset_index(drop=True)

print(f"[data] loading test CSV ...")
df_test = pd.read_csv('../../new-tech-2019-result.csv')
test_index = df_test['ticker_and_date']

test_labels = df_test[bv]

train_y = th.tensor(train_labels.tolist()).to(device)
valid_y = th.tensor(valid_labels.tolist()).to(device)
test_y = th.tensor(test_labels.tolist()).to(device)

print(f"[data] train={len(train_index)} | valid={len(valid_index)} | test={len(test_index)}")
print(f"[data] target column : {bv} | plan : {sys.argv[6]}\n")

mse_loss  = nn.MSELoss()

def train(model,epoch):

    memory_file = open('tracking-files/memory_GATv2-dim-16x_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_ep_'+str(epoch)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+sys.argv[3]+'.txt', 'a+')
    model.train()

    total_loss, total_accuracy = 0, 0
    total_preds = []
    total_hist = []
    xs = []

    pbar = tqdm(range(len(train_index)), desc=f"  train ep{epoch:03d}", unit="doc", leave=False)
    for i in pbar:

        memory_file.write("doc num: "+str(i)+"\n")
        memory_file.flush()

        sent_id = get_graph(train_index[i])
        # Menggantikan: features = sent_id.ndata['h']
        features = sent_id.x.to(device)
        edge_index = sent_id.edge_index.to(device)
        labels = train_y[i].unsqueeze(0)

        model.zero_grad()

        # Menggantikan: model(sent_id, features)
        x, preds = model(features, edge_index)

        loss = mse_loss(preds, labels)
        preds = preds.detach().cpu().numpy()
        x = x.detach().cpu().numpy().ravel()

        total_loss = total_loss + loss.item()

        loss.backward()

        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_preds.append(preds)
        xs.append(x)

        pbar.set_postfix(loss=f"{total_loss/(i+1):.6f}")

    avg_loss = total_loss / len(train_index)

    xs = np.array(xs)

    total_preds = np.concatenate(total_preds, axis=0)
    memory_file.close()
    return avg_loss, total_preds, xs

def train_x(model):

    model.train()

    total_loss, total_accuracy = 0, 0
    total_preds = []
    total_hist = []
    xs = []

    for i in tqdm(range(len(train_index)), desc="  train_x", unit="doc", leave=False):

        sent_id = get_graph(train_index[i])
        features = sent_id.x.to(device)
        edge_index = sent_id.edge_index.to(device)
        labels = train_y[i].unsqueeze(0)

        with th.no_grad():
            x, preds = model(features, edge_index)

            preds = preds.detach().cpu().numpy()
            x = x.detach().cpu().numpy().ravel()

            total_preds.append(preds)
            xs.append(x)

    xs = np.array(xs)

    total_preds = np.concatenate(total_preds, axis=0)
    return xs, total_preds

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

            x, preds = model(features, edge_index)

            loss = mse_loss(preds, labels)
            preds = preds.detach().cpu().numpy()

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
            x, preds = model(features, edge_index)

            loss = mse_loss(preds, labels)
            preds = preds.detach().cpu().numpy()
            total_loss = total_loss + loss.item()

            x = x.detach().cpu().numpy().ravel()
            total_preds.append(preds)
            total_xs.append(x)

    total_xs = np.array(total_xs)
    avg_loss = total_loss / len(test_index)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds, total_xs

train_losses=[]
valid_losses=[]
learning_rate = float(sys.argv[4])
optimizer = th.optim.Adam(net.parameters(), lr=learning_rate)
dur = []

total_epochs = int(sys.argv[5])
start_epoch = int(sys.argv[7])
end_epoch = int(sys.argv[8])

if (os.path.isfile('best-valid-losses/best_valid_loss_GATv2-dim-16x_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.txt')):
    with open('best-valid-losses/best_valid_loss_GATv2-dim-16x_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.txt') as f:
        lines = f.readlines()
    best_valid_loss = float(lines[0])
    best_epoch = int(lines[1])
    print(f"[resume] loaded prev best_valid_loss={best_valid_loss:.6f} (epoch {best_epoch})")
else:
    best_valid_loss = float('inf')
    best_epoch = 0

if (os.path.isfile('../saved-model-weights/saved_weights_GATv2-dim-16x_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.pt')):
    path = '../saved-model-weights/saved_weights_GATv2-dim-16x_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.pt'
    net.load_state_dict(th.load(path, weights_only=True))
    print(f"[resume] loaded saved weights from {path}")

print(f"\n[train] starting epochs {start_epoch} → {end_epoch}  (total={total_epochs}  lr={learning_rate})\n")
training_log = open('training-logs/training_log_GATv2-dim-16x_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.txt', 'a+')
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
        th.save(model_to_save.state_dict(), '../saved-model-weights/saved_weights_GATv2-dim-16x_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.pt')
        print(f"[epoch {epoch:03d}] ** new best model saved (valid_loss={valid_loss:.9f}) **")
    print("Target: "+bv+" | Layers {:03d}| Heads {:03d}| Epoch {:03d} | Train Loss {:.9f} | Valid Loss {:.9f}| lr {:.9f} | time {:.3f}".format(
            num_of_layers,num_of_attn_heads, epoch, train_loss, valid_loss, learning_rate, end_time-start_time))
    training_log.write("Target: "+bv+" | Layers {:03d}| Heads {:03d}| Epoch {:03d} | Train Loss {:.9f} | Valid Loss {:.9f}| lr {:.9f} | time {:.3f}\n".format(
            num_of_layers,num_of_attn_heads, epoch, train_loss, valid_loss, learning_rate, end_time-start_time))
    training_log.flush()
training_log.close()
print(f"\n[train] done. best_valid_loss={best_valid_loss:.9f} at epoch {best_epoch}")

valid_loss_file = open('best-valid-losses/best_valid_loss_GATv2-dim-16x_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.txt', 'w')
valid_loss_file.write(str(best_valid_loss)+"\n")
valid_loss_file.write(str(best_epoch))
valid_loss_file.close()

if end_epoch == total_epochs:
    print(f"\n[final] loading best model weights ...")
    model = Net(num_of_attn_heads).to(device)

    path = '../saved-model-weights/saved_weights_GATv2-dim-16x_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.pt'
    model.load_state_dict(th.load(path, weights_only=True))

    print(f"[final] running validation inference ...")
    _ , preds, xs_valid = evaluate(model)
    preds = np.asarray(preds)
    valid_y = valid_y.cpu().data.numpy()
    valid_mse = mean_squared_error(valid_y, preds)

    print(f"[final] running test inference ...")
    _, preds,xs_test= test(model,test_index)
    preds = np.asarray(preds)
    np.savetxt('mse-results/test_preds_GATv2-dim-16x_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.csv',preds, delimiter=',')
    test_y = test_y.cpu().data.numpy()
    np.savetxt('mse-results/test_trues_GATv2-dim-16x_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.csv',test_y, delimiter=',')
    test_mse = mean_squared_error(test_y, preds)

    print("GATv2-dim-16x "+str(num_of_layers)+" layers "+str(num_of_attn_heads)+" heads mse: "+str(test_mse)+'---bare---'+str(valid_mse))

    mse_file = open('mse-results/mse_GATv2-dim-16x_'+str(num_of_layers)+'_layers_'+str(num_of_attn_heads)+'_heads_'+sec+'_'+bv+'_eps'+str(total_epochs)+'_lr='+'{:.1e}'.format(learning_rate)+'_plan-'+sys.argv[6]+'_'+hist+'.txt', "w")

    mse_file.write(str(test_mse)+'---bare---'+str(valid_mse)+"\n")
    mse_file.write(str(best_valid_loss)+" after epoch: "+str(best_epoch)+"\n")

    mse_file.close()
    print(f"[final] test_mse={test_mse:.6f} | valid_mse={valid_mse:.6f}")
    print(f"[final] results saved to mse-results/")
