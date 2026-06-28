import torch
from torch_geometric.data import Data
import pandas as pd

torch.serialization.add_safe_globals([Data])

'''
df_train = pd.read_csv('../new-tech-2010-to-2018-result.csv')

for plan in ['C','D','E']:
    parent_dir = f'truly-all-results-graphs-hk-finbert-plan-{plan}'
    graph_list = []
    for index, row in df_train.iterrows():
        print(f'{plan}-{index}')
        graph_path = parent_dir+f'/{row["ticker_and_date"]}.pt'
        graph_list.append(torch.load(graph_path, weights_only=False))
    torch.save(graph_list, f'tech-2010-to-2018-plan-{plan}.pt')
'''

df_train = pd.read_csv('../8170218/new-tech-2019-result.csv')

for plan in ['C','D','E']:
    parent_dir = f'truly-all-results-graphs-hk-finbert-plan-{plan}'
    graph_list = []
    for index, row in df_train.iterrows():
        print(f'{plan}-{index}')
        graph_path = parent_dir+f'/{row["ticker_and_date"]}.pt'
        graph_list.append(torch.load(graph_path, weights_only=False))
    torch.save(graph_list, f'tech-2019-plan-{plan}.pt')
