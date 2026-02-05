# import torch
# from torch_geometric.data import Data, Batch
# from torch_geometric.nn import GCNConv, global_mean_pool
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import DeterministicFakeEmbedding # Ganti dengan OpenAI/HuggingFace
# import re

import agent.pdf_parser as pdf_parser
import pandas as pd

# --- CARA PAKAI ---
agent = pdf_parser.FinancialDocAgent("AADI_2024_Annual_Report.pdf")
data = agent.process_document()

print(data)

# Output untuk dikirim ke RAG / AMR:
# combined_text = data[0]['narration'] + "\n DATA TABEL: \n" + data[0]['tables']

# --- STEP 1: GNN MODEL (PyTorch Geometric) ---
# class FinancialGNN(torch.nn.Module):
#     def __init__(self, feature_dim, hidden_dim):
#         super(FinancialGNN, self).__init__()
#         self.conv1 = GCNConv(feature_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.classifier = torch.nn.Linear(hidden_dim, 1) # Output: Importance Score

#     def forward(self, x, edge_index, batch):
#         # 1. Message Passing
#         x = self.conv1(x, edge_index).relu()
#         x = self.conv2(x, edge_index).relu()
#         # 2. Pooling (Readout layer)
#         x = global_mean_pool(x, batch) 
#         return x

# # --- STEP 2: AMR TO PYG DATA CONVERTER ---
# def amr_to_pyg_data(amr_string):
#     # Ekstraksi simpel: nyari konsep untuk jadi Nodes
#     nodes = re.findall(r'/ ([a-z0-9\-\.]+)', amr_string)
#     num_nodes = len(nodes)
    
#     # Feature matrix X (Dummy: pakai index sebagai fitur)
#     x = torch.eye(num_nodes) # One-hot encoding sederhana
    
#     # Edge Index (Dummy: menyambungkan node secara sekuensial)
#     # Di real case, gunakan library 'penman' untuk parse relasi AMR
#     edge_index = torch.tensor([[i for i in range(num_nodes-1)],
#                                [i+1 for i in range(num_nodes-1)]], dtype=torch.long)
    
#     return Data(x=x, edge_index=edge_index)

# # --- STEP 3: RAG AGENT ---
# class FinancialRAG:
#     def __init__(self, texts):
#         # Sederhananya kita buat Vector Store dari chunks teks laporan
#         self.vector_db = FAISS.from_texts(texts, DeterministicFakeEmbedding(size=1536))

#     def retrieve(self, query):
#         docs = self.vector_db.similarity_search(query, k=1)
#         return docs[0].page_content

# # --- STEP 4: RUNNING THE PIPELINE ---

# # 1. Inisialisasi Data & Model
# sample_texts = ["Laba bersih naik dikarenakan efisiensi biaya.", "Utang perusahaan menurun."]
# rag = FinancialRAG(sample_texts)
# model = FinancialGNN(feature_dim=len(sample_texts)*5, hidden_dim=16) # Contoh dim

# # 2. Ambil konteks via RAG
# context = rag.retrieve("Bagaimana kondisi laba?")

# # 3. Ubah ke AMR (Mockup string) dan masukkan ke GNN
# mock_amr = "(i / increase-01 :ARG1 (p / profit))"
# data = amr_to_pyg_data(mock_amr)
# data.batch = torch.zeros(data.x.shape[0], dtype=torch.long) # Single graph batch

# # 4. Dapatkan Graph Embedding
# with torch.no_grad():
#     graph_embedding = model(data.x, data.edge_index, data.batch)

# print("Konteks RAG:", context)
# print("Graph Embedding Shape:", graph_embedding.shape)
