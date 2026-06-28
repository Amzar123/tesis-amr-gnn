import torch
from torch_geometric.data import Data
import torch_geometric.utils as pg_utils
#from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import BertTokenizer, BertModel
from tokenizers.pre_tokenizers import CharDelimiterSplit
import re
import sys
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Otomatis pakai CPU jika di Mac tanpa CUDA

# bert_path = "/gpfs/u/home/DLTM/DLTMboxi/scratch/env/finbert-pretrain/"

bert_path = "ProsusAI/finbert"

#config = AutoConfig.from_pretrained(bert_path, output_hidden_states = True)

model = BertModel.from_pretrained(bert_path,output_hidden_states = True)
tokenizer = BertTokenizer.from_pretrained(bert_path)
tokenizer.pre_tokenizer = CharDelimiterSplit(' ')

model = model.to(device)
model.eval()

class Node:
    def __init__(self,word,entity,embedding,doc_id):
        self.word = word
        self.entity = entity
        self.embedding = embedding
        self.doc_id = doc_id
    def __str__(self):
        return self.entity+"; "+self.word+";"+str(self.doc_id)

def sentence_to_bert_embeddings(sentence):
    # Add the special tokens.
    marked_text = "[CLS] " + sentence + " [SEP]"
    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)

    if len(tokenized_text) > 512:
        track_file.write("more than 512!\n")
        track_file.write("len: "+str(len(tokenized_text)))
        track_file.flush()
        tokenized_text = tokenizer.tokenize(sentence)
        tokens = tokenizer.encode_plus(tokenized_text, add_special_tokens=False, return_tensors='pt')
        input_id_chunks = list(tokens['input_ids'][0].split(510))
        mask_chunks = list(tokens['attention_mask'][0].split(510))
        tokenized_text_list = [['[CLS]']+tokenized_text[:510]+['[SEP]'],['[CLS]']+tokenized_text[510:]+['[SEP]']+(510-len(tokenized_text[510:]))*['[PAD]']]

        for i in range(len(input_id_chunks)):
            cls_tok = torch.Tensor([tokenizer.cls_token_id])
            sep_tok = torch.Tensor([tokenizer.sep_token_id])
            mask_tok = torch.Tensor([1])
            input_id_chunks[i] = torch.cat([cls_tok, input_id_chunks[i], sep_tok])
            mask_chunks[i] = torch.cat((mask_tok, mask_chunks[i], mask_tok))
            pad_len = 512 - input_id_chunks[i].shape[0]
            if pad_len > 0:
                input_id_chunks[i] = torch.cat((input_id_chunks[i], torch.Tensor([tokenizer.pad_token_id] * pad_len)))
                mask_chunks[i] = torch.cat((mask_chunks[i], torch.Tensor([tokenizer.pad_token_id] * pad_len)))

        input_ids = torch.stack(input_id_chunks).to(device)
        attention_mask = torch.stack(mask_chunks).to(device)

        input_dict = {
            'input_ids': input_ids.long(),
            'attention_mask': attention_mask.int()
        }

        model.eval()
        with torch.no_grad():
            outputs = model(**input_dict)
            hidden_states = outputs[2]

        token_embeddings = torch.stack(hidden_states, dim=0)
        del outputs
        del hidden_states

        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,2,0,3)

        final_results = []
        for i in range(token_embeddings.shape[0]):
            token_vecs_sum = []
            for token in token_embeddings[i]:
                sum_vec = torch.sum(token[-4:], dim=0)
                token_vecs_sum.append(sum_vec)
            count = 0
            results = []
            previous_vector = None
            vectors_to_mean = []
            while(True):
                if count == len(token_vecs_sum):
                    break
                current_vector = token_vecs_sum[count].unsqueeze(0)
                current_word_token = tokenized_text_list[i][count]
                if current_word_token[:2] == "##":
                    vectors_to_mean.append(previous_vector)
                    vectors_to_mean.append(current_vector)
                    increment = 1
                    while(True):
                        next_word_token = tokenized_text_list[i][count+increment]
                        next_vector = token_vecs_sum[count+increment].unsqueeze(0)
                        if next_word_token[:2] == "##":
                            vectors_to_mean.append(next_vector)
                            increment += 1
                            continue
                        else:
                            count += increment
                            break
                    vectors_to_mean = torch.cat(vectors_to_mean,0)
                    current_vector = torch.mean(vectors_to_mean,0).unsqueeze(0)
                    vectors_to_mean = []
                    results.pop()
                    results.append(current_vector)
                    continue
                results.append(current_vector)
                previous_vector = current_vector
                count += 1
            results = torch.cat(results,0)
            final_results.append(results)

        del input_dict
        del input_ids
        del attention_mask

        final_result = torch.cat((final_results[0][:-1],final_results[1][1:]))[:len(sentence.split())+2]
        return final_result
    else:
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens]).to(device)
        segments_tensors = torch.tensor([segments_ids]).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]

        token_embeddings = torch.stack(hidden_states, dim=0)
        del outputs
        del hidden_states

        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)

        token_vecs_sum = []
        for token in token_embeddings:
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vecs_sum.append(sum_vec)

        count = 0
        results = []
        previous_vector = None
        vectors_to_mean = []
        while(True):
            current_vector = token_vecs_sum[count].unsqueeze(0)
            current_word_token = tokenized_text[count]
            if current_word_token[:2] == "##":
                vectors_to_mean.append(previous_vector)
                vectors_to_mean.append(current_vector)
                increment = 1
                while(True):
                    next_word_token = tokenized_text[count+increment]
                    next_vector = token_vecs_sum[count+increment].unsqueeze(0)
                    if next_word_token[:2] == "##":
                        vectors_to_mean.append(next_vector)
                        increment += 1
                        continue
                    else:
                        count += increment
                        break
                vectors_to_mean = torch.cat(vectors_to_mean,0)
                current_vector = torch.mean(vectors_to_mean,0).unsqueeze(0)
                vectors_to_mean = []
                results.pop()
                results.append(current_vector)
                continue

            results.append(current_vector)
            previous_vector = current_vector
            count += 1
            if count == len(token_vecs_sum):
                break
        results = torch.cat(results,0)
        tokens_tensor.detach().cpu()
        segments_tensors.detach().cpu()
        del tokens_tensor
        del segments_tensors
        return results

sec_fname = "../punkt-truly-all-amrs/"
start_id = int(sys.argv[1])
end_id = int(sys.argv[2])
try:
    os.mkdir('truly-all-results-graphs-hk-finbert-plan-D')
except OSError as error:
    print(error)

track_file = open('tracking_file_all-results_construct_hk-finbert_plan_D_'+str(start_id)+"-"+str(end_id)+".txt",'a+')
with open('truly-all-amrs-file-names.txt','r') as f:
    filenames = f.readlines()

for i in range(start_id, end_id):
    filename = filenames[i].strip()
    track_file.write("i = "+str(i)+'\n')
    track_file.write("doc num: "+str(i)+" before train: "+str(int(torch.cuda.memory_reserved()/1024/1024))+' mem reserved\n')
    track_file.write("doc num: "+str(i)+" before train: "+str(int(torch.cuda.memory_allocated()/1024/1024))+' mem allocated\n')

    fname = sec_fname+filename
    ffile = open(fname,'r')
    flines = ffile.readlines()

    current_snt = ''
    current_snt_embeddings = None
    current_snt_cls = None
    previous_snt_cls = None
    current_snt_nodes_dict = {}

    count = 0
    doc_nodes_list = []
    node_index = 0
    doc_nodes_list.append(Node("doc_cls","doc_cls",torch.zeros(768),node_index))
    node_index += 1

    edge_list1 = []
    edge_list2 = []
    while(True):
        line = flines[count]
        if line[:7] == "# ::tok":
            # edges from the PREVIOUS snt_cls to each node in PREVIOUS snt happen here
            # also edge from doc_cls to PREVIOUS snt_cls and every node in PREVIOUS snt happens here
            if current_snt_cls is not None:
                for key in current_snt_nodes_dict:
                    # connect PREVIOUS snt_cls to a node in PREVIOUS snt
                    n1 = current_snt_cls.doc_id
                    n2 = current_snt_nodes_dict[key].doc_id
                    edge_list1.append(n1)
                    edge_list2.append(n2)
                    # connect doc_cls to a node in PREVIOUS snt
                    n1 = 0
                    n2 = current_snt_nodes_dict[key].doc_id
                    edge_list1.append(n1)
                    edge_list2.append(n2)
                n1 = 0
                n2 = current_snt_cls.doc_id
                edge_list1.append(n1)
                edge_list2.append(n2)

            previous_snt_cls = current_snt_cls
            current_snt_nodes_dict.clear()

            current_snt = re.sub(r'(?<=\S)[^a-zA-Z0-9\s]','',line[7:])
            current_snt = re.sub(r'[^a-zA-Z0-9\s](?=\S)','',current_snt)
            current_snt_embeddings = sentence_to_bert_embeddings(current_snt)[1:-1].cpu()
            current_snt = current_snt.split()
            assert len(current_snt) == current_snt_embeddings.shape[0]
            current_snt_cls = Node("snt_cls","snt_cls",torch.zeros(768),node_index)
            node_index += 1
            doc_nodes_list.append(current_snt_cls)
            # connect previous snt_cls to current snt_cls
            if previous_snt_cls is not None:
                n1 = previous_snt_cls.doc_id
                n2 = current_snt_cls.doc_id
                edge_list1.append(n1)
                edge_list2.append(n2)
            count += 1
            continue
        elif line[:8] == "# ::node":
            node_info_list = line[8:].split()
            snt_node_id = node_info_list[0]
            word_index = int(node_info_list[-1].split('-')[0])
            word = current_snt[word_index]
            entity = node_info_list[1]
            embedding = current_snt_embeddings[word_index]
            node = Node(word, entity, embedding, node_index)
            node_index += 1
            doc_nodes_list.append(node)
            current_snt_nodes_dict[snt_node_id] = node
            count += 1
            continue
        elif line[:8] == "# ::edge":
            edge_info_list = line[8:].split()
            key1 = edge_info_list[-2]
            key2 = edge_info_list[-1]
            n1 = current_snt_nodes_dict[key1].doc_id
            n2 = current_snt_nodes_dict[key2].doc_id
            edge_list1.append(n1)
            edge_list2.append(n2)
            count += 1
            continue

        count += 1
        if count == len(flines):
            for key in current_snt_nodes_dict:
                # connect PREVIOUS snt_cls to a node in PREVIOUS snt
                n1 = current_snt_cls.doc_id
                n2 = current_snt_nodes_dict[key].doc_id
                edge_list1.append(n1)
                edge_list2.append(n2)
                # connect doc_cls to a node in PREVIOUS snt
                n1 = 0
                n2 = current_snt_nodes_dict[key].doc_id
                edge_list1.append(n1)
                edge_list2.append(n2)
            n1 = 0
            n2 = current_snt_cls.doc_id
            edge_list1.append(n1)
            edge_list2.append(n2)
            break

    num_of_nodes = len(doc_nodes_list)

    # --- PROSES MIGRASI DGL KE PYTORCH GEOMETRIC (PyG) ---

    # 1. Di PyG, edges direpresentasikan sebagai tensor [2, jumlah_edges] bertipe long
    edge_index = torch.tensor([edge_list1, edge_list2], dtype=torch.long)

    # 2. Mengumpulkan semua node embedding menjadi satu tensor node features `x`
    node_features = torch.zeros((num_of_nodes, 768))
    for l in range(num_of_nodes):
        node_features[l] = doc_nodes_list[l].embedding.detach().cpu()

    del doc_nodes_list

    # 3. Membuat objek Data Graph PyG (di-set langsung di CPU)
    g = Data(x=node_features, edge_index=edge_index)

    # 4. Mengubah graph menjadi Bidirectional (pengganti dgl.to_bidirected)
    g.edge_index = pg_utils.to_undirected(g.edge_index, num_nodes=num_of_nodes)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Menyimpan graph PyG (PyG menggunakan torch.save bawaan PyTorch untuk menyimpan data)
    graph_save_path = "truly-all-results-graphs-hk-finbert-plan-D/" + filename[:-3] + "pt"
    torch.save(g, graph_save_path)

    track_file.write("doc num: "+str(i)+" after train: "+str(int(torch.cuda.memory_reserved()/1024/1024))+' mem reserved\n')
    track_file.write("doc num: "+str(i)+" after train: "+str(int(torch.cuda.memory_allocated()/1024/1024))+' mem allocated\n')
    track_file.flush()

track_file.close()
