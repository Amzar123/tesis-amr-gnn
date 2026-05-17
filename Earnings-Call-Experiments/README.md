# FLAG — Earnings Call Experiments

**FLAG**: Financial Long document classification via AMR-based GNNs
Paper: *"Semantic Graph Learning for Trend Prediction from Long Financial Documents"*

Predicts stock price changes (daily, weekly, monthly) from earnings call transcripts by converting text into AMR-based graphs and training a GATv2 Graph Neural Network.

---

## Table of Contents

- [Requirements](#requirements)
- [Data Setup](#data-setup)
- [Pipeline Overview](#pipeline-overview)
- [Step 1: Build Graphs from AMR Files](#step-1-build-graphs-from-amr-files)
- [Step 2: Combine Graphs](#step-2-combine-graphs)
- [Step 3: Train the GATv2 Model](#step-3-train-the-gatv2-model)
- [Step 4: Evaluate Accuracy](#step-4-evaluate-accuracy)
- [Skipping to Inference with Pre-trained Weights](#skipping-to-inference-with-pre-trained-weights)
- [Output Files Reference](#output-files-reference)

---

## Requirements

Install all Python dependencies:

```bash
pip install -r ../requirements.txt
```

For DGL with CUDA support (recommended):

```bash
# CUDA 11.8 example — see https://www.dgl.ai/pages/start.html for your version
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
```

> **GPU note:** All scripts default to `device = torch.device("cuda")`. If you do not have a GPU, change this to `"cpu"` in each script before running. Training on CPU is very slow.

---

## Data Setup

You need three external datasets, all available on Zenodo:

### 1. CSV Dataset Files
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8170218.svg)](https://doi.org/10.5281/zenodo.8170218)

Download and place the following files directly in this directory (`earnings-call-transcripts/`):
```
earnings-call-transcripts/
├── new-tech-2010-to-2018-result.csv   ← training data (2010–2018)
└── new-tech-2019-result.csv           ← test data (2019)
```

### 2. AMR Files
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8188443.svg)](https://doi.org/10.5281/zenodo.8188443)

Download and extract into this directory structure:
```
earnings-call-transcripts/
├── punkt-truly-all-amrs/              ← directory of .txt AMR files
│   ├── AAPL_2010-01-01.txt
│   └── ...
└── truly-all-amrs-file-names.txt      ← ordered list of AMR filenames (one per line)
```

### 3. FinBERT Pretrained Model

The graph-building scripts require FinBERT. Download from HuggingFace:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('ProsusAI/finbert', local_dir='./finbert-pretrain')"
```

Then update the `bert_path` variable in every `amrs-to-graphs/amrs-to-graphs-plan-*.py` file:

```python
# Change this line:
bert_path = "/gpfs/u/home/DLTM/DLTMboxi/scratch/env/finbert-pretrain/"
# To your local path:
bert_path = "../../finbert-pretrain/"   # or an absolute path
```

---

## Pipeline Overview

```
AMR text files  +  FinBERT
        │
        ▼
[Step 1] amrs-to-graphs-plan-{C/D/E}-hk-finbert.py
        │   produces: truly-all-results-graphs-hk-finbert-plan-{C/D/E}/*.graph
        ▼
[Step 2] combine-tech-graphs.py
        │   produces: tech-2010-to-2018-plan-{C/D/E}.graphs
        │              tech-2019-plan-{C/D/E}.graphs
        ▼
[Step 3] tech-gnn/GATv2/new-GATv2-4-layers-no-hist.py
        │   produces: saved model weights, MSE results, training logs
        ▼
[Step 4] tech-gnn/GATv2/get-accuracy.py
             produces: accuracy / classification report
```

Plans **C**, **D**, and **E** refer to different AMR graph construction strategies (how nodes and edges from the AMR parse are connected). Plan C is the default and most commonly used.

---

## Step 1: Build Graphs from AMR Files

**Directory:** `amrs-to-graphs/`

This step reads AMR-parsed text files, encodes each concept node with FinBERT embeddings (768-dim), and saves a DGL graph for each document.

```bash
cd amrs-to-graphs/

python amrs-to-graphs-plan-C-hk-finbert.py <start_id> <end_id>
```

| Argument | Description |
|----------|-------------|
| `start_id` | Index of first document to process (inclusive) |
| `end_id`   | Index of last document to process (exclusive) |

**Example** — process the first 100 documents:
```bash
python amrs-to-graphs-plan-C-hk-finbert.py 0 100
```

To find the total number of documents:
```bash
wc -l ../truly-all-amrs-file-names.txt
```

You can split the range across multiple runs (or machines) for large datasets:
```bash
python amrs-to-graphs-plan-C-hk-finbert.py 0    500
python amrs-to-graphs-plan-C-hk-finbert.py 500  1000
python amrs-to-graphs-plan-C-hk-finbert.py 1000 1500
```

**Output:** Individual `.graph` files saved to:
```
amrs-to-graphs/truly-all-results-graphs-hk-finbert-plan-C/
```

Repeat for plans D and E if needed:
```bash
python amrs-to-graphs-plan-D-hk-finbert.py 0 <total>
python amrs-to-graphs-plan-E-hk-finbert.py 0 <total>
```

---

## Step 2: Combine Graphs

**Directory:** `amrs-to-graphs/`

Merges all individual `.graph` files into two combined `.graphs` files (train set and test set) for each plan.

```bash
cd amrs-to-graphs/

python combine-tech-graphs.py
```

This reads `new-tech-2010-to-2018-result.csv` and `new-tech-2019-result.csv` to determine the ordering of documents, then saves:

```
amrs-to-graphs/
├── tech-2010-to-2018-plan-C.graphs   ← training graphs
├── tech-2019-plan-C.graphs           ← test graphs
├── tech-2010-to-2018-plan-D.graphs
├── tech-2019-plan-D.graphs
├── tech-2010-to-2018-plan-E.graphs
└── tech-2019-plan-E.graphs
```

> If you only built graphs for plan C, comment out the D and E loops in `combine-tech-graphs.py` before running.

---

## Step 3: Train the GATv2 Model

**Directory:** `tech-gnn/GATv2/`

```bash
cd tech-gnn/GATv2/

python new-GATv2-4-layers-no-hist.py \
  <sec> <bv> <hist> <lr> <total_epochs> <plan> <start_epoch> <end_epoch> <num_heads>
```

### Arguments

| Position | Argument | Description | Example Values |
|----------|----------|-------------|----------------|
| 1 | `sec` | Dataset identifier | `tech-earnings-calls` |
| 2 | `bv` | Target variable (what to predict) | `daily_price_change`, `weekly_avg_price_change`, `monthly_avg_price_change`, `monthly_vol_change` |
| 3 | `hist` | History flag | `no-hist` |
| 4 | `lr` | Learning rate | `9e-5`, `3e-5`, `1e-5` |
| 5 | `total_epochs` | Total number of epochs | `10` |
| 6 | `plan` | Graph construction plan | `C`, `D`, `E` |
| 7 | `start_epoch` | Epoch to start from (for resuming) | `0` |
| 8 | `end_epoch` | Epoch to train until | `10` |
| 9 | `num_heads` | Number of attention heads | `4`, `8`, `12` |

### Example — Quick test run (5 epochs, plan C, 4 heads)
```bash
python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-5 5 C 0 5 4
```

### Example — Full run (10 epochs, plan C, 4 heads)
```bash
python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-5 10 C 0 10 4
```

### Example — Resume training (continue from epoch 5 to 10)
```bash
python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-5 10 C 5 10 4
```

### Training outputs (saved automatically)

```
tech-gnn/GATv2/
├── best-valid-losses/    ← best validation loss + epoch per run
├── mse-results/          ← test MSE + predictions CSV after final epoch
├── saved-model-weights/  ← model checkpoint (.pt file)
├── tracking-files/       ← per-epoch memory usage logs
└── training-logs/        ← epoch-by-epoch train/val loss
```

---

## Step 4: Evaluate Accuracy

After training completes, evaluate directional accuracy (up/down classification):

```bash
cd tech-gnn/GATv2/

python get-accuracy.py
```

> The script is currently configured for specific result files. Edit the `np.loadtxt(...)` path at the bottom of the file to point to your run's prediction CSV in `mse-results/`.

**Example edit in `get-accuracy.py`:**
```python
test_preds = np.loadtxt(
    'mse-results/test_preds_GATv2_4_layers_4_heads_tech-earnings-calls_daily_price_change_eps10_lr=9.0e-05_plan-C_no-hist.csv',
    delimiter=",", dtype=float
)
```

---

## Skipping to Inference with Pre-trained Weights

If you have the pre-trained weights from Zenodo, you can skip Steps 1–3 and run evaluation directly.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8170309.svg)](https://doi.org/10.5281/zenodo.8170309)

1. Download weights and place in `tech-gnn/GATv2/saved-model-weights/`
2. You still need the CSV test file (`new-tech-2019-result.csv`) and the combined `.graphs` test file
3. Modify the training script to load the model and call `test()` directly, or use `get-accuracy.py`

---

## Output Files Reference

| File pattern | Description |
|---|---|
| `mse-results/mse_GATv2_*.txt` | Test MSE and best validation loss |
| `mse-results/test_preds_GATv2_*.csv` | Raw model predictions on test set |
| `mse-results/test_trues_GATv2_*.csv` | Ground truth labels for test set |
| `best-valid-losses/best_valid_loss_*.txt` | Best validation loss and epoch number |
| `saved-model-weights/saved_weights_*.pt` | PyTorch model checkpoint |
| `training-logs/training_log_*.txt` | Per-epoch train/val loss history |
