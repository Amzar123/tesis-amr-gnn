"""
Inference script for the trained GATv2 model (PyG version).
Counterpart to inference.py (DGL) — uses PyG .pt graph files and
weights saved by new-GATv2-4-layers-no-hist-blow-up-dim-from-768-pyg.py.

NOTE on weight compatibility
-----------------------------
Weights saved by the *original DGL* training scripts are NOT loadable here
because DGL GATv2Conv and PyG GATv2Conv use different internal parameter names.
Use weights saved by the new PyG training script:
    new-GATv2-4-layers-no-hist-blow-up-dim-from-768-pyg.py
  or
    new-GATv2-4-layers-no-hist-pyg.py   (if/when a non-blow-up variant exists)

Example usage (best model from analyze_results.py):
    python inference_pyg.py \
        --weights  ../saved-model-weights/saved_weights_GATv2-dim-16x_4_layers_4_heads_call_monthly_vol_eps5_lr=1.0e-03_plan-C_with-hist.pt \
        --graphs_dir ../amrs-to-graphs/truly-all-results-graphs-hk-finbert-plan-C \
        --csv       ../../new-tech-2019-result.csv \
        --target    monthly_vol \
        --num_heads 4 \
        --arch      blowup \
        --output    inference_output.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
)
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------

class NetBlowUp(nn.Module):
    """
    4-layer GATv2 with dimension blow-up: 768 → 1536 → 3072 → 6144 → 12288.
    Matches new-GATv2-4-layers-no-hist-blow-up-dim-from-768-pyg.py.
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_attn_heads = num_heads
        self.layer0 = GATv2Conv(768,           int(768 / num_heads * 2),           heads=num_heads, concat=True)
        self.layer1 = GATv2Conv(768 * 2,       int(768 * 2 / num_heads * 2),       heads=num_heads, concat=True)
        self.layer2 = GATv2Conv(768 * 2 * 2,   int(768 * 2 * 2 / num_heads * 2),   heads=num_heads, concat=True)
        self.layer3 = GATv2Conv(768 * 2 * 2 * 2, int(768 * 2 * 2 * 2 / num_heads * 2), heads=num_heads, concat=True)
        self.dropout = nn.Dropout(0.1)
        self.leakyrelu = nn.LeakyReLU()
        self.fc1 = nn.Linear(768 * 2 * 2 * 2 * 2, 600 * 2 * 2 * 2 * 2)
        self.fc2 = nn.Linear(600 * 2 * 2 * 2 * 2, 1)

    def forward(self, features, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=features.size(0))
        x = self.layer0(features, edge_index)
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = self.layer3(x, edge_index)
        x = x[0]
        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        y = self.fc2(x)
        return x, y


class NetRegular(nn.Module):
    """
    4-layer GATv2 with constant 768 dimension (non-blow-up).
    Equivalent to the DGL Net in inference.py / new-GATv2-4-layers-no-hist.py.
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_attn_heads = num_heads
        self.layer0 = GATv2Conv(768, int(768 / num_heads), heads=num_heads, concat=True)
        self.layer1 = GATv2Conv(768, int(768 / num_heads), heads=num_heads, concat=True)
        self.layer2 = GATv2Conv(768, int(768 / num_heads), heads=num_heads, concat=True)
        self.layer3 = GATv2Conv(768, int(768 / num_heads), heads=num_heads, concat=True)
        self.dropout = nn.Dropout(0.1)
        self.leakyrelu = nn.LeakyReLU()
        self.fc1 = nn.Linear(768, 600)
        self.fc2 = nn.Linear(600, 1)

    def forward(self, features, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=features.size(0))
        x = self.layer0(features, edge_index)
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = self.layer3(x, edge_index)
        x = x[0]
        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        y = self.fc2(x)
        return x, y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_label(value: float) -> int:
    return 1 if value >= 0 else 0


def load_graph(graphs_dir: str, ticker_date: str):
    path = os.path.join(graphs_dir, f"{ticker_date}.pt")
    return th.load(path, weights_only=False)


def run_inference(model, graphs_dir, indices, device):
    model.eval()
    all_preds = []

    with th.no_grad():
        pbar = tqdm(indices, desc="  inference", unit="doc")
        for ticker_date in pbar:
            graph = load_graph(graphs_dir, ticker_date)
            features   = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            _, pred = model(features, edge_index)
            all_preds.append(pred.detach().cpu().numpy().ravel()[0])

    return np.array(all_preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inference script for GATv2 model (PyG-compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--weights", required=True,
        help="Path to saved .pt model weights (from the PyG training script)",
    )
    parser.add_argument(
        "--graphs_dir", required=True,
        help="Directory containing PyG .pt graph files (one per document)",
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to CSV with a 'ticker_and_date' column and ground-truth target column",
    )
    parser.add_argument(
        "--target", required=True,
        help="Target column in the CSV (e.g. monthly_vol, daily_price_change)",
    )
    parser.add_argument(
        "--num_heads", required=True, type=int,
        help="Number of attention heads — must match the trained model",
    )
    parser.add_argument(
        "--arch", default="blowup", choices=["blowup", "regular"],
        help=(
            "Model architecture: "
            "'blowup' = dim blow-up 768→12288 (new PyG script), "
            "'regular' = constant 768 (non-blow-up, equivalent to DGL original)"
        ),
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save predictions CSV",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------
    if th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")
    print(f"[setup] device     : {device}")
    print(f"[setup] torch      : {th.__version__}")

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    print(f"\n[data] loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    if "ticker_and_date" not in df.columns:
        raise ValueError("CSV must contain a 'ticker_and_date' column")
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in CSV. "
                         f"Available columns: {list(df.columns)}")

    indices     = df["ticker_and_date"].tolist()
    true_values = df[args.target].tolist()
    print(f"[data] samples     : {len(indices)}")
    print(f"[data] target      : {args.target}")
    print(f"[data] graphs_dir  : {args.graphs_dir}")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    print(f"\n[model] arch       : {args.arch}")
    print(f"[model] num_heads  : {args.num_heads}")
    if args.arch == "blowup":
        model = NetBlowUp(args.num_heads).to(device)
    else:
        model = NetRegular(args.num_heads).to(device)

    print(f"[model] params     : {sum(p.numel() for p in model.parameters()):,}")
    print(f"[model] loading weights: {args.weights}")
    state = th.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"[model] weights loaded OK")

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------
    print(f"\n[inference] running on {len(indices)} documents ...")
    preds = run_inference(model, args.graphs_dir, indices, device)

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    mse = mean_squared_error(true_values, preds)
    rmse = np.sqrt(mse)
    print(f"\n{'='*55}")
    print(f"  MSE  : {mse:.8f}")
    print(f"  RMSE : {rmse:.8f}")
    print(f"  Pred std : {np.std(preds):.6f}")
    print(f"  True std : {np.std(true_values):.6f}")

    y_pred = [get_label(p) for p in preds]
    y_true = [get_label(t) for t in true_values]

    acc = accuracy_score(y_true, y_pred)
    print(f"\n  Directional Accuracy : {acc:.4f} ({int(acc*len(y_true))}/{len(y_true)})")
    print(f"{'='*55}\n")

    print("Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["True Turun (0)", "True Naik (1)"],
        columns=["Pred Turun (0)", "Pred Naik (1)"],
    )
    print(cm_df.to_string())
    print()

    print("Classification Report:")
    print(
        classification_report(
            y_true, y_pred,
            target_names=["Turun (0)", "Naik (1)"],
            labels=[0, 1],
            zero_division=0,
            digits=4,
        )
    )

    # -----------------------------------------------------------------------
    # Save output
    # -----------------------------------------------------------------------
    if args.output:
        out_df = pd.DataFrame({
            "ticker_and_date": indices,
            "prediction":      preds,
            "true_value":      true_values,
            "pred_direction":  y_pred,
            "true_direction":  y_true,
            "correct":         [int(p == t) for p, t in zip(y_pred, y_true)],
        })
        out_df.to_csv(args.output, index=False)
        print(f"[output] predictions saved to {args.output}")


if __name__ == "__main__":
    main()
