import dgl
import torch as th
import torch.nn as nn
from dgl.nn import GATv2Conv
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Net(nn.Module):
    def __init__(self, num_heads):
        super(Net, self).__init__()
        self.num_attn_heads = num_heads
        self.layer0 = GATv2Conv(768, int(768 / num_heads), num_heads=num_heads)
        self.layer1 = GATv2Conv(768, int(768 / num_heads), num_heads=num_heads)
        self.layer2 = GATv2Conv(768, int(768 / num_heads), num_heads=num_heads)
        self.layer3 = GATv2Conv(768, int(768 / num_heads), num_heads=num_heads)
        self.dropout = nn.Dropout(0.1)
        self.leakyrelu = nn.LeakyReLU()
        self.fc1 = nn.Linear(768, 600)
        self.fc2 = nn.Linear(600, 1)

    def forward(self, g, features):
        g = dgl.add_self_loop(g)
        x = self.layer0(g, features)
        x = th.cat([x[:, i, :] for i in range(self.num_attn_heads)], dim=1)
        x = self.layer1(g, x)
        x = th.cat([x[:, i, :] for i in range(self.num_attn_heads)], dim=1)
        x = self.layer2(g, x)
        x = th.cat([x[:, i, :] for i in range(self.num_attn_heads)], dim=1)
        x = self.layer3(g, x)
        x = th.cat([x[:, i, :] for i in range(self.num_attn_heads)], dim=1)
        x = x[0]
        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        y = self.fc2(x)
        del g, features
        return x, y


def get_label(x):
    """Convert regression value to binary direction label."""
    return 1 if x >= 0 else 0


def run_inference(model, graph_path, indices):
    model.eval()
    all_preds = []

    with th.no_grad():
        for i in indices:
            g = dgl.load_graphs(graph_path, [i])[0][0]
            features = g.ndata['h'].to(device)
            g = g.to(device)
            _, pred = model(g, features)
            all_preds.append(pred.detach().cpu().numpy().ravel()[0])

    return np.array(all_preds)


def main():
    parser = argparse.ArgumentParser(description="FLAG inference script")
    parser.add_argument("--weights",   required=True,  help="Path to saved .pt model weights")
    parser.add_argument("--graphs",    required=True,  help="Path to .graphs file (combined DGL graphs)")
    parser.add_argument("--csv",       required=True,  help="Path to CSV file with ground truth labels")
    parser.add_argument("--target",    required=True,  help="Target column in CSV (e.g. daily_price_change)")
    parser.add_argument("--num_heads", required=True,  type=int, help="Number of attention heads (must match trained model)")
    parser.add_argument("--output",    default=None,   help="Optional: save predictions to this CSV file")
    args = parser.parse_args()

    # Load CSV and build index list
    df = pd.read_csv(args.csv)
    indices = list(range(len(df)))
    true_labels = df[args.target].tolist()
    print(f"Loaded {len(indices)} samples from {args.csv}")

    # Load model
    model = Net(args.num_heads).to(device)
    model.load_state_dict(th.load(args.weights, map_location=device))
    model.eval()
    print(f"Loaded weights from {args.weights}")

    # Run inference
    print("Running inference...")
    preds = run_inference(model, args.graphs, indices)

    # MSE
    mse = mean_squared_error(true_labels, preds)
    print(f"\nMSE: {mse:.6f}")
    print(f"Std of predictions : {np.std(preds):.6f}")
    print(f"Std of true values : {np.std(true_labels):.6f}")

    # Directional accuracy
    y_pred = [get_label(p) for p in preds]
    y_true = [get_label(t) for t in true_labels]
    acc = accuracy_score(y_true, y_pred)
    print(f"\nDirectional Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Turun (0)", "Naik (1)"], digits=4))

    # Save predictions if requested
    if args.output:
        out_df = pd.DataFrame({
            "prediction": preds,
            "true_value": true_labels,
            "pred_direction": y_pred,
            "true_direction": y_true
        })
        out_df.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
