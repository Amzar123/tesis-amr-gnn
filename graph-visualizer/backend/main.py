import os
import re
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.decomposition import PCA

app = FastAPI(title="AMR Graph Visualizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # open for local dev; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_NODES_DEFAULT = 1500   # sample if graph is larger


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\s\-.]", "", name).strip()
    if not name.endswith(".pt"):
        name += ".pt"
    return name


def bfs_sample(ei: np.ndarray, total_nodes: int, n: int) -> list[int]:
    """
    Return n node IDs via BFS starting from the highest-degree node.
    This guarantees the sampled subgraph is connected and has edges.
    """
    # Build undirected adjacency list
    adj: defaultdict[int, list[int]] = defaultdict(list)
    degree_count: defaultdict[int, int] = defaultdict(int)
    for j in range(ei.shape[1]):
        s, d = int(ei[0, j]), int(ei[1, j])
        adj[s].append(d)
        adj[d].append(s)
        degree_count[s] += 1
        degree_count[d] += 1

    # seed = node with most connections
    seed = max(degree_count, key=lambda k: degree_count[k]) if degree_count else 0

    visited: list[int] = []
    seen: set[int] = set()
    queue: deque[int] = deque([seed])
    seen.add(seed)

    while queue and len(visited) < n:
        node = queue.popleft()
        visited.append(node)
        # visit neighbours sorted by degree (hubs first → denser subgraph)
        for nb in sorted(adj[node], key=lambda k: -degree_count[k]):
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)

    # if BFS couldn't reach n nodes (disconnected graph), fill with remaining
    if len(visited) < n:
        remaining = [i for i in range(total_nodes) if i not in seen]
        visited.extend(remaining[: n - len(visited)])

    return visited[:n]


def load_graph(path: Path, max_nodes: int = MAX_NODES_DEFAULT, semantic: bool = False) -> dict:
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        data = torch.load(path, map_location="cpu")

    # --- node features ---
    if hasattr(data, "x") and data.x is not None:
        x_full = data.x.float().numpy()
    else:
        n = int(data.num_nodes) if hasattr(data, "num_nodes") else 1
        x_full = np.zeros((n, 1), dtype=np.float32)

    total_nodes = x_full.shape[0]
    ei_raw = data.edge_index.numpy() if hasattr(data, "edge_index") and data.edge_index is not None else np.zeros((2, 0), dtype=np.int64)
    total_edges = int(ei_raw.shape[1])

    # --- sampling (BFS for small previews, random for large) ---
    sampled = total_nodes > max_nodes
    if sampled:
        # Use BFS when user wants a small preview (≤200) so the subgraph has edges.
        # Fall back to random for large counts (fast, edges still appear).
        if max_nodes <= 200 and total_edges > 0:
            keep_idx = bfs_sample(ei_raw, total_nodes, max_nodes)
        else:
            rng = np.random.default_rng(42)
            keep_idx = rng.choice(total_nodes, size=max_nodes, replace=False).tolist()
        keep_set = set(keep_idx)
        x = x_full[keep_idx]
        old_to_new = {old: new for new, old in enumerate(keep_idx)}
    else:
        keep_idx = list(range(total_nodes))
        keep_set = set(keep_idx)
        old_to_new = {i: i for i in range(total_nodes)}
        x = x_full

    num_nodes = x.shape[0]

    # Priority: concepts > node_labels > concept > fallback to "node_N"
    has_labels = False
    if hasattr(data, "concepts") and data.concepts is not None:
        all_labels = list(data.concepts)
        has_labels = True
    elif hasattr(data, "node_labels") and data.node_labels is not None:
        all_labels = list(data.node_labels)
        has_labels = True
    elif hasattr(data, "concept") and data.concept is not None:
        all_labels = list(data.concept)
        has_labels = True
    else:
        # Use original index so label is stable across sampling runs,
        all_labels = [str(i) for i in range(total_nodes)]
        has_labels = False

    # --- PCA positions (only when requested) ---
    if semantic and num_nodes > 1 and x.shape[1] > 1:
        n_comp = min(3, x.shape[1], num_nodes)
        pca = PCA(n_components=n_comp)
        coords = pca.fit_transform(x).astype(np.float32)
        while coords.shape[1] < 3:
            coords = np.hstack([coords, np.zeros((num_nodes, 1), dtype=np.float32)])
        for col in range(3):
            lo, hi = coords[:, col].min(), coords[:, col].max()
            if hi > lo:
                coords[:, col] = (coords[:, col] - lo) / (hi - lo) * 200 - 100
    else:
        coords = None

    # --- edges (only between kept nodes) ---
    links = []
    degree: dict[int, int] = {}   # new_id → degree count
    if hasattr(data, "edge_index") and data.edge_index is not None:
        ei = data.edge_index.numpy()
        edge_labels: list = []
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            ea = data.edge_attr
            if ea.dtype in (torch.long, torch.int32, torch.int64):
                edge_labels = ea.numpy().tolist()

        for j in range(ei.shape[1]):
            src, dst = int(ei[0, j]), int(ei[1, j])
            if src in keep_set and dst in keep_set:
                ns, nd = old_to_new[src], old_to_new[dst]
                link: dict = {"source": ns, "target": nd}
                if j < len(edge_labels):
                    link["label"] = str(edge_labels[j])
                links.append(link)
                degree[ns] = degree.get(ns, 0) + 1
                degree[nd] = degree.get(nd, 0) + 1

    # --- build node list (after edges so degree is known) ---
    nodes = []
    for new_id, old_id in enumerate(keep_idx):
        deg = degree.get(new_id, 0)
        # Concept label if available; otherwise show sequential new_id
        if has_labels:
            base_label = str(all_labels[old_id]) if old_id < len(all_labels) else str(new_id)
        else:
            base_label = str(new_id)   # clean sequential: 0, 1, 2, …
        node: dict = {
            "id": new_id,
            "label": base_label,
            "degree": deg,
        }
        if coords is not None:
            node["fx2d"] = float(coords[new_id, 0])
            node["fy2d"] = float(coords[new_id, 1])
            node["fx3d"] = float(coords[new_id, 0])
            node["fy3d"] = float(coords[new_id, 1])
            node["fz3d"] = float(coords[new_id, 2])
        nodes.append(node)

    stats = {
        "num_nodes": num_nodes,
        "num_edges": len(links),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "feature_dim": int(x.shape[1]),
        "sampled": sampled,
        "semantic": semantic,
        "has_labels": has_labels,
    }
    if hasattr(data, "y") and data.y is not None:
        stats["label"] = data.y.tolist()

    return {"nodes": nodes, "links": links, "stats": stats}


# ──────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────

@app.get("/api/files")
def list_files():
    files = []
    for f in sorted(UPLOAD_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        s = f.stat()
        files.append({"name": f.name, "size": s.st_size, "modified": s.st_mtime})
    return files


@app.post("/api/files", status_code=201)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pt"):
        raise HTTPException(400, "Only .pt files are accepted")
    dest = UPLOAD_DIR / safe_filename(file.filename)
    content = await file.read()
    dest.write_bytes(content)
    return {"name": dest.name, "size": len(content)}


@app.get("/api/files/{filename}/graph")
def get_graph(
    filename: str,
    max_nodes: int = Query(default=MAX_NODES_DEFAULT, ge=50, le=10000),
    semantic: bool = Query(default=False),
):
    path = UPLOAD_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    try:
        result = load_graph(path, max_nodes=max_nodes, semantic=semantic)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(500, f"Failed to parse graph: {e}")


class RenameBody(BaseModel):
    name: str


@app.patch("/api/files/{filename}")
def rename_file(filename: str, body: RenameBody):
    old_path = UPLOAD_DIR / filename
    if not old_path.exists():
        raise HTTPException(404, "File not found")
    new_name = safe_filename(body.name)
    new_path = UPLOAD_DIR / new_name
    if new_path.exists() and new_path != old_path:
        raise HTTPException(409, f"'{new_name}' already exists")
    old_path.rename(new_path)
    return {"name": new_name}


@app.delete("/api/files/{filename}", status_code=204)
def delete_file(filename: str):
    path = UPLOAD_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    path.unlink()
