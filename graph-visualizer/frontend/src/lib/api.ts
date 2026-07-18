const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface FileInfo {
  name: string;
  size: number;
  modified: number;
}

export interface GraphNode {
  id: number;
  label: string;
  degree: number;
  fx2d?: number;
  fy2d?: number;
  fx3d?: number;
  fy3d?: number;
  fz3d?: number;
}

export interface GraphLink {
  source: number;
  target: number;
  label?: string;
}

export interface GraphStats {
  num_nodes: number;
  num_edges: number;
  total_nodes: number;
  total_edges: number;
  feature_dim: number;
  sampled: boolean;
  semantic: boolean;
  has_labels: boolean;
  label?: unknown;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
  stats: GraphStats;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const msg = await res.text().catch(() => res.statusText);
    throw new Error(msg);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

export const api = {
  listFiles: () => req<FileInfo[]>("/api/files"),

  uploadFile: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return req<FileInfo>("/api/files", { method: "POST", body: form });
  },

  getGraph: (filename: string, opts?: { maxNodes?: number; semantic?: boolean }) => {
    const params = new URLSearchParams();
    if (opts?.maxNodes) params.set("max_nodes", String(opts.maxNodes));
    if (opts?.semantic) params.set("semantic", "true");
    const qs = params.toString() ? `?${params}` : "";
    return req<GraphData>(`/api/files/${encodeURIComponent(filename)}/graph${qs}`);
  },

  renameFile: (filename: string, newName: string) =>
    req<{ name: string }>(`/api/files/${encodeURIComponent(filename)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: newName }),
    }),

  deleteFile: (filename: string) =>
    req<void>(`/api/files/${encodeURIComponent(filename)}`, { method: "DELETE" }),
};
