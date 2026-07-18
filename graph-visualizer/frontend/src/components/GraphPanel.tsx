"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import {
  Maximize2, RotateCcw, Box, Square, AlertCircle, RefreshCw,
  Cpu, GitBranch, Layers, SlidersHorizontal,
} from "lucide-react";
import clsx from "clsx";
import { GraphData, GraphNode } from "@/lib/api";

const ForceGraph2D = dynamic(
  () => import(/* webpackChunkName: "fg2d" */ "@/lib/fg2d"),
  { ssr: false }
);
const ForceGraph3D = dynamic(
  () => import(/* webpackChunkName: "fg3d" */ "@/lib/fg3d"),
  { ssr: false }
);

interface Props {
  filename: string;
  color: string;
}

type Mode = "2d" | "3d";

const NODE_PRESETS = [
  { label: "10", value: 10 },
  { label: "20", value: 20 },
  { label: "50", value: 50 },
  { label: "100", value: 100 },
  { label: "500", value: 500 },
  { label: "All", value: 1500 },
] as const;

const PALETTE = [
  "#6366f1", "#8b5cf6", "#06b6d4", "#10b981",
  "#f59e0b", "#ef4444", "#ec4899", "#84cc16",
];

function degreeColor(deg: number, maxDeg: number): string {
  if (maxDeg === 0) return PALETTE[0];
  const t = Math.min(deg / maxDeg, 1);
  if (t > 0.8) return "#ef4444";
  if (t > 0.5) return "#f59e0b";
  if (t > 0.2) return "#06b6d4";
  return "#6366f1";
}

export default function GraphPanel({ filename, color }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef    = useRef<any>(null); // eslint-disable-line @typescript-eslint/no-explicit-any
  const mousePos    = useRef({ x: 0, y: 0 });

  const [mode, setMode]         = useState<Mode>("2d");
  const [data, setData]         = useState<GraphData | null>(null);
  const [status, setStatus]     = useState<"idle" | "loading" | "error">("loading");
  const [loadMsg, setLoadMsg]   = useState("Connecting to backend…");
  const [error, setError]       = useState<string | null>(null);
  const [dims, setDims]         = useState({ w: 600, h: 500 });
  const [tooltip, setTooltip]   = useState<{ x: number; y: number; label: string } | null>(null);
  const [semantic, setSemantic] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const [maxNodes, setMaxNodes] = useState<number>(1500);

  // fetch graph data 
  useEffect(() => {
    const ctrl = new AbortController();
    setStatus("loading");
    setError(null);
    setData(null);
    setLoadMsg("Loading graph data…");

    const timer = setTimeout(() => {
      if (!ctrl.signal.aborted)
        setLoadMsg(semantic ? "Computing PCA layout…" : "Building graph…");
    }, 600);

    const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
    const params = new URLSearchParams({ max_nodes: String(maxNodes) });
    if (semantic) params.set("semantic", "true");
    const url = `${BASE}/api/files/${encodeURIComponent(filename)}/graph?${params}`;

    fetch(url, { signal: ctrl.signal })
      .then(async (res) => {
        if (!res.ok) throw new Error(`Server error ${res.status}: ${await res.text()}`);
        setLoadMsg("Rendering…");
        return res.json();
      })
      .then((d: GraphData) => { clearTimeout(timer); setData(d); setStatus("idle"); })
      .catch((e: Error) => {
        clearTimeout(timer);
        if (e.name === "AbortError") return;
        let msg = e.message;
        if (msg.includes("Load failed") || msg.includes("Failed to fetch") || msg.includes("NetworkError"))
          msg = "Cannot reach backend.\n\nRun:  uvicorn main:app --reload --port 8000";
        setError(msg);
        setStatus("error");
      });

    return () => { clearTimeout(timer); ctrl.abort(); };
  }, [filename, semantic, maxNodes, retryCount]);

  // resize observer
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setDims({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setDims({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  const handleFit = useCallback(() => graphRef.current?.zoomToFit?.(400), []);

  // whether to always show labels (small graph → always readable)
  const alwaysShowLabels = maxNodes <= 20 || (data?.stats.num_nodes ?? 9999) <= 20;

  // stable graphData
  const graphDataMemo = useMemo(() => {
    if (!data) return { nodes: [], links: [] };
    const maxDeg = Math.max(...data.nodes.map((n) => n.degree ?? 0), 1);
    return {
      nodes: data.nodes.map((n) => ({
        ...n,
        color: degreeColor(n.degree ?? 0, maxDeg),
        val: 2 + Math.sqrt(n.degree ?? 0) * 2,
      })),
      links: data.links.map((l) => ({ ...l })),
    };
  }, [data]);

  const nodeLabel = (n: any) => {
    const node = n as GraphNode;
    return `${node.label}${node.degree !== undefined ? `  (deg: ${node.degree})` : ""}`;
  };

  const onNodeHover = (node: any) => {
    if (!node) { setTooltip(null); return; }
    const n = node as GraphNode;
    setTooltip({
      x: mousePos.current.x,
      y: mousePos.current.y,
      label: `${n.label}${n.degree !== undefined ? `\ndegree: ${n.degree}` : ""}`,
    });
  };

  // Props shared between modes — only values valid in BOTH canvas and THREE.js
  const sharedProps = {
    ref: graphRef,
    graphData: graphDataMemo,
    width: dims.w,
    height: dims.h,
    linkWidth: 1,
    nodeLabel,
    onNodeHover,
    cooldownTicks: semantic ? 0 : 150,
    enableNodeDrag: true,
  };

  // 2D-only props (canvas accepts rgba / "transparent")
  const props2d = {
    ...sharedProps,
    backgroundColor: "transparent",
    linkColor: () => "rgba(148,163,184,0.3)",
  };

  // 3D-only props (THREE.js only accepts hex / rgb(r,g,b) / hsl — no rgba / "transparent")
  const props3d = {
    ...sharedProps,
    backgroundColor: "#0a0d14",       // surface-900, matches page bg
    linkColor: () => "#334155",        // slate-700 solid hex
    linkOpacity: 0.5,
    linkWidth: 0.8,
    nodeColor: (n: any) => n.color,   // hex from degreeColor()
    nodeVal: (n: any) => n.val ?? 4,
    nodeOpacity: 0.9,
    nodeResolution: 12,
  };

  const stats = data?.stats;

  return (
    <div className="flex flex-col h-full bg-surface-800 rounded-xl border border-white/5 overflow-hidden">
      <div className="flex items-center gap-2 px-3 pt-2 pb-1 border-b border-white/5 shrink-0">
        <div className="w-2 h-2 rounded-full shrink-0" style={{ background: color }} />
        <span className="text-xs font-medium text-slate-200 truncate flex-1 min-w-0">{filename}</span>

        {stats && (
          <div className="flex items-center gap-2 text-[10px] text-slate-500 shrink-0">
            {stats.sampled && (
              <span className="bg-amber-500/10 border border-amber-500/30 text-amber-400 rounded px-1.5 py-0.5">
                {stats.num_nodes.toLocaleString()}/{stats.total_nodes.toLocaleString()} nodes
              </span>
            )}
            {stats.has_labels && (
              <span className="bg-green-500/10 border border-green-500/30 text-green-400 rounded px-1.5 py-0.5">
                has labels
              </span>
            )}
            <span className="flex items-center gap-0.5"><Cpu size={9} />{stats.num_nodes.toLocaleString()}</span>
            <span className="flex items-center gap-0.5"><GitBranch size={9} />{stats.num_edges.toLocaleString()}</span>
            <span className="flex items-center gap-0.5"><Layers size={9} />{stats.feature_dim}d</span>
          </div>
        )}
      </div>

      {/* ── Header row 2: controls ── */}
      <div className="flex items-center gap-1.5 px-3 py-1.5 border-b border-white/5 shrink-0 flex-wrap">

        {/* Node count preset */}
        <div className="flex items-center gap-1 mr-1">
          <SlidersHorizontal size={10} className="text-slate-500 shrink-0" />
          <span className="text-[10px] text-slate-500 shrink-0">nodes:</span>
          <div className="flex rounded overflow-hidden border border-white/10">
            {NODE_PRESETS.map(({ label, value }) => (
              <button
                key={value}
                onClick={() => setMaxNodes(value)}
                className={clsx(
                  "px-1.5 py-0.5 text-[10px] font-medium transition-colors",
                  maxNodes === value
                    ? "bg-accent text-white"
                    : "text-slate-400 hover:text-white hover:bg-white/5"
                )}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Semantic layout toggle */}
        <button
          onClick={() => setSemantic((v) => !v)}
          title={semantic ? "Force-directed layout" : "Semantic layout (PCA of embeddings)"}
          className={clsx(
            "text-[10px] px-2 py-0.5 rounded border transition-colors shrink-0",
            semantic
              ? "bg-accent/20 border-accent/40 text-accent-hover"
              : "border-white/10 text-slate-400 hover:text-white"
          )}
        >
          {semantic ? "Semantic" : "Force"}
        </button>

        {/* 2D / 3D */}
        <div className="flex rounded-lg overflow-hidden border border-white/10 shrink-0">
          {(["2d", "3d"] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={clsx(
                "flex items-center gap-0.5 px-2 py-1 text-[11px] font-medium transition-colors",
                mode === m ? "bg-accent text-white" : "text-slate-400 hover:text-white hover:bg-white/5"
              )}
            >
              {m === "2d" ? <Square size={10} /> : <Box size={10} />}
              {m.toUpperCase()}
            </button>
          ))}
        </div>

        <button
          onClick={handleFit}
          className="p-1.5 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-colors shrink-0"
          title="Fit to screen"
        >
          <Maximize2 size={12} />
        </button>
        <button
          onClick={() => { setSemantic(false); graphRef.current?.d3ReheatSimulation?.(); }}
          className="p-1.5 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-colors shrink-0"
          title="Reset layout"
        >
          <RotateCcw size={12} />
        </button>
      </div>

      {/* ── Graph area ── */}
      <div
        ref={containerRef}
        className="flex-1 relative overflow-hidden"
        onMouseMove={(e) => { mousePos.current = { x: e.clientX, y: e.clientY }; }}
      >
        {/* Loading */}
        {status === "loading" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-5 z-10">
            <div className="relative w-24 h-24">
              <svg viewBox="0 0 96 96" className="w-full h-full">
                {[0, 60, 120, 180, 240, 300].map((deg, i) => {
                  const rad = (deg * Math.PI) / 180;
                  return (
                    <g key={i}>
                      <line x1="48" y1="48" x2={48 + 30 * Math.cos(rad)} y2={48 + 30 * Math.sin(rad)}
                        stroke="rgba(148,163,184,0.2)" strokeWidth="1" />
                      <circle cx={48 + 34 * Math.cos(rad)} cy={48 + 34 * Math.sin(rad)} r="4"
                        fill={PALETTE[i % PALETTE.length]} opacity="0.7"
                        className="animate-pulse" style={{ animationDelay: `${i * 100}ms` }} />
                    </g>
                  );
                })}
                <circle cx="48" cy="48" r="8" fill={color} />
              </svg>
              <div className="absolute inset-0 rounded-full border-2 border-transparent animate-spin"
                style={{ borderTopColor: color, animationDuration: "1.2s" }} />
            </div>
            <div className="text-center">
              <p className="text-sm font-medium text-slate-200">{loadMsg}</p>
              <p className="text-[11px] text-slate-500 mt-1 font-mono">
                {filename} · {maxNodes === 1500 ? "all" : maxNodes} nodes
              </p>
            </div>
            <div className="flex flex-col gap-1.5 text-[10px] w-44">
              {[
                { label: "Connect to backend",  done: true },
                { label: "Load & sample graph", done: loadMsg.includes("PCA") || loadMsg.includes("Build") || loadMsg.includes("Render") },
                { label: "Render",               done: loadMsg.includes("Render") },
              ].map(({ label, done }) => (
                <div key={label} className="flex items-center gap-2">
                  <div className={clsx("w-2 h-2 rounded-full shrink-0 transition-colors",
                    done ? "bg-green-500" : "bg-slate-700 animate-pulse")} />
                  <span className={done ? "text-slate-400" : "text-slate-600"}>{label}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Error */}
        {status === "error" && error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 px-8 z-10">
            <div className="w-12 h-12 rounded-full bg-danger/10 flex items-center justify-center">
              <AlertCircle size={24} className="text-danger" />
            </div>
            <div className="text-center">
              <p className="text-sm font-medium text-danger/90">Failed to load graph</p>
              <p className="text-xs text-slate-500 mt-2 whitespace-pre-line">{error}</p>
            </div>
            <button onClick={() => setRetryCount((c) => c + 1)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-slate-300 text-xs transition-colors border border-white/10">
              <RefreshCw size={12} /> Retry
            </button>
          </div>
        )}

        {/* Graph */}
        {status === "idle" && data && (
          <>
            {mode === "2d" ? (
              <ForceGraph2D
                {...props2d}
                nodeCanvasObject={(node: any, ctx, globalScale) => {
                  const r = Math.max(2, (node.val ?? 4) / globalScale);
                  if ((node.degree ?? 0) > 5) {
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, r * 2, 0, 2 * Math.PI);
                    ctx.fillStyle = node.color + "18";
                    ctx.fill();
                  }
                  ctx.beginPath();
                  ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
                  ctx.fillStyle = node.color;
                  ctx.fill();

                  const showLabel = alwaysShowLabels || globalScale > 2;
                  if (showLabel && node.label) {
                    const fs = alwaysShowLabels
                      ? Math.min(12, Math.max(9, r * 1.4))
                      : Math.max(7, 10 / globalScale);
                    ctx.font = `${fs}px sans-serif`;
                    ctx.fillStyle = "rgba(226,232,240,0.95)";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.shadowColor = "rgba(0,0,0,0.8)";
                    ctx.shadowBlur = 3;
                    ctx.fillText(node.label, node.x, node.y - r - fs * 0.7);
                    ctx.shadowBlur = 0;
                  }
                }}
                nodePointerAreaPaint={(node: any, c, ctx) => {
                  ctx.fillStyle = c;
                  ctx.beginPath();
                  ctx.arc(node.x, node.y, node.val ?? 4, 0, 2 * Math.PI);
                  ctx.fill();
                }}
              />
            ) : (
              <ForceGraph3D {...props3d} />
            )}

            {/* Sampled banner */}
            {stats?.sampled && (
              <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-surface-900/80 border border-white/10 text-slate-400 text-[10px] rounded-full px-3 py-1 pointer-events-none backdrop-blur-sm">
                Showing {stats.num_nodes.toLocaleString()} of {stats.total_nodes.toLocaleString()} nodes
                {stats.has_labels ? " · with concept labels" : " · no labels (see How-to-Run.md)"}
              </div>
            )}
          </>
        )}

        {/* Tooltip */}
        {tooltip && status === "idle" && (
          <div className="pointer-events-none fixed z-50 bg-surface-900/95 border border-white/10 rounded-lg px-2.5 py-1.5 shadow-xl backdrop-blur-sm"
            style={{ left: tooltip.x + 14, top: tooltip.y - 10 }}>
            {tooltip.label.split("\n").map((line, i) => (
              <p key={i} className={i === 0 ? "text-xs text-slate-200" : "text-[10px] text-slate-400 mt-0.5"}>
                {line}
              </p>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
