"use client";

import { memo, useCallback, useEffect, useRef, useState } from "react";
import { GitCompare, Share2 } from "lucide-react";
import FileList from "@/components/FileList";
import GraphPanel from "@/components/GraphPanel";
import { api, FileInfo } from "@/lib/api";

const PANEL_COLORS = ["#6366f1", "#06b6d4"];

// Memoised so it only re-renders when filename / color actually change
const MemoPanel = memo(GraphPanel, (prev, next) =>
  prev.filename === next.filename && prev.color === next.color
);

export default function Home() {
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const selectedRef = useRef(selected);
  selectedRef.current = selected;

  const refresh = useCallback(async () => {
    try {
      const incoming = await api.listFiles();

      // Only update files state if the list actually changed
      setFiles((prev) => {
        if (
          prev.length === incoming.length &&
          prev.every((f, i) => f.name === incoming[i].name && f.size === incoming[i].size)
        ) {
          return prev; // same reference → no re-render
        }
        return incoming;
      });

      // Only update selected if a selected file was deleted
      setSelected((prev) => {
        const filtered = prev.filter((n) => incoming.some((f) => f.name === n));
        if (filtered.length === prev.length) return prev; // same reference → no re-render
        return filtered;
      });
    } catch {
      // backend may not be ready yet
    }
  }, []);

  useEffect(() => {
    refresh();
    // Poll slowly — only for detecting external file changes / deletions
    const id = setInterval(refresh, 15000);
    return () => clearInterval(id);
  }, [refresh]);

  const toggleSelect = useCallback((name: string) => {
    setSelected((prev) => {
      if (prev.includes(name)) return prev.filter((n) => n !== name);
      if (prev.length >= 2) return prev;
      return [...prev, name];
    });
  }, []);

  return (
    <div className="flex h-screen overflow-hidden bg-surface-900">
      <FileList
        files={files}
        selected={selected}
        onSelect={toggleSelect}
        onRefresh={refresh}
      />

      <main className="flex-1 flex flex-col min-w-0 overflow-hidden p-4 gap-4">
        {/* Top bar */}
        <div className="flex items-center gap-3 shrink-0">
          <Share2 size={18} className="text-accent" />
          <h2 className="text-sm font-semibold text-white">
            {selected.length === 0
              ? "Graph Viewer"
              : selected.length === 1
              ? selected[0]
              : "Comparing 2 graphs"}
          </h2>
          {selected.length === 2 && (
            <span className="flex items-center gap-1 text-[10px] bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 rounded-full px-2 py-0.5">
              <GitCompare size={10} /> Compare mode
            </span>
          )}
        </div>

        {/* Content */}
        {selected.length === 0 ? (
          <EmptyState />
        ) : (
          <div className={`flex-1 grid gap-4 min-h-0 ${selected.length === 2 ? "grid-cols-2" : "grid-cols-1"}`}>
            {selected.map((name, i) => (
              <MemoPanel key={name} filename={name} color={PANEL_COLORS[i]} />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-5 text-center">
      <div className="relative">
        <div className="w-20 h-20 rounded-2xl bg-accent/10 border border-accent/20 flex items-center justify-center">
          <Share2 size={36} strokeWidth={1} className="text-accent/60" />
        </div>
        <div className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-accent animate-pulse" />
      </div>
      <div>
        <p className="text-slate-200 font-semibold text-base">No graph selected</p>
        <p className="text-slate-500 text-sm mt-1 max-w-xs">
          Upload a <code className="text-accent font-mono text-xs">.pt</code> file from the
          sidebar (or drag &amp; drop it there), then check it to visualise the AMR graph in 2D or 3D.
        </p>
      </div>
      <div className="flex gap-3 text-xs text-slate-600">
        <span className="bg-surface-700 border border-white/5 rounded-lg px-3 py-2">
          ☑ Check 1 file → single view
        </span>
        <span className="bg-surface-700 border border-white/5 rounded-lg px-3 py-2">
          ☑ Check 2 files → compare side-by-side
        </span>
      </div>
    </div>
  );
}
