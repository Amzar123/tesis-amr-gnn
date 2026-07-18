"use client";

import { useCallback, useRef, useState } from "react";
import { Upload, Trash2, Pencil, Check, X, FileText } from "lucide-react";
import clsx from "clsx";
import { api, FileInfo } from "@/lib/api";

interface Props {
  files: FileInfo[];
  selected: string[];
  onSelect: (name: string) => void;
  onRefresh: () => void;
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export default function FileList({ files, selected, onSelect, onRefresh }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [editingName, setEditingName] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const dragCounter = useRef(0);

  async function uploadFile(file: File) {
    if (!file.name.endsWith(".pt")) {
      setError("Only .pt files are accepted");
      return;
    }
    setUploading(true);
    setError(null);
    try {
      await api.uploadFile(file);
      onRefresh();
    } catch (err) {
      setError(String(err));
    } finally {
      setUploading(false);
    }
  }

  async function handleInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) await uploadFile(file);
    e.target.value = "";
  }

  // ── Drag-and-drop ──────────────────────────────────────────────
  const onDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current += 1;
    if (dragCounter.current === 1) setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current -= 1;
    if (dragCounter.current === 0) setIsDragging(false);
  }, []);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  }, []);

  const onDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      dragCounter.current = 0;
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) await uploadFile(file);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  // ── Rename ─────────────────────────────────────────────────────
  function startEdit(name: string) {
    setEditingName(name);
    setEditValue(name.replace(/\.pt$/, ""));
    setError(null);
  }

  async function commitEdit(oldName: string) {
    const trimmed = editValue.trim();
    if (!trimmed || trimmed + ".pt" === oldName) {
      setEditingName(null);
      return;
    }
    try {
      await api.renameFile(oldName, trimmed);
      setEditingName(null);
      onRefresh();
    } catch (err) {
      setError(String(err));
    }
  }

  async function handleDelete(name: string) {
    if (!confirm(`Delete "${name}"?`)) return;
    try {
      await api.deleteFile(name);
      onRefresh();
    } catch (err) {
      setError(String(err));
    }
  }

  const canSelect = (name: string) =>
    selected.includes(name) || selected.length < 2;

  return (
    <aside
      className="relative flex flex-col h-full bg-surface-800 border-r border-white/5 w-72 shrink-0"
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDragOver={onDragOver}
      onDrop={onDrop}
    >
      {/* Drag overlay */}
      {isDragging && (
        <div className="absolute inset-2 z-20 rounded-xl border-2 border-dashed border-accent bg-accent/10 flex flex-col items-center justify-center gap-2 pointer-events-none">
          <Upload size={32} className="text-accent animate-bounce" />
          <span className="text-sm font-medium text-accent">Drop .pt file here</span>
        </div>
      )}

      {/* Header */}
      <div className="px-4 pt-5 pb-3 border-b border-white/5 shrink-0">
        <h1 className="text-base font-semibold text-white tracking-tight">
          AMR Graph Visualizer
        </h1>
        <p className="text-xs text-slate-400 mt-0.5">Select up to 2 graphs to compare</p>
      </div>

      {/* Upload button */}
      <div className="px-4 py-3 border-b border-white/5 shrink-0">
        <input
          ref={inputRef}
          type="file"
          accept=".pt"
          className="hidden"
          onChange={handleInputChange}
        />
        <button
          onClick={() => inputRef.current?.click()}
          disabled={uploading}
          className={clsx(
            "w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
            uploading
              ? "bg-accent/40 text-white/50 cursor-not-allowed"
              : "bg-accent hover:bg-accent-hover text-white"
          )}
        >
          <Upload size={15} />
          {uploading ? "Uploading…" : "Upload .pt file"}
        </button>
        <p className="mt-1.5 text-[10px] text-slate-500 text-center">
          or drag &amp; drop anywhere in the sidebar
        </p>
        {error && (
          <p className="mt-2 text-xs text-danger bg-danger/10 rounded px-2 py-1">{error}</p>
        )}
      </div>

      {/* File list */}
      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1">
        {files.length === 0 && !isDragging && (
          <div className="flex flex-col items-center justify-center py-12 text-slate-500 text-xs text-center gap-2">
            <FileText size={28} strokeWidth={1.2} />
            <span>No files yet.<br />Upload or drag a .pt file to start.</span>
          </div>
        )}

        {files.map((f) => {
          const isSelected = selected.includes(f.name);
          const disabled = !canSelect(f.name);
          const isEditing = editingName === f.name;

          return (
            <div
              key={f.name}
              className={clsx(
                "group relative flex items-center gap-2 px-2 py-2 rounded-lg transition-colors cursor-pointer",
                isSelected
                  ? "bg-accent/20 border border-accent/40"
                  : disabled
                  ? "opacity-40 cursor-not-allowed"
                  : "hover:bg-surface-700 border border-transparent"
              )}
              onClick={() => !disabled && !isEditing && onSelect(f.name)}
            >
              {/* Checkbox */}
              <div
                className={clsx(
                  "w-4 h-4 shrink-0 rounded border-2 flex items-center justify-center transition-colors",
                  isSelected ? "bg-accent border-accent" : "border-slate-500"
                )}
              >
                {isSelected && <Check size={10} strokeWidth={3} className="text-white" />}
              </div>

              {/* Name / edit */}
              <div className="flex-1 min-w-0" onClick={(e) => e.stopPropagation()}>
                {isEditing ? (
                  <div className="flex items-center gap-1">
                    <input
                      autoFocus
                      className="flex-1 min-w-0 bg-surface-900 border border-accent/60 rounded px-1.5 py-0.5 text-xs text-white outline-none"
                      value={editValue}
                      onChange={(e) => setEditValue(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") commitEdit(f.name);
                        if (e.key === "Escape") setEditingName(null);
                      }}
                    />
                    <button
                      onClick={() => commitEdit(f.name)}
                      className="text-green-400 hover:text-green-300 shrink-0"
                    >
                      <Check size={13} />
                    </button>
                    <button
                      onClick={() => setEditingName(null)}
                      className="text-slate-400 hover:text-white shrink-0"
                    >
                      <X size={13} />
                    </button>
                  </div>
                ) : (
                  <>
                    <p className="text-xs font-medium text-slate-200 truncate">{f.name}</p>
                    <p className="text-[10px] text-slate-500">{formatBytes(f.size)}</p>
                  </>
                )}
              </div>

              {/* Actions */}
              {!isEditing && (
                <div
                  className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
                  onClick={(e) => e.stopPropagation()}
                >
                  <button
                    onClick={() => startEdit(f.name)}
                    className="p-1 rounded hover:bg-white/10 text-slate-400 hover:text-white transition-colors"
                    title="Rename"
                  >
                    <Pencil size={12} />
                  </button>
                  <button
                    onClick={() => handleDelete(f.name)}
                    className="p-1 rounded hover:bg-danger/20 text-slate-400 hover:text-danger transition-colors"
                    title="Delete"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-white/5 text-[10px] text-slate-500 shrink-0">
        {files.length} file{files.length !== 1 ? "s" : ""} · {selected.length}/2 selected
      </div>
    </aside>
  );
}
