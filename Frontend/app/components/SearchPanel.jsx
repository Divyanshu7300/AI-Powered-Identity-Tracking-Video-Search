"use client";

import React from "react";
import { ImageIcon, RefreshIcon, SearchIcon } from "./Icons";
import { PRESET_QUERIES } from "./UIHelpers";

export default function SearchPanel({
  textQuery,
  setTextQuery,
  searchByText,
  busy,
  useLlm,
  setUseLlm,
  clipEnabled,
  toggleClipEnabled,
  searchByImage,
  imageSearchMode,
  setImageSearchMode,
  queryFile,
  setQueryFile,
  imageInputRef,
}) {
  const [isDragging, setIsDragging] = React.useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setQueryFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="app-surface p-5 md:p-6 space-y-6">
      <div className="flex items-center justify-between pb-4 border-b border-zinc-100">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center shrink-0">
            <SearchIcon className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-zinc-900">Intelligence search desk</h2>
            <p className="text-xs text-zinc-400">Natural language AI query &amp; visual re-identification</p>
          </div>
        </div>
        <span className="text-xs font-medium text-indigo-700 bg-indigo-50 px-3 py-1 rounded-full">Hybrid Search</span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Natural Language Search Form */}
        <form onSubmit={searchByText} className="space-y-4">
          <div className="flex justify-between items-center">
            <label className="text-xs font-medium text-zinc-500">Natural language search</label>
            <div className="flex items-center gap-3">
              <label className="inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={useLlm}
                  onChange={(e) => setUseLlm(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-7 h-4 bg-zinc-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-zinc-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-indigo-600 relative"></div>
                <span className="ml-1.5 text-[11px] font-medium text-zinc-500">Groq LLM</span>
              </label>

              <label className="inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={clipEnabled}
                  onChange={(e) => toggleClipEnabled && toggleClipEnabled(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-7 h-4 bg-zinc-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-zinc-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-indigo-600 relative"></div>
                <span className="ml-1.5 text-[11px] font-medium text-zinc-500">CLIP Vector</span>
              </label>
            </div>
          </div>

          <div className="relative">
            <input
              className="w-full bg-zinc-50 border border-zinc-200/80 focus:border-indigo-300 focus:ring-2 focus:ring-indigo-100 rounded-2xl py-3 px-4 text-sm font-medium text-zinc-800 placeholder-zinc-400 outline-none transition-all pr-24"
              placeholder="e.g. person in red shirt carrying a backpack"
              value={textQuery}
              onChange={(e) => setTextQuery(e.target.value)}
            />
            <button
              className="absolute right-2 top-2 bottom-2 bg-zinc-900 hover:bg-zinc-800 text-white font-medium text-xs px-3.5 rounded-xl transition-all flex items-center gap-1.5 disabled:opacity-50"
              disabled={busy === "text" || !textQuery.trim()}
              type="submit"
            >
              {busy === "text" ? <RefreshIcon className="w-3.5 h-3.5 animate-spin" /> : <SearchIcon className="w-3.5 h-3.5" />}
              Search
            </button>
          </div>

          <div className="flex flex-wrap gap-1.5 pt-1">
            {PRESET_QUERIES.map((preset) => (
              <button
                key={preset}
                type="button"
                className="text-[11px] font-medium text-zinc-500 hover:text-indigo-600 bg-zinc-100 hover:bg-indigo-50 px-2.5 py-1 rounded-full transition-all border border-transparent hover:border-indigo-100"
                onClick={() => setTextQuery(preset)}
              >
                {preset}
              </button>
            ))}
          </div>
        </form>

        {/* Person Image Search Form */}
        <form onSubmit={searchByImage} className="space-y-4">
          <div className="flex justify-between items-center">
            <label className="text-xs font-medium text-zinc-500">Visual Re-ID search</label>
            <span className="text-[10px] text-zinc-400 font-mono">Appearance + Face Embedding</span>
          </div>

          <div className="flex items-center gap-1.5">
            {[
              { value: "auto", label: "Auto (Face + Body)" },
              { value: "reid", label: "Re-ID Body" },
              { value: "face", label: "Face Only" },
            ].map(({ value, label }) => (
              <button
                key={value}
                type="button"
                onClick={() => setImageSearchMode(value)}
                className={`px-3 py-1 rounded-full text-xs font-medium transition-all border ${
                  imageSearchMode === value
                    ? "bg-indigo-50 text-indigo-700 border-indigo-200"
                    : "bg-white text-zinc-500 border-zinc-200/80 hover:bg-zinc-50"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <input
            ref={imageInputRef}
            className="hidden"
            type="file"
            accept="image/*"
            onChange={(e) => setQueryFile(e.target.files?.[0] || null)}
          />
          <div className="flex items-center gap-2">
            <div
              className={`flex-1 border border-dashed rounded-2xl p-3 text-center cursor-pointer transition-all flex items-center justify-center gap-2 ${
                isDragging
                  ? "border-indigo-500 bg-indigo-50/70 scale-[1.01]"
                  : "border-zinc-300 hover:border-indigo-300 bg-zinc-50 hover:bg-indigo-50/40"
              }`}
              onClick={() => imageInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <ImageIcon className="w-4 h-4 text-zinc-400" />
              <span className="text-xs font-medium text-zinc-600 truncate max-w-[180px]">
                {queryFile ? queryFile.name : "Select crop photo or drag & drop"}
              </span>
            </div>

            <button
              className="bg-zinc-900 hover:bg-zinc-800 text-white font-medium text-sm py-3 px-4 rounded-xl shadow-sm transition-all flex items-center gap-1.5 disabled:opacity-50 shrink-0"
              disabled={busy === "image"}
              type="submit"
            >
              {busy === "image" ? <RefreshIcon className="w-3.5 h-3.5 animate-spin" /> : <SearchIcon className="w-3.5 h-3.5" />}
              Search image
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
