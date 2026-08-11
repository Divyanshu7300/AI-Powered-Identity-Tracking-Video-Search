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
  return (
    <div className="app-surface p-5 md:p-6 space-y-6">
      <div className="flex items-center justify-between pb-4 border-b border-zinc-100">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center shrink-0">
            <SearchIcon className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-zinc-900">Search workspace</h2>
            <p className="text-xs text-zinc-400">Natural language &amp; visual Re-ID</p>
          </div>
        </div>
        <span className="text-xs font-medium text-indigo-700 bg-indigo-50 px-3 py-1 rounded-full">SigLIP Re-ID</span>
      </div>

      {/* Section 1: Text Query */}
      <form onSubmit={searchByText} className="space-y-4">
        <div>
          <label className="text-xs font-medium text-zinc-500 mb-2 block">Natural language description</label>
          <textarea
            className="w-full bg-zinc-50 border border-zinc-200/80 focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 rounded-2xl p-3.5 text-sm text-zinc-800 outline-none transition-all placeholder:text-zinc-400 min-h-[90px] leading-relaxed"
            value={textQuery}
            onChange={(e) => setTextQuery(e.target.value)}
            placeholder="e.g. person with red jacket and black pants near lower left corner"
          />
        </div>

        {/* Preset Chips */}
        <div>
          <p className="text-[11px] text-zinc-400 font-medium mb-2">Quick prompt presets</p>
          <div className="flex flex-wrap gap-1.5">
            {PRESET_QUERIES.map((preset, i) => (
              <button
                key={i}
                type="button"
                className="px-3 py-1 rounded-full bg-zinc-50 hover:bg-indigo-50 hover:text-indigo-700 text-zinc-600 text-xs font-medium transition-all border border-zinc-200/80"
                onClick={() => setTextQuery(preset)}
              >
                {preset}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-3 pt-3 border-t border-zinc-100">
          <div className="flex items-center justify-between gap-3">
            <label className="flex items-center gap-2 text-xs text-zinc-600 font-medium cursor-pointer select-none">
              <input
                type="checkbox"
                checked={clipEnabled}
                onChange={(e) => toggleClipEnabled && toggleClipEnabled(e.target.checked)}
                className="w-4 h-4 accent-indigo-500 rounded cursor-pointer"
              />
              SigLIP semantic engine
            </label>

            <button
              className="bg-zinc-900 hover:bg-zinc-800 text-white font-semibold text-sm py-2.5 px-4 rounded-xl shadow-sm transition-all flex items-center gap-1.5 disabled:opacity-50 shrink-0"
              disabled={busy === "text"}
              type="submit"
            >
              {busy === "text" ? <RefreshIcon className="w-3.5 h-3.5 animate-spin" /> : <SearchIcon className="w-3.5 h-3.5" />}
              Search text
            </button>
          </div>

          <label className="flex items-center gap-2 text-xs text-zinc-500 font-medium cursor-pointer select-none">
            <input
              type="checkbox"
              checked={useLlm}
              onChange={(e) => setUseLlm(e.target.checked)}
              className="w-4 h-4 accent-indigo-500 rounded cursor-pointer"
            />
            Enhance query &amp; explain evidence with LLM
          </label>
        </div>
      </form>

      {/* Section 2: Visual Crop Image Search */}
      <div className="border-t border-zinc-100 pt-5">
        <form onSubmit={searchByImage} className="space-y-3">
          <label className="text-xs font-medium text-zinc-500 block">Target crop photo search</label>
          <div className="flex flex-wrap gap-1.5">
            {[["face", "Face only"], ["appearance", "Dress & hair"], ["hybrid", "Face + appearance"]].map(([value, label]) => (
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
              className="flex-1 border border-dashed border-zinc-300 hover:border-indigo-300 rounded-2xl p-3 text-center cursor-pointer bg-zinc-50 hover:bg-indigo-50/40 transition-all flex items-center justify-center gap-2"
              onClick={() => imageInputRef.current?.click()}
            >
              <ImageIcon className="w-4 h-4 text-zinc-400" />
              <span className="text-xs font-medium text-zinc-600 truncate max-w-[180px]">
                {queryFile ? queryFile.name : "Select crop photo"}
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
