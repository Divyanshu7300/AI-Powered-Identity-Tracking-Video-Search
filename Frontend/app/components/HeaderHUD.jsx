"use client";

import React from "react";
import { AlertIcon, CheckIcon, CpuIcon, RefreshIcon, TrashIcon } from "./Icons";
import { ViewfinderCorners } from "./UIHelpers";

export default function HeaderHUD({
  health,
  refreshDashboard,
  busy,
  handleResetClick,
  notice,
  error,
  setNotice,
  setError,
  overview,
  semanticStatus,
  username,
  onLogout,
}) {
  return (
    <>
      {/* Header Bar */}
      <header className="flex flex-col sm:flex-row items-start sm:items-center justify-between pb-5 mb-6 border-b border-zinc-200 gap-4">
        <div className="flex items-center gap-3">
          <div className="relative w-10 h-10 rounded-xl bg-zinc-950 flex items-center justify-center">
            <span className="w-2 h-2 rounded-full bg-orange-500" />
            <ViewfinderCorners inset="inset-1" color="border-orange-500" thickness="border-[1.5px]" />
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-zinc-900 leading-tight">
              AURA <span className="text-zinc-400 font-normal">{"// AI-Powered Identity Tracking & Video Search"}</span>
            </h1>
            <p className="text-xs font-mono text-zinc-500">AI-powered identity tracking &amp; video search</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {username && (
            <div className="hidden sm:block text-right">
              <div className="font-mono text-[9px] uppercase tracking-wider text-zinc-400">Signed in as</div>
              <div className="max-w-32 truncate text-xs font-semibold text-zinc-700" title={username}>{username}</div>
            </div>
          )}
          <button
            type="button"
            onClick={onLogout}
            className="rounded-xl border border-zinc-200 bg-white px-3 py-1.5 text-xs font-semibold text-zinc-600 transition hover:border-rose-200 hover:bg-rose-50 hover:text-rose-600"
          >
            Log out
          </button>
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white border border-zinc-200 text-xs font-mono font-semibold text-zinc-700">
            <span className={`w-2 h-2 rounded-full ${health === "online" ? "bg-emerald-500 animate-pulse" : "bg-rose-500"}`} />
            <span>{health === "online" ? "Engine ready" : "Offline"}</span>
          </div>

          <button
            type="button"
            className="p-2 rounded-xl bg-white hover:bg-zinc-100 border border-zinc-200 text-zinc-700 transition-all cursor-pointer"
            title="Refresh Dashboard"
            onClick={refreshDashboard}
          >
            <RefreshIcon className="w-3.5 h-3.5" />
          </button>

          <button
            type="button"
            className="px-3 py-1.5 rounded-xl bg-white hover:bg-rose-50 border border-zinc-200 hover:border-rose-200 text-rose-600 font-semibold text-xs transition-all cursor-pointer flex items-center gap-1.5"
            title="Reset All Data"
            disabled={busy === "reset"}
            onClick={handleResetClick}
          >
            <TrashIcon className="w-3.5 h-3.5" />
            <span>Reset data</span>
          </button>
        </div>
      </header>

      {/* Global Notice Box */}
      {(notice || error) && (
        <div className={`mb-6 p-4 rounded-xl flex items-center justify-between text-xs font-medium border ${
          error ? "bg-rose-50 border-rose-200 text-rose-700" : "bg-zinc-900 text-white border-zinc-800"
        }`}>
          <div className="flex items-center gap-2">
            {error ? <AlertIcon className="w-4 h-4 flex-shrink-0 text-rose-500" /> : <CheckIcon className="w-4 h-4 flex-shrink-0 text-emerald-400" />}
            <span>{error || notice}</span>
          </div>
          <button type="button" className="opacity-70 hover:opacity-100 font-bold" onClick={() => { setNotice(""); setError(""); }}>
            ✕
          </button>
        </div>
      )}

      {/* Metric Cards Row */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white border border-zinc-200 rounded-2xl p-5 hover:border-zinc-300 transition-all duration-200">
          <div className="text-[11px] font-mono font-semibold text-zinc-400 uppercase tracking-wider mb-1.5">Indexed profiles</div>
          <div className="text-2xl font-mono font-semibold tracking-tight text-zinc-900 tabular-nums">{overview.indexed_track_memories ?? 0}</div>
        </div>
        <div className="bg-white border border-zinc-200 rounded-2xl p-5 hover:border-zinc-300 transition-all duration-200">
          <div className="text-[11px] font-mono font-semibold text-zinc-400 uppercase tracking-wider mb-1.5">Processed streams</div>
          <div className="text-2xl font-mono font-semibold tracking-tight text-zinc-900 tabular-nums">{overview.sources_processed ?? 0}</div>
        </div>
        <div className="bg-white border border-zinc-200 rounded-2xl p-5 hover:border-zinc-300 transition-all duration-200">
          <div className="text-[11px] font-mono font-semibold text-zinc-400 uppercase tracking-wider mb-1.5">Total frames</div>
          <div className="text-2xl font-mono font-semibold tracking-tight text-zinc-900 tabular-nums">{overview.frames_processed ?? 0}</div>
        </div>
        <div className="bg-white border border-zinc-200 rounded-2xl p-5 hover:border-zinc-300 transition-all duration-200">
          <div className="text-[11px] font-mono font-semibold text-zinc-400 uppercase tracking-wider mb-1.5">CLIP semantic index</div>
          <div className="text-base font-semibold text-zinc-900 flex items-center gap-1.5">
            <span className={`w-1.5 h-1.5 rounded-full ${semanticStatus.clip_ready ? "bg-emerald-500" : "bg-amber-500"}`} />
            {semanticStatus.clip_ready ? "Ready" : "Keyword mode"}
          </div>
        </div>
      </section>
    </>
  );
}
