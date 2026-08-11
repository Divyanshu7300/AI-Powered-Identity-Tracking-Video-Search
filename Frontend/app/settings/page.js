"use client";

import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@clerk/nextjs";
import { useRouter } from "next/navigation";
import { apiRequest, fetchCurrentIdentity, getUsername, loadDashboard as fetchDashboard } from "../lib/api";
import ResetConfirmationModal from "../components/ResetConfirmationModal";
import { SettingsIcon, TrashIcon, RefreshIcon, SparklesIcon, AlertIcon, CheckIcon } from "../components/Icons";

export default function SettingsPage() {
  const router = useRouter();
  const { isLoaded, isSignedIn } = useAuth();
  const [nativeUser, setNativeUser] = useState("");

  const [showResetModal, setShowResetModal] = useState(false);
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  React.useEffect(() => {
    fetchCurrentIdentity().then((identity) => {
      if (identity?.username) setNativeUser(identity.username);
    });
  }, []);

  const effectiveIsSignedIn = Boolean(isSignedIn || nativeUser || getUsername());

  const { data: dashboardData, error: dashboardError, refetch: refreshDashboard } = useQuery({
    queryKey: ["dashboard"],
    queryFn: fetchDashboard,
    staleTime: 2_000,
    refetchInterval: 5_000,
    enabled: effectiveIsSignedIn,
  });

  const { data: searchSettingsData, refetch: refreshSettings } = useQuery({
    queryKey: ["searchSettings"],
    queryFn: () => apiRequest("/api/tracking/search/settings"),
    enabled: effectiveIsSignedIn,
  });

  const semanticStatus = searchSettingsData || dashboardData?.dashboard?.semantic_status || {};
  const overview = dashboardData?.dashboard?.overview || {};

  async function toggleClipEnabled(enabled) {
    setBusy("clip");
    setError("");
    setNotice("");
    try {
      await apiRequest("/api/tracking/search/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clip_enabled: Boolean(enabled) }),
      });
      setNotice(enabled ? "SigLIP / CLIP vision search enabled." : "CLIP disabled — dual-zone attribute search active.");
      await refreshSettings();
      await refreshDashboard();
    } catch (err) {
      console.error(err);
      setError(typeof err === "string" ? err : err.message || "Failed to update search engine settings.");
    } finally {
      setBusy("");
    }
  }

  async function reindexSemanticSearch() {
    setBusy("reindex");
    setError("");
    setNotice("");
    try {
      const payload = await apiRequest("/api/tracking/search/reindex", { method: "POST" });
      setNotice(`Search index refreshed: ${payload.refreshed || 0} representative frames updated.`);
      await refreshSettings();
      await refreshDashboard();
    } catch (err) {
      console.error(err);
      setError(typeof err === "string" ? err : err.message || "Unable to refresh search index.");
    } finally {
      setBusy("");
    }
  }

  async function confirmResetData() {
    setShowResetModal(false);
    setBusy("reset");
    setError("");
    setNotice("");
    try {
      await apiRequest("/api/tracking/reset", { method: "POST" });
      setNotice("All past video archives, track memories, clips, and search history reset successfully.");
      await refreshSettings();
      await refreshDashboard();
    } catch (err) {
      console.error(err);
      setError(typeof err === "string" ? err : err.message || "Failed to reset session data.");
    } finally {
      setBusy("");
    }
  }

  const requiresLogin = isLoaded && (!effectiveIsSignedIn || /login|token/i.test(dashboardError?.message || ""));
  React.useEffect(() => {
    if (requiresLogin) router.replace("/login");
  }, [requiresLogin, router]);

  if (requiresLogin) return null;

  const isClipEnabled = semanticStatus?.clip_enabled !== false;
  const isAlertError = Boolean(error);

  return (
    <div className="relative app-page w-full animate-fade-in">
      {/* full-bleed ambient wash — neutral slate accent for settings */}
      <div className="pointer-events-none absolute -top-40 -right-20 w-[600px] h-[600px] rounded-full bg-gradient-to-br from-zinc-200/60 via-slate-200/35 to-transparent blur-3xl opacity-70" />

      <div className="relative app-container space-y-6">
        {/* Header */}
        <header className="app-page-header">
          <div>
            <h1 className="app-page-title">Settings</h1>
            <p className="app-page-description">
              Manage your search engine configuration and workspace data.
            </p>
          </div>
          <button
            type="button"
            onClick={() => { refreshSettings(); refreshDashboard(); }}
            className="shrink-0 px-3.5 py-2 rounded-xl bg-white border border-zinc-200/80 text-sm font-medium text-zinc-700 hover:border-zinc-300 hover:shadow-sm flex items-center gap-2 transition-all cursor-pointer"
          >
            <RefreshIcon className="w-3.5 h-3.5" />
            Refresh status
          </button>
        </header>

        {/* Global Alerts */}
        {(notice || error) && (
          <div
            className={`p-4 rounded-2xl flex items-center justify-between gap-3 text-sm border shadow-sm ${
              isAlertError ? "bg-rose-50 border-rose-100 text-rose-700" : "bg-white border-zinc-200/80 text-zinc-700"
            }`}
          >
            <div className="flex items-center gap-2.5 min-w-0">
              {isAlertError ? (
                <AlertIcon className="w-4 h-4 text-rose-500 shrink-0" />
              ) : (
                <CheckIcon className="w-4 h-4 text-emerald-500 shrink-0" />
              )}
              <span className="font-medium truncate">{error || notice}</span>
            </div>
            <button
              type="button"
              onClick={() => { setError(""); setNotice(""); }}
              className="shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-zinc-400 hover:text-zinc-700 hover:bg-zinc-100 transition-colors"
            >
              ✕
            </button>
          </div>
        )}

        {/* Overview Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="app-surface p-5">
            <p className="text-xs font-medium text-zinc-400 mb-2">Semantic vision model</p>
            <p className="text-sm font-semibold text-zinc-900 truncate">
              {semanticStatus?.model_name || "google/siglip-base-patch16-224"}
            </p>
            <p className="text-xs text-zinc-400 mt-1.5">Zero-shot Re-ID embedding</p>
          </div>

          <div className="app-surface p-5">
            <p className="text-xs font-medium text-zinc-400 mb-2">Indexed keyframes</p>
            <p className="text-2xl font-semibold text-zinc-900 tracking-tight tabular-nums">
              {semanticStatus?.indexed_frames ?? overview.indexed_track_memories ?? 0}
            </p>
            <p className="text-xs text-zinc-400 mt-1.5">Stored in vector index</p>
          </div>

          <div className="app-surface p-5">
            <p className="text-xs font-medium text-zinc-400 mb-2">Engine status</p>
            <p className="text-sm font-semibold text-emerald-600 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              FastAPI operational
            </p>
            <p className="text-xs text-zinc-400 mt-1.5">Backend heartbeat ready</p>
          </div>
        </div>

        {/* Main Settings Panel */}
        <div className="space-y-4">
          {/* Card 1: SigLIP / CLIP Search Toggle */}
          <div className="app-surface p-5 md:p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-zinc-100 pb-4">
              <div className="flex items-center gap-2">
                <SparklesIcon className="w-4 h-4 text-indigo-500" />
                <h2 className="font-semibold text-zinc-900 text-sm">Vision engine config</h2>
              </div>
              <span
                className={`text-xs font-medium px-2.5 py-1 rounded-full ${
                  isClipEnabled ? "bg-emerald-50 text-emerald-700" : "bg-zinc-100 text-zinc-600"
                }`}
              >
                {isClipEnabled ? "SigLIP active" : "Attribute mode"}
              </span>
            </div>

            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div>
                <h3 className="font-medium text-zinc-900 text-sm">SigLIP / CLIP semantic search</h3>
                <p className="text-sm text-zinc-500 mt-0.5 max-w-lg">
                  Enable zero-shot natural language description searching across tracked person frames.
                </p>
              </div>

              <button
                type="button"
                disabled={busy === "clip"}
                onClick={() => toggleClipEnabled(!isClipEnabled)}
                className={`px-4 py-2 rounded-xl text-sm font-medium transition-all shrink-0 disabled:opacity-50 ${
                  isClipEnabled
                    ? "bg-zinc-900 text-white hover:bg-zinc-800"
                    : "bg-zinc-100 text-zinc-700 hover:bg-zinc-200"
                }`}
              >
                {busy === "clip" ? "Updating…" : isClipEnabled ? "Semantic mode active ✓" : "Enable SigLIP mode"}
              </button>
            </div>

            <div className="pt-4 border-t border-zinc-100 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div>
                <h3 className="font-medium text-zinc-900 text-sm">Re-index search database</h3>
                <p className="text-sm text-zinc-500 mt-0.5">
                  Refresh visual embeddings for stored representative frames across video archives.
                </p>
              </div>

              <button
                type="button"
                disabled={busy === "reindex"}
                onClick={reindexSemanticSearch}
                className="px-4 py-2 rounded-xl bg-white border border-zinc-200/80 hover:border-zinc-300 hover:shadow-sm text-zinc-700 font-medium text-sm flex items-center gap-2 transition-all disabled:opacity-50 shrink-0"
              >
                <RefreshIcon className={`w-3.5 h-3.5 ${busy === "reindex" ? "animate-spin" : ""}`} />
                {busy === "reindex" ? "Re-indexing…" : "Re-index vector search"}
              </button>
            </div>
          </div>

          {/* Card 2: Reset Data */}
          <div className="bg-rose-50/40 border border-rose-200/70 rounded-[1.25rem] p-5 md:p-6 space-y-4">
            <div className="flex items-center gap-2 border-b border-rose-100 pb-4">
              <TrashIcon className="w-4 h-4 text-rose-500" />
              <h2 className="font-semibold text-rose-700 text-sm">Danger zone</h2>
            </div>

            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div>
                <h3 className="font-medium text-zinc-900 text-sm">Reset all tracking &amp; memory data</h3>
                <p className="text-sm text-zinc-500 mt-0.5 max-w-lg">
                  Permanently deletes all indexed track memories, person crops, evidence clips, and video pipeline jobs.
                </p>
              </div>

              <button
                type="button"
                disabled={busy === "reset"}
                onClick={() => setShowResetModal(true)}
                className="px-4 py-2 rounded-xl bg-rose-600 hover:bg-rose-700 text-white font-medium text-sm transition-all shrink-0 disabled:opacity-50"
              >
                Reset all data
              </button>
            </div>
          </div>
        </div>

        {/* Reset Confirmation Modal */}
        {showResetModal && (
          <ResetConfirmationModal
            onConfirm={confirmResetData}
            onCancel={() => setShowResetModal(false)}
            busy={busy === "reset"}
          />
        )}
      </div>
    </div>
  );
}
