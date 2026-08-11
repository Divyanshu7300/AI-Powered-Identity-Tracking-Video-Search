"use client";

import React, { useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@clerk/nextjs";
import { useRouter } from "next/navigation";
import { apiRequest, fetchCurrentIdentity, getUsername, loadDashboard as fetchDashboard } from "../lib/api";
import SearchPanel from "../components/SearchPanel";
import MatchCard from "../components/MatchCard";
import FormattedRagAnswer from "../components/FormattedRagAnswer";
import InspectModal from "../components/InspectModal";
import { SearchIcon, SparklesIcon, AlertIcon, CheckIcon } from "../components/Icons";
import { DEFAULT_TEXT_QUERY, mediaUrl } from "../components/UIHelpers";

export default function SearchPage() {
  const router = useRouter();
  const { isLoaded, isSignedIn } = useAuth();
  const [nativeUser, setNativeUser] = useState("");

  const [textQuery, setTextQuery] = useState(DEFAULT_TEXT_QUERY);
  const [queryFile, setQueryFile] = useState(null);
  const [imageSearchMode, setImageSearchMode] = useState("hybrid");
  const [useLlm, setUseLlm] = useState(true);
  const [searchResults, setSearchResults] = useState(null);
  const [clipEnabledOverride, setClipEnabledOverride] = useState(null);
  const [inspectTrack, setInspectTrack] = useState(null);
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  const imageInputRef = useRef(null);

  React.useEffect(() => {
    fetchCurrentIdentity().then((identity) => {
      if (identity?.username) setNativeUser(identity.username);
    });
  }, []);

  const effectiveIsSignedIn = Boolean(isSignedIn || nativeUser || getUsername());

  const { data: dashboardData, error: dashboardError, refetch: refreshDashboard } = useQuery({
    queryKey: ["dashboard"],
    queryFn: fetchDashboard,
    staleTime: 3_000,
    refetchInterval: 5_000,
    enabled: effectiveIsSignedIn,
  });

  const semanticStatus = dashboardData?.dashboard?.semantic_status || {};
  const clipEnabled = clipEnabledOverride ?? Boolean(semanticStatus?.clip_enabled);

  async function searchByText(e) {
    if (e) e.preventDefault();
    if (!textQuery.trim()) return;
    setBusy("text");
    setError("");
    try {
      const payload = await apiRequest("/api/tracking/search/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: textQuery, top_k: 8, use_llm: useLlm }),
      });
      setSearchResults({
        type: "text",
        query: textQuery,
        matches: payload.matches || [],
        message: payload.message,
        rag: payload.rag,
        queryContext: payload.query_context,
        strategy: payload.search_strategy,
        noResultGuidance: payload.no_result_guidance,
      });
      setNotice(payload.message || `Text search complete — ${payload.matches?.length || 0} matches found.`);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  async function searchByImage(e) {
    if (e) e.preventDefault();
    if (!queryFile) {
      setError("Select a query crop image first.");
      return;
    }
    setBusy("image");
    setError("");
    const formData = new FormData();
    formData.append("file", queryFile);
    formData.append("mode", imageSearchMode);
    try {
      const payload = await apiRequest("/api/tracking/search?top_k=8", {
        method: "POST",
        body: formData,
      });
      setSearchResults({
        type: "image",
        query: queryFile.name,
        matches: payload.matches || [],
        message: payload.message,
      });
      setNotice(payload.message || `Image search complete — ${payload.matches?.length || 0} matches found.`);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  async function toggleClipEnabled(enabled) {
    try {
      await apiRequest("/api/tracking/search/settings", {
        method: "POST",
        body: JSON.stringify({ clip_enabled: enabled }),
      });
      setClipEnabledOverride(enabled);
      setNotice(enabled ? "SigLIP vision semantic engine enabled." : "SigLIP disabled — attribute search mode active.");
      await refreshDashboard();
    } catch (err) {
      setError(err.message);
    }
  }

  async function reindexSemanticSearch() {
    setBusy("reindex");
    setError("");
    try {
      const payload = await apiRequest("/api/tracking/search/reindex", { method: "POST" });
      setNotice(`Vector index refreshed: ${payload.refreshed || 0} representative keyframes updated.`);
      await refreshDashboard();
    } catch (err) {
      setError(err.message || "Unable to refresh search index.");
    } finally {
      setBusy("");
    }
  }

  async function watchClip(memoryId, title) {
    setBusy(`watch:${memoryId}`);
    setError("");
    try {
      const res = await apiRequest(`/api/tracking/clips/${encodeURIComponent(memoryId)}`, {
        method: "POST",
        body: JSON.stringify({ padding_frames: 15 }),
      });
      if (res.clip_url) {
        router.push(`/surveillance?videoUrl=${encodeURIComponent(res.clip_url)}`);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  async function exportClip(memoryId, title) {
    setBusy(`clip:${memoryId}`);
    setError("");
    try {
      const res = await apiRequest(`/api/tracking/clips/${encodeURIComponent(memoryId)}`, {
        method: "POST",
        body: JSON.stringify({ padding_frames: 15 }),
      });
      setNotice(`Exported evidence video clip for ${title}`);
      await refreshDashboard();
      if (res.clip_url) {
        const downloadUrl = mediaUrl(res.clip_url);
        const a = document.createElement("a");
        a.href = downloadUrl;
        a.download = `${title.replace(/[^a-z0-9_-]/gi, "_")}_evidence.mp4`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setNotice(`Exported and downloaded evidence clip for ${title}`);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  const requiresLogin = isLoaded && (!effectiveIsSignedIn || /login|token/i.test(dashboardError?.message || ""));
  React.useEffect(() => {
    if (requiresLogin) router.replace("/login");
  }, [requiresLogin, router]);

  if (requiresLogin) return null;

  return (
    <div className="relative app-page w-full animate-fade-in">
      {/* full-bleed ambient wash — tuned to indigo/violet search accent */}
      <div className="pointer-events-none absolute -top-40 -right-20 w-[600px] h-[600px] rounded-full bg-gradient-to-br from-indigo-200/50 via-violet-200/35 to-transparent blur-3xl opacity-70" />

      <div className="relative app-container space-y-6">
        {/* Header */}
        <header className="app-page-header">
          <div>
            <h1 className="app-page-title">AI search</h1>
            <p className="app-page-description">
              Find subjects with natural-language descriptions or reference crop Re-ID.
            </p>
          </div>
          <button
            type="button"
            disabled={busy === "reindex"}
            onClick={reindexSemanticSearch}
            className="shrink-0 px-3.5 py-2 rounded-xl bg-white border border-zinc-200/80 text-sm font-medium text-zinc-700 hover:border-zinc-300 hover:shadow-sm flex items-center gap-2 transition-all cursor-pointer disabled:opacity-50"
          >
            <SparklesIcon className="w-3.5 h-3.5 text-indigo-500" />
            {busy === "reindex" ? "Re-indexing…" : "Refresh vector index"}
          </button>
        </header>

        {/* Global Alerts */}
        {(notice || error) && (
          <div
            className={`p-4 rounded-2xl flex items-center justify-between gap-3 text-sm border shadow-sm ${
              error ? "bg-rose-50 border-rose-100 text-rose-700" : "bg-white border-zinc-200/80 text-zinc-700"
            }`}
          >
            <div className="flex items-center gap-2.5 min-w-0">
              {error ? (
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

        {/* Main Search Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Search Panel Column */}
          <div className="lg:col-span-5">
            <SearchPanel
              searchByText={searchByText}
              textQuery={textQuery}
              setTextQuery={setTextQuery}
              useLlm={useLlm}
              setUseLlm={setUseLlm}
              searchByImage={searchByImage}
              queryFile={queryFile}
              setQueryFile={setQueryFile}
              imageInputRef={imageInputRef}
              imageSearchMode={imageSearchMode}
              setImageSearchMode={setImageSearchMode}
              busy={busy}
              clipEnabled={clipEnabled}
              toggleClipEnabled={toggleClipEnabled}
              reindexSemanticSearch={reindexSemanticSearch}
              semanticStatus={semanticStatus}
            />
          </div>

          {/* Search Results Column */}
          <div className="lg:col-span-7 space-y-6">
            {!searchResults ? (
              <div className="app-surface p-8 md:p-10 text-center space-y-4">
                <div className="w-14 h-14 rounded-2xl bg-indigo-50 border border-indigo-100 flex items-center justify-center mx-auto">
                  <SearchIcon className="w-6 h-6 text-indigo-600" />
                </div>
                <h3 className="font-semibold text-zinc-900 text-[15px]">Enter a query to see matches</h3>
                <p className="text-sm text-zinc-500 max-w-sm mx-auto leading-relaxed">
                  Try a description like{" "}
                  <code className="bg-zinc-100 text-zinc-700 px-1.5 py-0.5 rounded-md text-[13px]">
                    &quot;man with red jacket near center&quot;
                  </code>{" "}
                  or upload a target crop photo.
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* RAG AI Answer Card */}
                {searchResults.rag && (
                  <FormattedRagAnswer rag={searchResults.rag} queryContext={searchResults.queryContext} />
                )}

                {/* Match Cards Container */}
                <div className="app-surface p-5 md:p-6 space-y-4">
                  <div className="flex items-center justify-between border-b border-zinc-100 pb-3.5">
                    <div className="flex items-center gap-2">
                      <SparklesIcon className="w-4 h-4 text-indigo-500" />
                      <h3 className="font-semibold text-zinc-900 text-sm">Matched subject profiles</h3>
                    </div>
                    <span className="text-xs font-medium px-3 py-1 rounded-full bg-indigo-50 text-indigo-700">
                      {searchResults.matches?.length || 0} matches
                    </span>
                  </div>

                  {searchResults.matches?.length === 0 ? (
                    <p className="text-sm text-zinc-400 py-8 text-center">No matching person tracks found for this query.</p>
                  ) : (
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      {searchResults.matches.map((match, idx) => (
                        <MatchCard
                          key={match.memory_id || match.track_id || idx}
                          match={match}
                          rank={idx + 1}
                          onInspect={() => setInspectTrack(match)}
                          onExportClip={() => exportClip(match.memory_id || match.track_id, match.label || `Track #${match.track_id}`)}
                          busy={busy}
                        />
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Inspect Modal */}
        {inspectTrack && (
          <InspectModal
            track={inspectTrack}
            onClose={() => setInspectTrack(null)}
            onExportClip={exportClip}
            onWatchClip={watchClip}
            busy={busy}
          />
        )}
      </div>
    </div>
  );
}
