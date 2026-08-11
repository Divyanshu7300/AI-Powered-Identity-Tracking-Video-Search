"use client";

import React, { useMemo, useState } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@clerk/nextjs";
import { useRouter } from "next/navigation";
import { apiRequest, fetchCurrentIdentity, getUsername, loadDashboard as fetchDashboard } from "../lib/api";
import TrackDetections from "../components/TrackDetections";
import InspectModal from "../components/InspectModal";
import { LayersIcon, SearchIcon, AlertIcon, CheckIcon, RefreshIcon } from "../components/Icons";
import { mediaUrl, sourceLabel } from "../components/UIHelpers";

export default function TracksPage() {
  const router = useRouter();
  const { isLoaded, isSignedIn } = useAuth();
  const [nativeUser, setNativeUser] = useState("");

  const [trackSearchFilter, setTrackSearchFilter] = useState("");
  const [showArchive, setShowArchive] = useState(false);
  const [inspectTrack, setInspectTrack] = useState(null);
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
    staleTime: 3_000,
    refetchInterval: 5_000,
    enabled: effectiveIsSignedIn,
  });

  const tracks = useMemo(() => dashboardData?.tracks || [], [dashboardData?.tracks]);
  const jobs = useMemo(() => dashboardData?.jobs || [], [dashboardData?.jobs]);

  const { filteredTracks, archiveGroups } = useMemo(() => {
    const rawGroups = {};
    const filtered = tracks.filter((t) => {
      const job = jobs.find((j) => j.source_name === t.source_name || j.job_id === t.source_name);
      const displaySource = t.uploaded_filename || job?.uploaded_filename || job?.filename || sourceLabel(t.source_label || t.source_name);
      if (!trackSearchFilter.trim()) return true;
      const q = trackSearchFilter.toLowerCase();
      const trackIdMatch = String(t.track_id).toLowerCase().includes(q);
      const labelMatch = (t.label || "").toLowerCase().includes(q);
      const sourceMatch = displaySource.toLowerCase().includes(q);
      return trackIdMatch || labelMatch || sourceMatch;
    });

    filtered.forEach((t) => {
      const srcKey = t.source_name || t.source_label || "unknown_source";
      if (!rawGroups[srcKey]) rawGroups[srcKey] = [];
      rawGroups[srcKey].push(t);
    });

    // Count occurrences of base filenames to differentiate multiple scans of the same video
    const filenameCounts = {};
    Object.keys(rawGroups).forEach((srcKey) => {
      const sampleTrack = rawGroups[srcKey][0];
      const job = jobs.find((j) => j.source_name === srcKey || j.job_id === srcKey);
      const name = sampleTrack?.uploaded_filename || job?.uploaded_filename || job?.filename || sourceLabel(sampleTrack?.source_label || srcKey);
      filenameCounts[name] = (filenameCounts[name] || 0) + 1;
    });

    const groupsList = [];
    const filenameSeen = {};

    Object.entries(rawGroups).forEach(([srcKey, srcTracks]) => {
      const sampleTrack = srcTracks[0];
      const matchedJob = jobs.find((j) => j.source_name === srcKey || j.job_id === srcKey);
      const baseName = sampleTrack?.uploaded_filename || matchedJob?.uploaded_filename || matchedJob?.filename || sourceLabel(sampleTrack?.source_label || srcKey);
      
      filenameSeen[baseName] = (filenameSeen[baseName] || 0) + 1;
      const hasMultiple = filenameCounts[baseName] > 1;
      const runIndex = filenameSeen[baseName];

      const timeCreated = matchedJob?.created_at
        ? new Date(matchedJob.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
        : null;

      const title = hasMultiple
        ? `${baseName} (Run ${runIndex}${timeCreated ? ` • ${timeCreated}` : ""})`
        : baseName;

      groupsList.push({
        sourceKey: srcKey,
        title,
        matchedJob,
        tracks: srcTracks,
      });
    });

    return { filteredTracks: filtered, archiveGroups: groupsList };
  }, [tracks, jobs, trackSearchFilter]);

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
      {/* full-bleed ambient wash — tuned to sky blue tracks accent */}
      <div className="pointer-events-none absolute -top-40 -right-20 w-[600px] h-[600px] rounded-full bg-gradient-to-br from-sky-200/50 via-blue-200/35 to-transparent blur-3xl opacity-70" />

      <div className="relative app-container space-y-6">
        {/* Header */}
        <header className="app-page-header">
          <div>
            <h1 className="app-page-title">Track library</h1>
            <p className="app-page-description">
              Browse indexed subject memories, inspect detections, and export video evidence.
            </p>
          </div>

          <button
            type="button"
            onClick={() => refreshDashboard()}
            className="shrink-0 px-3.5 py-2 rounded-xl bg-white border border-zinc-200/80 text-sm font-medium text-zinc-700 hover:border-zinc-300 hover:shadow-sm flex items-center gap-2 transition-all cursor-pointer"
            title="Refresh Database"
          >
            <RefreshIcon className="w-3.5 h-3.5" />
            Refresh database
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

      {/* Filter & Controls Bar */}
      <div className="app-surface p-4 flex flex-col sm:flex-row items-center justify-between gap-4">
        {/* Search filter input */}
        <div className="relative w-full sm:w-80">
          <SearchIcon className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-zinc-400" />
          <input
            type="text"
            value={trackSearchFilter}
            onChange={(e) => setTrackSearchFilter(e.target.value)}
            placeholder="Search tracks by ID or video source..."
            className="w-full text-sm bg-zinc-50 border border-zinc-200 rounded-xl pl-9 pr-3 py-2 text-zinc-800 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
        </div>

        {/* Source view toggles */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setShowArchive(false)}
            className={`px-3 py-1.5 rounded-xl text-xs font-medium transition-all cursor-pointer ${
              !showArchive ? "bg-indigo-600 text-white shadow-sm" : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200"
            }`}
          >
            All Tracks ({filteredTracks.length})
          </button>
          <button
            type="button"
            onClick={() => setShowArchive(true)}
            className={`px-3 py-1.5 rounded-xl text-xs font-medium transition-all cursor-pointer ${
              showArchive ? "bg-indigo-600 text-white shadow-sm" : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200"
            }`}
          >
            Grouped By Video Source ({archiveGroups.length})
          </button>
        </div>
      </div>

      {/* Main Track Display */}
      {!showArchive ? (
        <TrackDetections
          tracks={filteredTracks}
          onInspect={(track) => setInspectTrack(track)}
          onExportClip={exportClip}
          busy={busy}
        />
      ) : (
        <div className="space-y-6">
          {archiveGroups.map((group) => (
            <TrackDetections
              key={group.sourceKey}
              title={group.title}
              tracks={group.tracks}
              onInspect={(track) => setInspectTrack(track)}
              onExportClip={exportClip}
              busy={busy}
              headerAction={
                group.matchedJob ? (
                  <Link
                    href={`/surveillance?jobId=${group.matchedJob.job_id}`}
                    className="text-xs font-semibold text-indigo-600 hover:text-indigo-700 hover:underline flex items-center gap-1 ml-1"
                  >
                    Watch Feed →
                  </Link>
                ) : null
              }
            />
          ))}
          {archiveGroups.length === 0 && (
            <div className="app-surface p-8 text-center text-zinc-400 text-sm">
              No track groups found matching query filter.
            </div>
          )}
        </div>
      )}

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
