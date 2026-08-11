"use client";

import React, { Suspense, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@clerk/nextjs";
import { useRouter, useSearchParams } from "next/navigation";
import { apiRequest, fetchCurrentIdentity, getUsername, loadDashboard as fetchDashboard } from "../lib/api";
import SurveillancePlayer from "../components/SurveillancePlayer";
import TrackDetections from "../components/TrackDetections";
import InspectModal from "../components/InspectModal";
import { VideoIcon, AlertIcon, CheckIcon } from "../components/Icons";
import { mediaUrl, sourceLabel } from "../components/UIHelpers";

function SurveillancePageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const selectedJobId = searchParams.get("jobId");
  const queryVideoUrl = searchParams.get("videoUrl");

  const { isLoaded, isSignedIn } = useAuth();
  const [nativeUser, setNativeUser] = useState("");

  const [focusedVideo, setFocusedVideo] = useState(queryVideoUrl ? mediaUrl(queryVideoUrl) : null);
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

  // Determine active video job
  const activeJob = useMemo(() => {
    if (selectedJobId) {
      const found = jobs.find((j) => j.job_id === selectedJobId);
      if (found) return found;
    }
    return jobs.find((j) => j.status === "completed") || jobs[0] || null;
  }, [selectedJobId, jobs]);

  const latestRunResult = activeJob?.result || null;

  const currentVideoTracks = useMemo(() => {
    const activeSource = latestRunResult?.source_name || activeJob?.source_name || activeJob?.uploaded_filename;
    if (!activeSource) return tracks;
    return tracks.filter(
      (t) => t.source_name === activeSource || t.source_label === activeSource || sourceLabel(t.source_label || t.source_name) === sourceLabel(activeSource)
    );
  }, [tracks, latestRunResult, activeJob]);

  async function watchClip(memoryId, title) {
    setBusy(`watch:${memoryId}`);
    setError("");
    try {
      const res = await apiRequest(`/api/tracking/clips/${encodeURIComponent(memoryId)}`, {
        method: "POST",
        body: JSON.stringify({ padding_frames: 15 }),
      });
      if (res.clip_url) {
        setFocusedVideo(mediaUrl(res.clip_url));
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
      {/* full-bleed ambient wash — tuned to emerald surveillance accent */}
      <div className="pointer-events-none absolute -top-40 -right-20 w-[600px] h-[600px] rounded-full bg-gradient-to-br from-emerald-200/50 via-teal-200/35 to-transparent blur-3xl opacity-70" />

      <div className="relative app-container space-y-6">
        {/* Header */}
        <header className="app-page-header">
          <div>
            <h1 className="app-page-title">Surveillance player</h1>
            <p className="app-page-description">
              Review processed video feeds with live detection overlays and inspect tracked subjects.
            </p>
          </div>

          {/* Video selector dropdown if multiple jobs exist */}
          {jobs.length > 0 && (
            <div className="flex items-center gap-2 shrink-0">
              <span className="text-xs font-medium text-zinc-500">Source</span>
              <select
                value={activeJob?.job_id || ""}
                onChange={(e) => router.push(`/surveillance?jobId=${e.target.value}`)}
                className="text-sm bg-white border border-zinc-200/80 rounded-xl px-3.5 py-2 font-medium text-zinc-800 focus:outline-none focus:ring-2 focus:ring-indigo-500 shadow-sm"
              >
                {jobs.map((j) => (
                  <option key={j.job_id} value={j.job_id}>
                    {j.uploaded_filename || j.filename || sourceLabel(j.source_name, "Surveillance Stream")} ({j.status})
                  </option>
                ))}
              </select>
            </div>
          )}
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

      {/* Video Player & Detections Sidebar */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Main Video Canvas Player */}
        <div className="lg:col-span-8 space-y-4">
          <SurveillancePlayer
            activeJob={activeJob}
            latestRunResult={latestRunResult}
            focusedVideo={focusedVideo}
            setFocusedVideo={setFocusedVideo}
          />
        </div>

        {/* Right Column: Track Detections Feed */}
        <div className="lg:col-span-4 space-y-4">
          <div className="app-surface p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-zinc-100 pb-3">
              <h3 className="font-semibold text-zinc-900 text-sm flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span>Video Tracked Subjects</span>
              </h3>
              <span className="text-xs px-2.5 py-1 rounded-full bg-zinc-100 font-medium text-zinc-700">
                {currentVideoTracks.length} subjects
              </span>
            </div>

            <TrackDetections
              tracks={currentVideoTracks}
              onInspect={(track) => setInspectTrack(track)}
              onExportClip={exportClip}
              busy={busy}
            />
          </div>
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

export default function SurveillancePage() {
  return (
    <Suspense fallback={<div className="app-page text-center text-sm text-zinc-400">Loading surveillance player…</div>}>
      <SurveillancePageContent />
    </Suspense>
  );
}
