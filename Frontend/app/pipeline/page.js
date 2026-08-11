"use client";

import React, { useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@clerk/nextjs";
import { useRouter } from "next/navigation";
import { apiRequest, fetchCurrentIdentity, getUsername, loadDashboard as fetchDashboard } from "../lib/api";
import VideoPipelineCard from "../components/VideoPipelineCard";
import { RefreshIcon, AlertIcon, CheckIcon } from "../components/Icons";

export default function PipelinePage() {
  const router = useRouter();
  const { isLoaded, isSignedIn } = useAuth();
  const [nativeUser, setNativeUser] = useState("");

  const [videoFile, setVideoFile] = useState(null);
  const [detectorModel, setDetectorModel] = useState("yolov8n.pt");
  const [frameStride, setFrameStride] = useState(2);
  const [jobOverrides, setJobOverrides] = useState([]);
  const [activeJobId, setActiveJobId] = useState("");
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  const fileInputRef = useRef(null);

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

  const { data: activeJobUpdate, error: activeJobError } = useQuery({
    queryKey: ["job", activeJobId],
    queryFn: () => apiRequest(`/api/tracking/jobs/${activeJobId}`),
    enabled: Boolean(activeJobId),
    refetchInterval: (query) => ["completed", "failed", "canceled"].includes(query.state.data?.status) ? false : 1_500,
  });

  const jobs = React.useMemo(() => {
    const seen = new Set();
    return [activeJobUpdate, ...jobOverrides, ...(dashboardData?.jobs || [])]
      .filter(Boolean)
      .filter((job) => {
        if (seen.has(job.job_id)) return false;
        seen.add(job.job_id);
        return true;
      });
  }, [activeJobUpdate, jobOverrides, dashboardData?.jobs]);

  const activeJobSettled = ["completed", "failed", "canceled"].includes(activeJobUpdate?.status);
  const activeJobFailed = ["failed", "canceled"].includes(activeJobUpdate?.status);
  const currentBusy = busy === "upload" && activeJobSettled ? "" : busy;

  async function uploadVideo(e) {
    if (e) e.preventDefault();
    if (!videoFile) {
      setError("Select a video file before running pipeline.");
      return;
    }
    setBusy("upload");
    setError("");
    setNotice("Uploading video stream...");
    const formData = new FormData();
    formData.append("file", videoFile);
    formData.append("detector_model", detectorModel);
    formData.append("frame_stride", String(frameStride));
    try {
      const payload = await apiRequest("/api/tracking/upload", {
        method: "POST",
        body: formData,
      });
      setJobOverrides((current) => [payload, ...current.filter((job) => job.job_id !== payload.job_id)]);
      setActiveJobId(payload.job_id);
      setNotice("Video stream queued for processing successfully.");
      await refreshDashboard();
    } catch (err) {
      setError(typeof err === "string" ? err : err.message || "Upload failed.");
      setBusy("");
    }
  }

  async function updateJob(jobId, action) {
    setError("");
    try {
      const payload = await apiRequest(`/api/tracking/jobs/${jobId}/${action}`, { method: "POST" });
      setJobOverrides((current) => [payload, ...current.filter((j) => j.job_id !== payload.job_id)]);
      if (action === "retry") {
        setActiveJobId(payload.job_id);
        setBusy("upload");
      }
      await refreshDashboard();
    } catch (err) {
      setError(typeof err === "string" ? err : err.message || "Job action failed.");
    }
  }

  async function deleteJob(jobId) {
    setError("");
    try {
      await apiRequest(`/api/tracking/jobs/${jobId}`, { method: "DELETE" });
      setJobOverrides((current) => current.filter((j) => j.job_id !== jobId));
      if (activeJobId === jobId) setActiveJobId("");
      setNotice("Job deleted.");
    } catch (err) {
      setError(typeof err === "string" ? err : err.message || "Delete job failed.");
    }
  }

  const activeJob = activeJobUpdate || jobs.find((j) => j.job_id === activeJobId) || jobs[0] || null;

  const requiresLogin = isLoaded && (!effectiveIsSignedIn || /login|token/i.test(dashboardError?.message || ""));
  React.useEffect(() => {
    if (requiresLogin) router.replace("/login");
  }, [requiresLogin, router]);

  if (requiresLogin) return null;

  const isAlertError = Boolean(error || activeJobFailed);

  return (
    <div className="relative app-page w-full animate-fade-in">
      {/* full-bleed ambient wash — tuned to amber pipeline accent */}
      <div className="pointer-events-none absolute -top-40 -left-20 w-[600px] h-[600px] rounded-full bg-gradient-to-br from-amber-200/50 via-orange-200/35 to-transparent blur-3xl opacity-70" />

      <div className="relative app-container space-y-6">
        {/* Header */}
        <header className="app-page-header">
          <div>
            <h1 className="app-page-title">Video pipeline</h1>
            <p className="app-page-description">
              Upload a video, configure tracking, and monitor its processing status.
            </p>
          </div>
          <button
            type="button"
            onClick={() => refreshDashboard()}
            className="shrink-0 px-3.5 py-2 rounded-xl bg-white border border-zinc-200/80 text-sm font-medium text-zinc-700 hover:border-zinc-300 hover:shadow-sm flex items-center gap-2 transition-all cursor-pointer"
          >
            <RefreshIcon className="w-3.5 h-3.5" />
            Refresh queue
          </button>
        </header>

        {/* Global Alerts */}
        {(notice || error || activeJobError) && (
          <div
            className={`p-4 rounded-2xl flex items-center justify-between gap-3 text-sm border shadow-sm ${
              isAlertError
                ? "bg-rose-50 border-rose-100 text-rose-700"
                : "bg-white border-zinc-200/80 text-zinc-700"
            }`}
          >
            <div className="flex items-center gap-2.5 min-w-0">
              {isAlertError ? (
                <AlertIcon className="w-4 h-4 text-rose-500 shrink-0" />
              ) : (
                <CheckIcon className="w-4 h-4 text-emerald-500 shrink-0" />
              )}
              <span className="font-medium truncate">{error || activeJobUpdate?.error || notice}</span>
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

        {/* Video Pipeline Card */}
        <VideoPipelineCard
          uploadVideo={uploadVideo}
          videoFile={videoFile}
          setVideoFile={setVideoFile}
          fileInputRef={fileInputRef}
          detectorModel={detectorModel}
          setDetectorModel={setDetectorModel}
          frameStride={frameStride}
          setFrameStride={setFrameStride}
          busy={currentBusy}
          activeJob={activeJob}
          jobs={jobs}
          updateJob={updateJob}
          deleteJob={deleteJob}
        />
      </div>
    </div>
  );
}
