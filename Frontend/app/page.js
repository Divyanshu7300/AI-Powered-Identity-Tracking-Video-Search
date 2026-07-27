"use client";

import Image from "next/image";
import { useEffect, useMemo, useRef, useState } from "react";

const DEFAULT_TEXT_QUERY = "person wearing blue clothing near center middle";
const PLACEHOLDER_IMAGE = "data:image/gif;base64,R0lGODlhAQABAAAAACw=";

function mediaUrl(path) {
  if (!path) return PLACEHOLDER_IMAGE;
  if (/^https?:\/\//i.test(path)) return path;
  return `/api${path.startsWith("/") ? path : `/${path}`}`;
}

async function readJson(response) {
  const text = await response.text();
  try {
    return text ? JSON.parse(text) : {};
  } catch {
    return { message: text || response.statusText };
  }
}

export default function Home() {
  const [health, setHealth] = useState("checking");
  const [dashboard, setDashboard] = useState(null);
  const [tracks, setTracks] = useState([]);
  const [selectedTrack, setSelectedTrack] = useState(null);
  const [videoFile, setVideoFile] = useState(null);
  const [queryFile, setQueryFile] = useState(null);
  const [detectorModel, setDetectorModel] = useState("yolov8n.pt");
  const [frameStride, setFrameStride] = useState(2);
  const [textQuery, setTextQuery] = useState(DEFAULT_TEXT_QUERY);
  const [textResults, setTextResults] = useState(null);
  const [imageResults, setImageResults] = useState(null);
  const [runResult, setRunResult] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [activeJobId, setActiveJobId] = useState("");
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);
  const imageInputRef = useRef(null);
  const sessionIdRef = useRef("");

  function getSessionId() {
    if (sessionIdRef.current) return sessionIdRef.current;
    const storageKey = "mot-reid-session-id";
    const stored = window.localStorage.getItem(storageKey);
    const nextId = stored || crypto.randomUUID();
    if (!stored) window.localStorage.setItem(storageKey, nextId);
    sessionIdRef.current = nextId;
    return nextId;
  }

  function apiFetch(path, options = {}) {
    const headers = new Headers(options.headers || {});
    headers.set("x-session-id", getSessionId());
    return fetch(path, { ...options, headers });
  }

  const overview = dashboard?.overview || {};
  const semanticStatus = dashboard?.semantic_status || {};
  const reidStatus = dashboard?.reid_status || {};
  const topTracks = dashboard?.tracker?.top_tracks || [];

  const selectedMemory = useMemo(() => {
    if (!selectedTrack) return tracks[0] || null;
    return tracks.find((track) => track.memory_id === selectedTrack) || tracks[0] || null;
  }, [selectedTrack, tracks]);

  async function loadDashboard() {
    try {
      const [healthResponse, dashboardResponse, tracksResponse, jobsResponse] = await Promise.all([
        apiFetch("/api/health", { cache: "no-store" }),
        apiFetch("/api/tracking/analytics/dashboard", { cache: "no-store" }),
        apiFetch("/api/tracking/analytics/tracks", { cache: "no-store" }),
        apiFetch("/api/tracking/jobs", { cache: "no-store" }),
      ]);
      setHealth(healthResponse.ok ? "online" : "offline");
      if (dashboardResponse.ok) setDashboard(await dashboardResponse.json());
      if (tracksResponse.ok) {
        const payload = await tracksResponse.json();
        setTracks(payload.track_memories || []);
      }
      if (jobsResponse.ok) {
        const payload = await jobsResponse.json();
        setJobs(payload.jobs || []);
      }
    } catch (err) {
      setHealth("offline");
      setError(err.message || "Unable to reach backend.");
    }
  }

  useEffect(() => {
    Promise.resolve().then(loadDashboard);
    const id = setInterval(loadDashboard, 5000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!activeJobId) return undefined;
    let cancelled = false;

    async function pollJob() {
      try {
        const response = await apiFetch(`/api/tracking/jobs/${activeJobId}`, { cache: "no-store" });
        const payload = await readJson(response);
        if (!response.ok) throw new Error(payload.detail || payload.message || "Unable to load job.");
        if (cancelled) return;
        setJobs((current) => {
          const rest = current.filter((job) => job.job_id !== payload.job_id);
          return [payload, ...rest];
        });
        if (payload.status === "completed") {
          setRunResult(payload.result);
          setNotice(`Processed ${payload.result?.sampled_frames_processed || 0} sampled frames.`);
          setBusy("");
          setActiveJobId("");
          awaitDashboardRefresh();
        } else if (payload.status === "failed" || payload.status === "canceled") {
          setError(payload.error || "Video processing failed.");
          setBusy("");
          setActiveJobId("");
          awaitDashboardRefresh();
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    }

    function awaitDashboardRefresh() {
      Promise.resolve().then(loadDashboard);
    }

    pollJob();
    const id = setInterval(pollJob, 1500);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [activeJobId]);

  async function uploadVideo(event) {
    event.preventDefault();
    if (!videoFile) {
      setError("Select a video before running the pipeline.");
      return;
    }
    setBusy("upload");
    setError("");
    setNotice("Processing video. This can take a while.");
    const formData = new FormData();
    formData.append("file", videoFile);
    formData.append("detector_model", detectorModel);
    formData.append("frame_stride", String(frameStride));
    try {
      const response = await apiFetch("/api/tracking/upload", {
        method: "POST",
        body: formData,
      });
      const payload = await readJson(response);
      if (!response.ok) throw new Error(payload.detail || payload.message || "Upload failed.");
      setJobs((current) => [payload, ...current.filter((job) => job.job_id !== payload.job_id)]);
      setActiveJobId(payload.job_id);
      setNotice("Job queued. Progress will update here.");
      await loadDashboard();
    } catch (err) {
      setError(err.message);
      setBusy("");
    }
  }

  async function searchByText(event) {
    event.preventDefault();
    setBusy("text");
    setError("");
    try {
      const response = await apiFetch("/api/tracking/search/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: textQuery, top_k: 8 }),
      });
      const payload = await readJson(response);
      if (!response.ok) throw new Error(payload.detail || payload.message || "Text search failed.");
      setTextResults(payload);
      setNotice(payload.message || "Text search complete.");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  async function searchByImage(event) {
    event.preventDefault();
    if (!queryFile) {
      setError("Select a query image first.");
      return;
    }
    setBusy("image");
    setError("");
    const formData = new FormData();
    formData.append("file", queryFile);
    try {
      const response = await apiFetch("/api/tracking/search?top_k=8", {
        method: "POST",
        body: formData,
      });
      const payload = await readJson(response);
      if (!response.ok) throw new Error(payload.detail || payload.message || "Image search failed.");
      setImageResults(payload);
      setNotice("Image search complete.");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  async function exportClip(memoryId) {
    setBusy(`clip:${memoryId}`);
    setError("");
    try {
      const response = await apiFetch(`/api/tracking/clips/${encodeURIComponent(memoryId)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ padding_frames: 0 }),
      });
      const payload = await readJson(response);
      if (!response.ok) throw new Error(payload.detail || payload.message || "Clip export failed.");
      setNotice(`Clip exported for ${memoryId}.`);
      await loadDashboard();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  async function updateJob(jobId, action) {
    setError("");
    try {
      const response = await apiFetch(`/api/tracking/jobs/${jobId}/${action}`, { method: "POST" });
      const payload = await readJson(response);
      if (!response.ok) throw new Error(payload.detail || payload.message || `Job ${action} failed.`);
      setJobs((current) => [payload, ...current.filter((job) => job.job_id !== payload.job_id)]);
      if (action === "retry") {
        setActiveJobId(payload.job_id);
        setBusy("upload");
        setNotice("Retry queued. Progress will update here.");
      } else {
        setNotice(payload.progress?.message || `Job ${action} requested.`);
      }
      await loadDashboard();
    } catch (err) {
      setError(err.message);
    }
  }

  async function deleteJob(jobId) {
    setError("");
    try {
      const response = await apiFetch(`/api/tracking/jobs/${jobId}`, { method: "DELETE" });
      const payload = await readJson(response);
      if (!response.ok) throw new Error(payload.detail || payload.message || "Delete job failed.");
      setJobs((current) => current.filter((job) => job.job_id !== jobId));
      if (activeJobId === jobId) setActiveJobId("");
      setNotice("Job removed from history.");
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <main className="shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">MOT + Re-ID Video Search</p>
          <h1>Video memory dashboard</h1>
        </div>
        <div className={`status ${health}`}>
          <span />
          {health}
        </div>
      </section>

      {(notice || error) && (
        <section className={`notice ${error ? "error" : ""}`}>
          {error || notice}
        </section>
      )}

      <section className="metrics" aria-label="Dashboard metrics">
        <Metric label="Track memories" value={overview.indexed_track_memories ?? 0} />
        <Metric label="Sources" value={overview.sources_processed ?? 0} />
        <Metric label="Semantic observations" value={overview.semantic_observations ?? 0} />
        <Metric label="Active tracks" value={overview.active_tracks ?? 0} />
        <Metric label="Frames processed" value={overview.frames_processed ?? 0} />
        <Metric label="CLIP" value={semanticStatus.clip_ready ? "ready" : "fallback"} />
        <Metric label="Re-ID" value={reidStatus.backend || "not loaded"} />
      </section>

      <section className="workspace">
        <div className="left-rail">
          <form className="panel" onSubmit={uploadVideo}>
            <div className="panel-head">
              <h2>Run video</h2>
              <button className="icon-button" type="button" title="Refresh" onClick={loadDashboard}>
                R
              </button>
            </div>
            <input
              ref={fileInputRef}
              className="hidden-input"
              type="file"
              accept="video/*"
              onChange={(event) => setVideoFile(event.target.files?.[0] || null)}
            />
            <button className="file-picker" type="button" onClick={() => fileInputRef.current?.click()}>
              <span>{videoFile ? videoFile.name : "Choose video"}</span>
            </button>
            <label>
              Detector
              <select value={detectorModel} onChange={(event) => setDetectorModel(event.target.value)}>
                <option value="yolov8n.pt">yolov8n.pt</option>
                <option value="yolov8s.pt">yolov8s.pt</option>
              </select>
            </label>
            <label>
              Frame stride
              <input
                type="number"
                min="1"
                max="30"
                value={frameStride}
                onChange={(event) => setFrameStride(Math.max(1, Number(event.target.value) || 1))}
              />
            </label>
            <button className="primary" disabled={busy === "upload"} type="submit">
              {busy === "upload" ? "Queued..." : "Run pipeline"}
            </button>
          </form>

          <JobProgress
            jobs={jobs}
            activeJobId={activeJobId}
            onCancel={(jobId) => updateJob(jobId, "cancel")}
            onDelete={deleteJob}
            onRetry={(jobId) => updateJob(jobId, "retry")}
            onSelect={setActiveJobId}
          />

          <form className="panel" onSubmit={searchByText}>
            <div className="panel-head">
              <h2>Text search</h2>
            </div>
            <textarea value={textQuery} onChange={(event) => setTextQuery(event.target.value)} />
            <button className="primary" disabled={busy === "text"} type="submit">
              {busy === "text" ? "Searching..." : "Search text"}
            </button>
          </form>

          <form className="panel" onSubmit={searchByImage}>
            <div className="panel-head">
              <h2>Image search</h2>
            </div>
            <input
              ref={imageInputRef}
              className="hidden-input"
              type="file"
              accept="image/*"
              onChange={(event) => setQueryFile(event.target.files?.[0] || null)}
            />
            <button className="file-picker" type="button" onClick={() => imageInputRef.current?.click()}>
              <span>{queryFile ? queryFile.name : "Choose person image"}</span>
            </button>
            <button className="primary" disabled={busy === "image"} type="submit">
              {busy === "image" ? "Searching..." : "Search image"}
            </button>
          </form>
        </div>

        <section className="main-grid">
          <div className="panel output-panel">
            <div className="panel-head">
              <h2>Latest output</h2>
              {runResult?.output_url && (
                <a href={mediaUrl(runResult.output_url)} target="_blank" rel="noreferrer">
                  Open
                </a>
              )}
            </div>
            {runResult?.output_url ? (
              <video controls src={mediaUrl(runResult.output_url)} />
            ) : (
              <div className="empty-state">Run a video to preview the annotated output.</div>
            )}
          </div>

          <div className="panel">
            <div className="panel-head">
              <h2>Track memories</h2>
              <span>{tracks.length}</span>
            </div>
            <div className="track-list">
              {tracks.map((track) => (
                <button
                  className={`track-row ${selectedMemory?.memory_id === track.memory_id ? "selected" : ""}`}
                  key={track.memory_id}
                  type="button"
                  onClick={() => setSelectedTrack(track.memory_id)}
                >
                  <EvidenceImage src={track.best_crop_url || track.crop_url} width={56} height={56} />
                  <span>
                    <strong>{track.memory_id}</strong>
                    <small>{track.hits} sightings | conf {track.best_confidence}</small>
                  </span>
                </button>
              ))}
              {!tracks.length && <div className="empty-state compact">No memories indexed yet.</div>}
            </div>
          </div>

          <TrackDetail track={selectedMemory} busy={busy} onExport={exportClip} />
          <ResultsPanel title="Text matches" payload={textResults} />
          <ResultsPanel title="Image matches" payload={imageResults} />
          <TopTracks tracks={topTracks} />
        </section>
      </section>
    </main>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function JobProgress({ jobs, activeJobId, onCancel, onDelete, onRetry, onSelect }) {
  const activeJob = jobs.find((job) => job.job_id === activeJobId) || jobs[0] || null;
  const progress = activeJob?.progress || {};
  const percent = Number(progress.percent || 0);
  const isLive = activeJob && ["queued", "running", "cancel_requested"].includes(activeJob.status);
  const isFinished = activeJob && ["completed", "failed", "canceled"].includes(activeJob.status);

  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Job progress</h2>
        <span className={`job-pill ${activeJob?.status || "idle"}`}>{activeJob?.status || "idle"}</span>
      </div>
      {activeJob ? (
        <>
          <div className="progress-track" aria-label="Video processing progress">
            <span style={{ width: `${Math.max(0, Math.min(100, percent))}%` }} />
          </div>
          <div className="progress-meta">
            <strong>{Math.round(percent)}%</strong>
            <small>
              {progress.frames_processed || 0}
              {progress.total_frames ? ` / ${progress.total_frames}` : ""} frames
            </small>
          </div>
          <p className="message">{activeJob.error || progress.message || "Waiting for progress."}</p>
          <div className="job-actions">
            {isLive && (
              <button className="secondary" type="button" onClick={() => onCancel(activeJob.job_id)}>
                Cancel
              </button>
            )}
            {isFinished && (
              <button className="secondary" type="button" onClick={() => onRetry(activeJob.job_id)}>
                Retry
              </button>
            )}
            {isFinished && (
              <button className="secondary danger-button" type="button" onClick={() => onDelete(activeJob.job_id)}>
                Delete
              </button>
            )}
          </div>
          <div className="job-list">
            {jobs.slice(0, 5).map((job) => (
              <button
                className={`job-row ${job.job_id === activeJob.job_id ? "selected" : ""}`}
                key={job.job_id}
                type="button"
                onClick={() => onSelect(job.job_id)}
              >
                <span>{job.uploaded_filename || job.source_name}</span>
                <small>{job.status}</small>
              </button>
            ))}
          </div>
          {isLive && <p className="message">Dashboard stays usable while this runs.</p>}
        </>
      ) : (
        <div className="empty-state compact">No processing jobs yet.</div>
      )}
    </div>
  );
}

function TrackDetail({ track, busy, onExport }) {
  if (!track) {
    return (
      <div className="panel detail-panel">
        <div className="empty-state">Select a track to inspect evidence and export clips.</div>
      </div>
    );
  }

  return (
    <div className="panel detail-panel">
      <div className="panel-head">
        <h2>{track.memory_id}</h2>
        <button
          className="secondary"
          disabled={busy === `clip:${track.memory_id}`}
          type="button"
          onClick={() => onExport(track.memory_id)}
        >
          {busy === `clip:${track.memory_id}` ? "Exporting..." : "Export clip"}
        </button>
      </div>
      <div className="detail-body">
        <EvidenceImage src={track.evidence_url || track.best_crop_url || track.crop_url} width={220} height={165} />
        <dl>
          <div>
            <dt>Frames</dt>
            <dd>{track.first_frame} - {track.last_frame}</dd>
          </div>
          <div>
            <dt>Visible</dt>
            <dd>{track.visible_duration_seconds ?? track.duration_frames}</dd>
          </div>
          <div>
            <dt>Best confidence</dt>
            <dd>{track.best_confidence}</dd>
          </div>
          <div>
            <dt>Activity</dt>
            <dd>{track.episode?.activity || "tracked person"}</dd>
          </div>
        </dl>
      </div>
      {track.clip_url && (
        <a className="clip-link" href={mediaUrl(track.clip_url)} target="_blank" rel="noreferrer">
          Open exported clip
        </a>
      )}
    </div>
  );
}

function ResultsPanel({ title, payload }) {
  const matches = payload?.matches || [];
  return (
    <div className="panel">
      <div className="panel-head">
        <h2>{title}</h2>
        <span>{matches.length}</span>
      </div>
      {payload?.message && <p className="message">{payload.message}</p>}
      <div className="result-list">
        {matches.map((match) => (
          <div className="result-row" key={`${title}-${match.memory_id}`}>
            <EvidenceImage src={match.frame_url || match.best_crop_url || match.crop_url} width={56} height={56} />
            <span>
              <strong>{match.memory_id}</strong>
              <small>score {match.score ?? match.similarity ?? "n/a"}</small>
              {match.caption && <em>{match.caption}</em>}
            </span>
          </div>
        ))}
        {!matches.length && <div className="empty-state compact">No matches yet.</div>}
      </div>
    </div>
  );
}

function EvidenceImage({ src, width, height }) {
  return (
    <Image
      alt=""
      height={height}
      src={mediaUrl(src)}
      unoptimized
      width={width}
    />
  );
}

function TopTracks({ tracks }) {
  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Longest tracks</h2>
      </div>
      <div className="rank-list">
        {tracks.map((track, index) => (
          <div className="rank-row" key={track.memory_id || index}>
            <strong>{index + 1}</strong>
            <span>{track.memory_id}</span>
            <small>{track.duration_frames} frames</small>
          </div>
        ))}
        {!tracks.length && <div className="empty-state compact">No ranked tracks yet.</div>}
      </div>
    </div>
  );
}
