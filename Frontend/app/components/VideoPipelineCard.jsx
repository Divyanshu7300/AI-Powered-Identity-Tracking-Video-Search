"use client";

import React from "react";
import Link from "next/link";
import { CameraVideoIcon, CpuIcon, PlayIcon, RefreshIcon, TrashIcon, UploadIcon } from "./Icons";
import { mediaUrl, sourceLabel } from "./UIHelpers";

export default function VideoPipelineCard({
  uploadVideo,
  videoFile,
  setVideoFile,
  fileInputRef,
  detectorModel,
  setDetectorModel,
  frameStride,
  setFrameStride,
  busy,
  activeJob,
  updateJob,
  deleteJob,
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
      const droppedFile = e.dataTransfer.files[0];
      setVideoFile(droppedFile);
    }
  };

  return (
    <div className="space-y-6">
      {/* Form Card */}
      <form className="app-surface p-5 md:p-6 space-y-6" onSubmit={uploadVideo}>
        <div className="flex items-center justify-between pb-4 border-b border-zinc-100">
          <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center shrink-0">
              <CameraVideoIcon className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-zinc-900">Video stream ingestion</h2>
              <p className="text-xs text-zinc-400">Configure YOLO detector &amp; frame extraction stride</p>
            </div>
          </div>
          <span className="text-xs font-medium text-orange-700 bg-orange-50 px-3 py-1 rounded-full">YOLOv8 + Re-ID</span>
        </div>

        <input
          ref={fileInputRef}
          className="hidden"
          type="file"
          accept="video/*"
          onChange={(e) => setVideoFile(e.target.files?.[0] || null)}
        />

        {/* Dropzone */}
        <div
          className={`relative border-2 border-dashed rounded-2xl p-7 text-center cursor-pointer transition-all flex flex-col items-center justify-center gap-3 group ${
            isDragging
              ? "border-orange-500 bg-orange-50/70 scale-[1.01]"
              : "border-zinc-300 hover:border-orange-300 bg-zinc-50 hover:bg-orange-50/40"
          }`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="w-12 h-12 rounded-2xl bg-white border border-zinc-200/80 group-hover:border-orange-200 text-zinc-400 group-hover:text-orange-500 flex items-center justify-center transition-all group-hover:scale-105 shadow-sm">
            <UploadIcon className="w-5 h-5" />
          </div>
          <div>
            <p className="text-sm font-medium text-zinc-800 group-hover:text-orange-600 transition-colors">
              {videoFile ? videoFile.name : "Select a video file or drag and drop here"}
            </p>
            <p className="text-xs text-zinc-400 mt-1">MP4 · AVI · MOV · MKV</p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
          <div>
            <label className="text-xs font-medium text-zinc-500 mb-2 block">Detector variant</label>
            <select
              className="w-full bg-zinc-50 border border-zinc-200/80 focus:border-orange-300 focus:ring-2 focus:ring-orange-100 rounded-2xl px-3.5 py-2.5 text-sm font-medium text-zinc-800 outline-none transition-all"
              value={detectorModel}
              onChange={(e) => setDetectorModel(e.target.value)}
            >
              <option value="yolov8n.pt">yolov8n.pt (Nano — high speed)</option>
              <option value="yolov8s.pt">yolov8s.pt (Small — high accuracy)</option>
            </select>
          </div>

          <div>
            <div className="flex justify-between text-xs font-medium text-zinc-500 mb-2">
              <span>Frame stride</span>
              <span className="text-orange-600 font-semibold tabular-nums">{frameStride}x</span>
            </div>
            <input
              type="range"
              min="1"
              max="15"
              value={frameStride}
              onChange={(e) => setFrameStride(Number(e.target.value))}
              className="w-full accent-orange-500 cursor-pointer h-1.5 bg-zinc-100 rounded-lg mt-3"
            />
            <div className="flex justify-between text-[10px] text-zinc-400 mt-1.5">
              <span>1x (every frame)</span>
              <span>15x (fast sample)</span>
            </div>
          </div>
        </div>

        <button
          className="w-full bg-zinc-900 hover:bg-zinc-800 text-white font-semibold text-sm py-3 px-4 rounded-xl shadow-sm transition-all flex items-center justify-center gap-2 disabled:opacity-50"
          disabled={busy === "upload"}
          type="submit"
        >
          {busy === "upload" ? <RefreshIcon className="w-4 h-4 animate-spin" /> : <PlayIcon className="w-4 h-4" />}
          {busy === "upload" ? "Uploading & processing…" : "Launch tracking pipeline"}
        </button>
      </form>

      {/* Active Job Progress Card */}
      {activeJob && (
        <div className="app-surface p-5 md:p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-zinc-100 pb-3.5">
            <div className="flex items-center gap-2 min-w-0 pr-2">
              <CpuIcon className="w-4 h-4 text-zinc-400 shrink-0" />
              <span className="text-sm font-semibold text-zinc-900 truncate" title={activeJob.uploaded_filename || activeJob.filename || activeJob.source_name}>
                {activeJob.uploaded_filename || activeJob.filename || sourceLabel(activeJob.source_name, "Surveillance Stream")}
              </span>
            </div>
            <span
              className={`text-xs font-medium px-3 py-1 rounded-full shrink-0 ${
                activeJob.status === "completed"
                  ? "bg-emerald-50 text-emerald-700"
                  : activeJob.status === "processing"
                  ? "bg-amber-50 text-amber-700"
                  : "bg-zinc-100 text-zinc-500"
              }`}
            >
              {activeJob.status}
            </span>
          </div>

          <div className="space-y-2">
            <div className="w-full bg-zinc-100 h-2 rounded-full overflow-hidden">
              <div
                className="bg-gradient-to-r from-amber-400 to-orange-500 h-full rounded-full transition-all duration-300"
                style={{ width: `${Math.max(activeJob.progress?.percent || 0, 4)}%` }}
              />
            </div>

            <div className="flex justify-between text-xs text-zinc-500 font-medium tabular-nums">
              <span>{Math.round(activeJob.progress?.percent || 0)}% completed</span>
              <span>{activeJob.progress?.frames_processed || 0} frames sampled</span>
            </div>
          </div>

          <div className="flex flex-wrap gap-2.5 pt-1 border-t border-zinc-100/80">
            {activeJob.status === "completed" && (
              <Link
                href={`/surveillance?jobId=${activeJob.job_id}`}
                className="flex-1 py-2.5 px-4 rounded-xl bg-orange-600 hover:bg-orange-700 text-white font-medium text-sm transition-all text-center flex items-center justify-center gap-2 shadow-sm"
              >
                Watch in Surveillance Player →
              </Link>
            )}
            {["queued", "running", "processing"].includes(activeJob.status) && (
              <button
                type="button"
                className="flex-1 py-2.5 px-4 rounded-xl bg-rose-50 hover:bg-rose-100 text-rose-700 font-medium text-sm transition-all"
                onClick={() => updateJob(activeJob.job_id, "cancel")}
              >
                Cancel job
              </button>
            )}
            {["completed", "failed"].includes(activeJob.status) && (
              <button
                type="button"
                className="py-2.5 px-4 rounded-xl bg-zinc-100 hover:bg-zinc-200 text-zinc-700 font-medium text-sm transition-all"
                onClick={() => updateJob(activeJob.job_id, "retry")}
              >
                Retry job
              </button>
            )}
            {["completed", "failed", "canceled"].includes(activeJob.status) && (
              <button
                type="button"
                className="py-2.5 px-4 rounded-xl bg-white border border-zinc-200/80 hover:border-rose-200 hover:bg-rose-50 text-rose-600 font-medium text-sm transition-all flex items-center justify-center gap-1.5"
                onClick={() => deleteJob(activeJob.job_id)}
              >
                <TrashIcon className="w-3.5 h-3.5" />
                Delete job
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
