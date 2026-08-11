"use client";

import Image from "next/image";
import React from "react";
import { PlayIcon, UserIcon } from "./Icons";
import { mediaUrl, sourceLabel, ViewfinderCorners } from "./UIHelpers";

export default function InspectModal({ inspectTrack, track, onClose, onExportClip, onWatchClip, isExporting, busy }) {
  const subject = inspectTrack || track;
  if (!subject) return null;

  return (
    <div className="fixed inset-0 bg-zinc-950/40 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div
        className="bg-white border border-zinc-200/80 rounded-3xl max-w-xl w-full p-6 shadow-2xl space-y-4 max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-zinc-100 pb-3.5">
          <h3 className="text-sm font-semibold text-zinc-900 flex items-center gap-2">
            <UserIcon className="w-4 h-4 text-zinc-500" />
            Subject #{subject.track_id}
          </h3>
          <button
            type="button"
            className="w-6 h-6 rounded-full flex items-center justify-center text-zinc-400 hover:text-zinc-700 hover:bg-zinc-100 transition-colors"
            onClick={onClose}
          >
            ✕
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {/* Non-Cropped High-Res Feature Crop */}
          <div className="relative w-full h-64 bg-zinc-50 border border-zinc-200/80 rounded-2xl overflow-hidden flex items-center justify-center">
            <Image
              alt="High res crop"
              src={mediaUrl(subject.evidence_url || subject.best_crop_url || subject.crop_url)}
              width={260}
              height={240}
              unoptimized
              className="w-full h-full object-contain bg-zinc-50 p-1"
            />
            <ViewfinderCorners inset="inset-2" color="border-indigo-400" />
          </div>

          <div className="space-y-2 text-sm">
            <div className="p-3 rounded-2xl bg-zinc-50 border border-zinc-200/80 flex justify-between items-center">
              <span className="text-zinc-500 text-xs font-medium">Source video</span>
              <span className="font-medium text-zinc-900 text-sm truncate max-w-[140px]">
                {sourceLabel(subject.source_label || subject.source_name)}
              </span>
            </div>
            <div className="p-3 rounded-2xl bg-zinc-50 border border-zinc-200/80 flex justify-between items-center">
              <span className="text-zinc-500 text-xs font-medium">Frame range</span>
              <span className="font-mono font-semibold text-zinc-900 text-sm tabular-nums">
                {subject.first_frame} — {subject.last_frame}
              </span>
            </div>
            <div className="p-3 rounded-2xl bg-zinc-50 border border-zinc-200/80 flex justify-between items-center">
              <span className="text-zinc-500 text-xs font-medium">Visible time</span>
              <span className="font-mono font-semibold text-zinc-900 text-sm tabular-nums">
                {subject.visible_duration_seconds ?? subject.duration_frames}s
              </span>
            </div>
            <div className="p-3 rounded-2xl bg-zinc-50 border border-zinc-200/80 flex justify-between items-center">
              <span className="text-zinc-500 text-xs font-medium">Peak confidence</span>
              <span className="font-mono font-bold text-indigo-600 text-sm tabular-nums">
                {Math.round((subject.best_confidence || 0) * 100)}%
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 pt-3.5 border-t border-zinc-100">
          <button
            type="button"
            className="px-4 py-2 rounded-full bg-white border border-zinc-200/80 text-zinc-700 font-medium text-xs hover:bg-zinc-50 transition-all cursor-pointer"
            onClick={onClose}
          >
            Close
          </button>
          <button
            type="button"
            className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium text-xs py-2 px-4 rounded-full shadow-sm transition-all flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
            disabled={Boolean(busy)}
            onClick={() => {
              const memoryId = subject.memory_id || subject.track_id;
              if (onWatchClip) {
                onWatchClip(memoryId, `Subject #${subject.track_id}`);
              } else {
                const target = subject.job_id || subject.source_name || "";
                window.location.href = `/surveillance?jobId=${encodeURIComponent(target)}`;
              }
              onClose();
            }}
          >
            <PlayIcon className="w-3.5 h-3.5 text-white" />
            <span>Watch</span>
          </button>
          <button
            type="button"
            className="bg-zinc-900 hover:bg-zinc-800 text-white font-medium text-xs py-2 px-4 rounded-full shadow-sm transition-all flex items-center gap-1.5 disabled:opacity-50 cursor-pointer"
            disabled={isExporting || Boolean(busy)}
            onClick={() => {
              onExportClip(subject.memory_id || subject.track_id, `Subject #${subject.track_id}`);
            }}
          >
            <svg className="w-3.5 h-3.5 text-indigo-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            <span>Download</span>
          </button>
        </div>
      </div>
    </div>
  );
}
