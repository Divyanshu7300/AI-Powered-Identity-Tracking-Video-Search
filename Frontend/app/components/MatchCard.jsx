"use client";

import Image from "next/image";
import React from "react";
import { ClockIcon, EyeIcon, UserIcon, PlayIcon } from "./Icons";
import { mediaUrl, ViewfinderCorners } from "./UIHelpers";

export default function MatchCard({ match, onInspect, onExportClip, isExporting, busy }) {
  const rawScore = match.score ?? match.similarity ?? 0;
  const pctScore = typeof rawScore === "number" ? Math.round(rawScore > 1 ? rawScore : rawScore * 100) : 0;

  const isHighMatch = pctScore >= 70;
  const isMedMatch = pctScore >= 45 && pctScore < 70;

  const badgeColor = isHighMatch
    ? "bg-emerald-50 text-emerald-700"
    : isMedMatch
    ? "bg-amber-50 text-amber-700"
    : "bg-zinc-100 text-zinc-500";

  const badgeText = isHighMatch ? "Confirmed match" : isMedMatch ? "Possible match" : "Potential match";

  const startTime = match.first_seen_timestamp_seconds ?? match.timestamp_seconds;
  const endTime = match.last_seen_timestamp_seconds ?? (startTime != null ? startTime + (match.visible_duration_seconds || 2) : null);

  let timeStr = "";
  if (startTime != null && endTime != null) {
    timeStr = `${startTime.toFixed(1)}s – ${endTime.toFixed(1)}s`;
  } else if (startTime != null) {
    timeStr = `@ ${startTime.toFixed(1)}s`;
  }

  const frameStr = match.first_frame && match.last_frame ? `Frames ${match.first_frame}–${match.last_frame}` : "";

  return (
    <div className="bg-white border border-zinc-200/80 rounded-2xl p-4 space-y-3 shadow-sm hover:shadow-md transition-all">
      {/* Top Header Row */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-indigo-50 text-indigo-700 font-semibold text-xs">
            <UserIcon className="w-3.5 h-3.5" />
            #{match.track_id}
          </span>

          <span className={`inline-flex items-center text-[10px] font-medium px-2.5 py-1 rounded-full ${badgeColor}`}>
            {badgeText}
          </span>
        </div>

        <div className="text-right">
          <span className="text-sm font-bold text-indigo-600 tabular-nums">{pctScore}%</span>
          <span className="text-[9px] text-zinc-400 block leading-none tracking-wide uppercase">Relevance</span>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex gap-3.5 items-start">
        {/* Person Crop Thumbnail with Viewfinder */}
        <div
          className="relative w-16 h-20 bg-zinc-50 border border-zinc-200/80 rounded-xl overflow-hidden flex items-center justify-center shrink-0 group cursor-pointer"
          onClick={() => onInspect(match)}
          title="Click to inspect"
        >
          <Image
            alt={`Subject ${match.track_id} crop`}
            src={mediaUrl(match.frame_url || match.best_crop_url || match.crop_url)}
            width={64}
            height={80}
            unoptimized
            className="w-full h-full object-contain bg-zinc-50 p-0.5 group-hover:scale-110 transition-transform duration-300"
          />
          <div className="absolute inset-0 bg-zinc-900/0 group-hover:bg-zinc-900/40 transition-colors flex items-center justify-center opacity-0 group-hover:opacity-100">
            <EyeIcon className="w-4 h-4 text-white drop-shadow" />
          </div>
          <ViewfinderCorners inset="inset-1" color="border-indigo-400" thickness="border-[1.5px]" />
        </div>

        {/* Details Column */}
        <div className="flex-1 min-w-0 space-y-2">
          {/* Source & Timestamps */}
          <div className="flex flex-wrap items-center gap-2 text-xs">
            {timeStr && (
              <span className="inline-flex items-center gap-1 font-medium text-zinc-600 bg-zinc-50 border border-zinc-200/80 px-2 py-0.5 rounded-lg">
                <ClockIcon className="w-3 h-3 text-zinc-400" />
                <span className="tabular-nums">{timeStr}</span>
              </span>
            )}
            {frameStr && <span className="text-zinc-400 text-[11px] tabular-nums">{frameStr}</span>}
          </div>

          {/* Caption */}
          {match.caption && (
            <p className="text-xs text-zinc-600 line-clamp-2 leading-relaxed bg-zinc-50 border border-zinc-200/80 p-2 rounded-xl">
              &quot;{match.caption}&quot;
            </p>
          )}

          {/* Actions */}
          <div className="flex items-center gap-2 pt-0.5">
            <button
              type="button"
              onClick={() => onInspect(match)}
              className="px-3 py-1 rounded-full bg-white hover:bg-zinc-50 border border-zinc-200/80 hover:border-zinc-300 text-xs text-zinc-700 font-medium transition-all"
            >
              Inspect
            </button>
            {onExportClip && (
              <button
                type="button"
                disabled={isExporting || Boolean(busy)}
                onClick={() => onExportClip(match.memory_id || match.track_id, match.label || `Track #${match.track_id}`)}
                className="px-3 py-1 rounded-full bg-indigo-50 hover:bg-indigo-100 text-indigo-700 font-medium text-xs transition-all disabled:opacity-50 flex items-center gap-1"
              >
                <PlayIcon className="w-3 h-3" />
                Export clip
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
