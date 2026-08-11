"use client";

import Image from "next/image";
import { useMemo, useState } from "react";
import { UserIcon } from "./Icons";
import { mediaUrl } from "./UIHelpers";

const PAGE_SIZE = 24;

export default function TrackDetections({ tracks, onInspect, onExportClip, busy, title = "Detected subjects", headerAction }) {
  const [page, setPage] = useState(1);
  const totalPages = Math.max(1, Math.ceil(tracks.length / PAGE_SIZE));
  const currentPage = Math.min(page, totalPages);
  const visibleTracks = useMemo(
    () => tracks.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE),
    [currentPage, tracks]
  );

  return (
    <section className="app-surface p-5 md:p-6 space-y-4">
      <div className="flex items-center justify-between border-b border-zinc-100 pb-3.5">
        <h2 className="text-sm font-semibold text-zinc-900 flex items-center gap-2 min-w-0 pr-2">
          <UserIcon className="w-4 h-4 text-zinc-400 shrink-0" />
          <span className="truncate" title={title}>{title}</span>
        </h2>
        <div className="flex items-center gap-2.5 shrink-0">
          <span className="text-xs font-medium bg-zinc-100 text-zinc-600 px-2.5 py-1 rounded-full tabular-nums">
            {tracks.length} subjects
          </span>
          {headerAction}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 2xl:grid-cols-6 gap-3.5 max-h-[34rem] overflow-y-auto pr-1">
        {visibleTracks.map((track) => (
          <button
            key={track.memory_id}
            type="button"
            className="group p-3 rounded-2xl bg-white border border-zinc-200/80 hover:border-indigo-300 hover:shadow-md flex items-center gap-3 transition-all cursor-pointer text-left overflow-hidden min-w-0"
            onClick={() => onInspect(track)}
          >
            <div className="relative w-14 h-[72px] bg-zinc-950/5 border border-zinc-200/80 rounded-xl overflow-hidden flex items-center justify-center shrink-0">
              <Image
                alt="Subject crop"
                src={mediaUrl(track.best_crop_url || track.crop_url)}
                width={56}
                height={72}
                unoptimized
                className="w-full h-full object-contain p-0.5 group-hover:scale-105 transition-transform"
              />
            </div>
            <div className="flex-1 min-w-0 space-y-1">
              <div className="flex items-center justify-between gap-1.5 min-w-0">
                <span className="text-xs font-semibold text-zinc-900 truncate">
                  Subject #{track.track_id}
                </span>
                <span className="text-[10px] font-bold bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded-full shrink-0 tabular-nums">
                  {Math.round((track.best_confidence || 0) * 100)}%
                </span>
              </div>
              <div className="flex items-center justify-between text-[11px] font-medium pt-0.5">
                <span className="text-zinc-400 truncate tabular-nums">
                  {track.hits} detections
                </span>
                <span className="text-indigo-600 font-semibold group-hover:underline shrink-0 flex items-center gap-0.5">
                  Inspect →
                </span>
              </div>
            </div>
          </button>
        ))}
        {!tracks.length && (
          <div className="col-span-full py-8 text-center text-zinc-400 text-sm">
            No detections recorded for current video yet.
          </div>
        )}
      </div>

      {tracks.length > PAGE_SIZE && (
        <div className="flex items-center justify-between text-xs text-zinc-500 pt-1">
          <span>
            Showing {(currentPage - 1) * PAGE_SIZE + 1}–{Math.min(currentPage * PAGE_SIZE, tracks.length)} of {tracks.length}
          </span>
          <div className="flex gap-2">
            <button
              type="button"
              className="px-3 py-1 rounded-full border border-zinc-200/80 hover:bg-zinc-50 disabled:opacity-40 transition-colors cursor-pointer"
              disabled={currentPage === 1}
              onClick={() => setPage((value) => Math.max(1, value - 1))}
            >
              Previous
            </button>
            <button
              type="button"
              className="px-3 py-1 rounded-full border border-zinc-200/80 hover:bg-zinc-50 disabled:opacity-40 transition-colors cursor-pointer"
              disabled={currentPage === totalPages}
              onClick={() => setPage((value) => Math.min(totalPages, value + 1))}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
