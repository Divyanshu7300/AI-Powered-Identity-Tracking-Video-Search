"use client";

import Image from "next/image";
import { useMemo, useState } from "react";
import { UserIcon } from "./Icons";

const PAGE_SIZE = 24;
const PLACEHOLDER_IMAGE = "data:image/gif;base64,R0lGODlhAQABAAAAACw=";

function mediaUrl(path) {
  if (!path) return PLACEHOLDER_IMAGE;
  if (/^https?:\/\//i.test(path)) return path;
  return `/api${path.startsWith("/") ? path : `/${path}`}`;
}

export default function TrackDetections({ tracks, onInspect, onExportClip, busy }) {
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
        <h2 className="text-sm font-semibold text-zinc-900 flex items-center gap-2">
          <UserIcon className="w-4 h-4 text-zinc-400" />
          Detected subjects
        </h2>
        <span className="text-xs font-medium bg-zinc-100 text-zinc-600 px-2.5 py-1 rounded-full">{tracks.length} subjects</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3 max-h-[30rem] overflow-y-auto pr-1">
        {visibleTracks.map((track) => (
          <button
            key={track.memory_id}
            type="button"
            className="p-3 rounded-2xl bg-white border border-zinc-200/80 hover:border-indigo-200 hover:shadow-md flex gap-3.5 items-center transition-all text-left group"
            onClick={() => onInspect(track)}
          >
            <div className="relative w-14 h-[72px] bg-zinc-50 border border-zinc-200/80 rounded-xl overflow-hidden flex items-center justify-center shrink-0">
              <Image
                alt="Subject crop"
                src={mediaUrl(track.best_crop_url || track.crop_url)}
                width={56}
                height={72}
                unoptimized
                className="w-full h-full object-contain bg-zinc-50 p-0.5 group-hover:scale-105 transition-transform"
              />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between gap-1">
                <span className="text-xs font-semibold text-zinc-900 truncate">Subject #{track.track_id}</span>
                <span className="text-[10px] font-semibold bg-zinc-100 text-zinc-700 px-2 py-0.5 rounded-full">
                  {Math.round((track.best_confidence || 0) * 100)}%
                </span>
              </div>
              <p className="text-[11px] text-zinc-400 font-medium mt-1">{track.hits} detections</p>
              {onExportClip && (
                <span className="text-[11px] text-indigo-600 font-medium mt-1.5 inline-block group-hover:underline">
                  View details
                </span>
              )}
            </div>
          </button>
        ))}
        {!tracks.length && (
          <div className="col-span-full py-8 text-center text-zinc-400 text-sm">No detections recorded for current video yet.</div>
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
              className="px-3 py-1 rounded-full border border-zinc-200/80 hover:bg-zinc-50 disabled:opacity-40 transition-colors"
              disabled={currentPage === 1}
              onClick={() => setPage((value) => Math.max(1, value - 1))}
            >
              Previous
            </button>
            <button
              type="button"
              className="px-3 py-1 rounded-full border border-zinc-200/80 hover:bg-zinc-50 disabled:opacity-40 transition-colors"
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
