"use client";

import React from "react";
import { TrashIcon } from "./Icons";

export default function ResetConfirmationModal({ isOpen = true, onClose, onCancel, onConfirm, isResetting, busy }) {
  if (!isOpen) return null;
  const close = onClose || onCancel;
  const resetting = isResetting || busy;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-900/50 backdrop-blur-sm p-4">
      <div className="bg-white rounded-[1.25rem] max-w-md w-full p-5 md:p-6 shadow-2xl border border-zinc-200/80">
        <div className="flex items-center gap-3.5 mb-4">
          <div className="w-10 h-10 rounded-2xl bg-rose-50 flex items-center justify-center shrink-0">
            <TrashIcon className="w-5 h-5 text-rose-600" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-zinc-900">Reset all data &amp; past video archives?</h3>
            <p className="text-xs text-zinc-500 mt-0.5">This action is permanent and cannot be undone.</p>
          </div>
        </div>

        <div className="bg-zinc-50 rounded-2xl p-4 border border-zinc-200/80 mb-5 text-xs text-zinc-600 space-y-1.5 leading-relaxed">
          <p className="font-medium text-zinc-900 mb-1">The following will be completely deleted:</p>
          <p>• All uploaded past video archives &amp; processed video outputs</p>
          <p>• Track memories, bounding-box sightings &amp; episode timelines</p>
          <p>• Saved person crops &amp; evidence screenshots</p>
          <p>• Vector embeddings &amp; search observations</p>
          <p>• Exported MP4 evidence video clips &amp; active job histories</p>
        </div>

        <div className="flex items-center justify-end gap-2.5">
          <button
            type="button"
            className="px-4 py-2 rounded-xl border border-zinc-200/80 text-zinc-700 hover:bg-zinc-50 font-medium text-sm transition-all"
            onClick={close}
          >
            Cancel
          </button>
          <button
            type="button"
            className="px-4 py-2 rounded-xl bg-rose-600 hover:bg-rose-700 text-white font-medium text-sm shadow-sm transition-all flex items-center gap-1.5 disabled:opacity-50"
            disabled={resetting}
            onClick={onConfirm}
          >
            <TrashIcon className="w-3.5 h-3.5" />
            {resetting ? "Resetting…" : "Yes, reset everything"}
          </button>
        </div>
      </div>
    </div>
  );
}
