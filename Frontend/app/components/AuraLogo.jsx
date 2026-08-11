"use client";

import React from "react";
import { ViewfinderCorners } from "./UIHelpers";

export default function AuraLogo({ compact = false }) {
  return (
    <div className="flex items-center gap-2.5 select-none">
      <div className="relative w-8 h-8 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-sm shadow-indigo-500/20">
        <span className="w-2 h-2 rounded-full bg-white" />
        <ViewfinderCorners inset="inset-1" color="border-white/70" thickness="border-[1.5px]" />
      </div>
      <div className={compact ? "hidden" : "hidden sm:block"}>
        <div className="text-sm font-semibold tracking-tight text-zinc-900 leading-tight">
          Aura <span className="text-zinc-400 font-normal text-[11px]">Identity tracking</span>
        </div>
      </div>
    </div>
  );
}
