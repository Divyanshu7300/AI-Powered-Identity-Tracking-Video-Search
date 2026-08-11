"use client";

import React from "react";

export const DEFAULT_TEXT_QUERY = "person wearing blue clothing near center middle";
export const PLACEHOLDER_IMAGE = "data:image/gif;base64,R0lGODlhAQABAAAAACw=";

export const PRESET_QUERIES = [
  "person wearing blue clothing",
  "person with backpack",
  "subject wearing dark jacket",
  "person in light shirt",
];

export function sourceLabel(label, fallback = "Uploaded video") {
  const value = String(label || "");
  return /^[a-f0-9]{24,}(?:-[a-f0-9]{12})?$/i.test(value) ? fallback : value || fallback;
}

export function mediaUrl(path) {
  if (!path) return PLACEHOLDER_IMAGE;
  if (/^https?:\/\//i.test(path)) return path;
  return `/api${path.startsWith("/") ? path : `/${path}`}`;
}

/** Signature motif: four corner brackets, like a detector's bounding box. */
export function ViewfinderCorners({ inset = "inset-2", color = "border-indigo-400", thickness = "border-[2.5px]" }) {
  const arm = "absolute w-3 h-3";
  return (
    <div className={`pointer-events-none absolute ${inset}`} aria-hidden="true">
      <span className={`${arm} top-0 left-0 border-t border-l ${thickness} ${color} rounded-tl-[3px]`} />
      <span className={`${arm} top-0 right-0 border-t border-r ${thickness} ${color} rounded-tr-[3px]`} />
      <span className={`${arm} bottom-0 left-0 border-b border-l ${thickness} ${color} rounded-bl-[3px]`} />
      <span className={`${arm} bottom-0 right-0 border-b border-r ${thickness} ${color} rounded-br-[3px]`} />
    </div>
  );
}