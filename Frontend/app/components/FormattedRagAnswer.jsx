"use client";

import React from "react";
import { ClockIcon, SparklesIcon } from "./Icons";

export default function FormattedRagAnswer({ text, provider, model, rag }) {
  const answer = text || (typeof rag === "string" ? rag : rag?.answer);
  const answerProvider = provider || rag?.provider;
  const answerModel = model || rag?.model;
  if (!answer) return null;

  const lines = answer.split("\n").flatMap((l) => l.split(" - ")).map((l) => l.trim()).filter(Boolean);
  const bulletItems = [];
  const tableRows = [];
  let trackIds = [];
  let timestampText = "";

  lines.forEach((line) => {
    if (/^track ids:/i.test(line) || /^matching track/i.test(line)) {
      const ids = line.match(/\d+/g);
      if (ids) trackIds = Array.from(new Set(ids));
      const timeMatch = line.match(/time range:\s*([\d\.\s\-]+seconds?)/i);
      if (timeMatch) timestampText = timeMatch[1].trim();
    } else if (/^time range:/i.test(line)) {
      const match = line.replace(/^time range:\s*/i, "").replace(/\*\*/g, "").trim();
      if (match) timestampText = match;
    } else {
      // Check if line contains structured observation pattern with (Track ID X, timestamp Y, confidence Z)
      const trackIdMatch = line.match(/Track ID\s*(\d+)/i);
      const tsMatch = line.match(/timestamp\s*([\d\.]+)/i);
      const confMatch = line.match(/confidence\s*([\d\.]+)/i);

      if (trackIdMatch) {
        const trackId = trackIdMatch[1];
        const timestamp = tsMatch ? tsMatch[1] : null;
        let confVal = confMatch ? parseFloat(confMatch[1]) : null;
        if (confVal != null && confVal <= 1) confVal = Math.round(confVal * 100);

        // Clean description text by stripping (Track ID...) suffix and bullet markers
        const desc = line
          .replace(/^[\*\-\•\d\.]+\s*/, "")
          .replace(/\(Track ID.*$/i, "")
          .replace(/\*\*/g, "")
          .trim();

        tableRows.push({
          trackId,
          description: desc,
          timestamp,
          confidence: confVal,
        });

        if (!trackIds.includes(trackId)) trackIds.push(trackId);
      } else {
        const cleanLine = line
          .replace(/^[\*\-\•\d\.]+\s*/, "")
          .replace(/\*\*/g, "")
          .replace(/^Track IDs?:\s*\d+(,\s*\d+)*/i, "")
          .trim();
        if (cleanLine && cleanLine.length > 3) {
          bulletItems.push(cleanLine);
        }
      }
    }
  });

  return (
    <div className="app-surface p-5 text-sm text-zinc-800 hover:shadow-md transition-all space-y-4">
      {/* Card Header */}
      <div className="flex items-center justify-between border-b border-zinc-100 pb-3.5">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center shrink-0">
            <SparklesIcon className="h-4 w-4" />
          </div>
          <div>
            <h4 className="font-semibold text-zinc-900 text-sm">AI intelligence summary</h4>
            <p className="text-xs text-zinc-400">Surveillance evidence grounding</p>
          </div>
        </div>
        <span className="text-xs font-medium text-indigo-700 bg-indigo-50 px-3 py-1 rounded-full shrink-0">
          {answerProvider === "groq" ? (answerModel || "Groq Llama 3.1") : "Evidence summary"}
        </span>
      </div>

      {/* Metadata Badges Bar */}
      {(trackIds.length > 0 || timestampText) && (
        <div className="flex flex-wrap items-center gap-2">
          {trackIds.length > 0 && (
            <div className="flex items-center gap-1.5 bg-zinc-50 border border-zinc-200/80 rounded-2xl px-3 py-1.5">
              <span className="text-[10px] uppercase font-medium tracking-wide text-zinc-400">Subjects</span>
              {trackIds.map((id) => (
                <span
                  key={id}
                  className="inline-flex items-center px-2 py-0.5 rounded-lg bg-indigo-50 text-indigo-700 font-semibold text-[11px]"
                >
                  #{id}
                </span>
              ))}
            </div>
          )}

          {timestampText && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-2xl bg-zinc-50 border border-zinc-200/80 text-zinc-600 text-xs font-medium">
              <ClockIcon className="w-3.5 h-3.5 text-zinc-400 shrink-0" />
              <span className="tabular-nums">{timestampText}</span>
            </div>
          )}
        </div>
      )}

      {/* Table View for Structured Observations */}
      {tableRows.length > 0 && (
        <div className="overflow-x-auto rounded-2xl border border-zinc-200/80">
          <table className="w-full text-left border-collapse text-sm">
            <thead>
              <tr className="bg-zinc-50 border-b border-zinc-200/80 text-[10px] uppercase font-medium text-zinc-400 tracking-wide">
                <th className="py-2.5 px-4">Subject</th>
                <th className="py-2.5 px-4">Visual observation</th>
                <th className="py-2.5 px-4">Timestamp</th>
                <th className="py-2.5 px-4 text-right">Confidence</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100">
              {tableRows.map((row, idx) => (
                <tr key={idx} className="hover:bg-zinc-50 transition-colors">
                  <td className="py-3 px-4 whitespace-nowrap">
                    <span className="inline-flex items-center px-2 py-0.5 rounded-lg bg-indigo-50 text-indigo-700 font-semibold text-[11px]">
                      #{row.trackId}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-zinc-700 leading-relaxed">{row.description}</td>
                  <td className="py-3 px-4 whitespace-nowrap font-mono text-xs text-zinc-500 tabular-nums">
                    {row.timestamp ? `${row.timestamp}s` : "—"}
                  </td>
                  <td className="py-3 px-4 whitespace-nowrap text-right font-mono font-semibold text-zinc-900 tabular-nums">
                    {row.confidence != null ? `${row.confidence}%` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Bullet Observations Fallback */}
      {bulletItems.length > 0 && (
        <div className="space-y-2">
          {bulletItems.map((item, idx) => (
            <div
              key={idx}
              className="flex items-start gap-3 p-3 rounded-2xl bg-zinc-50 border border-zinc-200/80 hover:border-indigo-200 transition-colors"
            >
              <div className="w-4 h-4 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center shrink-0 mt-0.5 text-[10px] font-bold">
                ✓
              </div>
              <p className="text-sm text-zinc-700 leading-relaxed">{item}</p>
            </div>
          ))}
        </div>
      )}

      {tableRows.length === 0 && bulletItems.length === 0 && (
        <div className="p-3 rounded-2xl bg-zinc-50 border border-zinc-200/80 text-sm text-zinc-700 leading-relaxed">
          {answer.replace(/\*\*/g, "")}
        </div>
      )}
    </div>
  );
}
