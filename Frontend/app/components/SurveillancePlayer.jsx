"use client";

import React, { useRef, useState } from "react";
import Link from "next/link";
import { mediaUrl, ViewfinderCorners } from "./UIHelpers";

export default function SurveillancePlayer({ videoUrl, rawInputUrl, activeJob, latestRunResult, focusedVideo, setFocusedVideo }) {
  const videoRef = useRef(null);
  const containerRef = useRef(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [objectFit, setObjectFit] = useState("contain");
  const [useRaw, setUseRaw] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const rawAnnotated = videoUrl || focusedVideo || activeJob?.output_url || activeJob?.result?.output_url || activeJob?.result?.output_video_url || latestRunResult?.output_url || latestRunResult?.output_video_url || latestRunResult?.annotated_video_url || latestRunResult?.video_url;
  const rawSource = rawInputUrl || activeJob?.input_url || activeJob?.result?.input_url || activeJob?.result?.input_video_url || latestRunResult?.input_url || latestRunResult?.input_video_url || latestRunResult?.raw_video_url;

  const annotatedUrl = rawAnnotated ? mediaUrl(rawAnnotated) : "";
  const sourceUrl = rawSource ? mediaUrl(rawSource) : "";
  const currentSrc = useRaw && sourceUrl ? sourceUrl : annotatedUrl;

  const handlePlayPause = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
    } else {
      videoRef.current.play();
    }
  };


  const handleSpeedChange = (speed) => {
    setPlaybackRate(speed);
    if (videoRef.current) {
      videoRef.current.playbackRate = speed;
    }
  };

  const handleStepFrame = (frames) => {
    if (!videoRef.current) return;
    videoRef.current.pause();
    videoRef.current.currentTime = Math.max(0, videoRef.current.currentTime + frames * (1 / 30));
  };

  const handleDownloadVideo = () => {
    if (!currentSrc) return;
    const a = document.createElement("a");
    a.href = currentSrc;
    a.download = currentSrc.split("/").pop() || "surveillance_video.mp4";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleToggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch((err) => console.error(err));
    } else {
      document.exitFullscreen().catch((err) => console.error(err));
    }
  };

  const handleTakeSnapshot = () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const a = document.createElement("a");
    a.href = canvas.toDataURL("image/png");
    a.download = `surveillance_snapshot_${Date.now()}.png`;
    a.click();
  };

  const formatTime = (seconds) => {
    if (!seconds || isNaN(seconds)) return "00:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  };

  return (
    <div ref={containerRef} className="relative bg-zinc-950 text-white rounded-3xl overflow-hidden border border-zinc-800/60 shadow-2xl transition-all">
      {/* Top Header Controls Bar */}
      <div className="flex flex-wrap items-center justify-between px-4 py-3 bg-zinc-900/90 backdrop-blur-md border-b border-white/10 gap-3">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-white/5 border border-white/10">
            <span className="relative flex w-2 h-2">
              <span className={`absolute inline-flex h-full w-full animate-ping rounded-full ${currentSrc ? "bg-emerald-400" : "bg-zinc-500"} opacity-60`} />
              <span className={`relative inline-flex w-2 h-2 rounded-full ${currentSrc ? "bg-emerald-400" : "bg-zinc-400"}`} />
            </span>
            <span className="text-xs font-semibold tracking-wide text-zinc-200">
              {currentSrc ? (useRaw ? "Raw Stream Feed" : "Annotated YOLO Output") : "Standby Mode"}
            </span>
          </div>

          {sourceUrl && (
            <button
              type="button"
              onClick={() => setUseRaw(!useRaw)}
              className="px-3 py-1 rounded-full text-xs font-medium bg-indigo-500/20 hover:bg-indigo-500/30 text-indigo-300 border border-indigo-500/30 transition-all cursor-pointer"
            >
              {useRaw ? "Show YOLO Bounding Boxes" : "Switch to Raw Stream"}
            </button>
          )}
        </div>

        {/* Speed Selector */}
        <div className="flex items-center gap-1.5">
          <span className="text-[11px] text-zinc-400 font-medium mr-1">Speed</span>
          {[0.5, 1, 1.5, 2].map((speed) => (
            <button
              key={speed}
              type="button"
              onClick={() => handleSpeedChange(speed)}
              className={`px-2.5 py-0.5 rounded-full text-[11px] font-semibold transition-all cursor-pointer ${
                playbackRate === speed ? "bg-white text-zinc-900 shadow-sm" : "bg-white/5 text-zinc-400 hover:bg-white/15 hover:text-zinc-200"
              }`}
            >
              {speed}x
            </button>
          ))}
        </div>
      </div>

      {/* Main Player Display Area */}
      <div className="relative aspect-video bg-gradient-to-br from-zinc-950 via-slate-900 to-zinc-950 flex items-center justify-center group overflow-hidden">
        {currentSrc ? (
          <>
            <video
              ref={videoRef}
              key={currentSrc}
              src={currentSrc}
              className={`w-full h-full transition-all ${objectFit === "cover" ? "object-cover" : "object-contain"}`}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              onTimeUpdate={() => setCurrentTime(videoRef.current?.currentTime || 0)}
              onLoadedMetadata={() => setDuration(videoRef.current?.duration || 0)}
              onClick={handlePlayPause}
            />

            <ViewfinderCorners inset="inset-4" color="border-white/30" thickness="border-[1.5px]" />

            {/* Bottom Floating Glass Control Bar */}
            <div className="absolute inset-x-0 bottom-0 p-4 bg-gradient-to-t from-black/90 via-black/50 to-transparent opacity-95 group-hover:opacity-100 transition-opacity space-y-3">
              {/* Custom Scrubber Line */}
              <div className="relative w-full flex items-center group/scrubber">
                <input
                  type="range"
                  min="0"
                  max={duration || 100}
                  value={currentTime}
                  onChange={(e) => {
                    const val = Number(e.target.value);
                    setCurrentTime(val);
                    if (videoRef.current) videoRef.current.currentTime = val;
                  }}
                  className="w-full h-1.5 bg-white/20 accent-indigo-500 rounded-full cursor-pointer hover:h-2 transition-all"
                />
              </div>

              {/* Toolbar Action Row */}
              <div className="flex items-center justify-between gap-3 text-xs">
                {/* Left Controls */}
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={handlePlayPause}
                    className="w-8 h-8 rounded-full bg-white hover:bg-zinc-200 text-zinc-900 flex items-center justify-center transition-all shadow-sm shrink-0 cursor-pointer"
                    title={isPlaying ? "Pause" : "Play"}
                  >
                    {isPlaying ? (
                      <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
                        <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
                      </svg>
                    ) : (
                      <svg className="w-4 h-4 fill-current ml-0.5" viewBox="0 0 24 24">
                        <path d="M8 5v14l11-7z" />
                      </svg>
                    )}
                  </button>

                  <div className="flex items-center gap-1 bg-white/10 rounded-full p-1 border border-white/10">
                    <button
                      type="button"
                      onClick={() => handleStepFrame(-1)}
                      className="px-2 py-0.5 rounded-full hover:bg-white/20 text-zinc-300 text-[11px] font-semibold transition-colors cursor-pointer"
                      title="Step backward 1 frame"
                    >
                      -1f
                    </button>
                    <span className="text-white/20 text-[10px]">|</span>
                    <button
                      type="button"
                      onClick={() => handleStepFrame(1)}
                      className="px-2 py-0.5 rounded-full hover:bg-white/20 text-zinc-300 text-[11px] font-semibold transition-colors cursor-pointer"
                      title="Step forward 1 frame"
                    >
                      +1f
                    </button>
                  </div>

                  <span className="text-xs font-mono font-medium text-zinc-300 tabular-nums">
                    {formatTime(currentTime)} <span className="text-zinc-500">/</span> {formatTime(duration)}
                  </span>
                </div>

                {/* Right Controls */}
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setObjectFit(objectFit === "contain" ? "cover" : "contain")}
                    className="px-3 py-1.5 rounded-full bg-white/10 hover:bg-white/20 text-zinc-200 text-xs font-medium border border-white/10 transition-colors cursor-pointer"
                  >
                    {objectFit === "contain" ? "Fit" : "Fill"}
                  </button>

                  <button
                    type="button"
                    onClick={handleTakeSnapshot}
                    className="px-3 py-1.5 rounded-full bg-white/10 hover:bg-white/20 text-zinc-200 text-xs font-medium border border-white/10 transition-colors flex items-center gap-1.5 cursor-pointer"
                    title="Take frame snapshot"
                  >
                    <svg className="w-3.5 h-3.5 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                      <circle cx="12" cy="13" r="3" />
                    </svg>
                    <span>Snapshot</span>
                  </button>


                  <button
                    type="button"
                    onClick={handleDownloadVideo}
                    className="px-3 py-1.5 rounded-full bg-indigo-600/80 hover:bg-indigo-600 text-white text-xs font-semibold border border-indigo-400/40 transition-colors flex items-center gap-1.5 cursor-pointer shadow-sm"
                    title="Download video"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    <span>Download</span>
                  </button>

                  <button
                    type="button"
                    onClick={handleToggleFullscreen}
                    className="p-1.5 rounded-full bg-white/10 hover:bg-white/20 text-zinc-200 border border-white/10 transition-colors cursor-pointer"
                    title="Toggle Fullscreen"
                  >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                    </svg>
                  </button>
                </div>
              </div>
            </div>
          </>
        ) : (
          /* Empty / Standby Glass Canvas */
          <div className="text-center p-8 max-w-md mx-auto space-y-4">
            <div className="relative w-16 h-16 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center mx-auto shadow-inner">
              <span className="w-3 h-3 rounded-full bg-indigo-500/80 animate-ping absolute" />
              <svg className="w-8 h-8 text-indigo-400 relative z-10" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            </div>

            <div className="space-y-1.5">
              <h3 className="text-base font-semibold text-white">No Video Feed Selected</h3>
              <p className="text-xs text-zinc-400 leading-relaxed font-normal">
                Select a processed surveillance feed from the dropdown or launch a new video stream via the pipeline.
              </p>
            </div>

            <div className="pt-2 flex items-center justify-center gap-3">
              <Link
                href="/pipeline"
                className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs shadow-sm transition-all"
              >
                Upload Video Stream →
              </Link>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
