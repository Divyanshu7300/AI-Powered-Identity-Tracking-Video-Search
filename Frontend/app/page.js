"use client";

import React from "react";
import Link from "next/link";
import {
  CpuIcon,
  SearchIcon,
  VideoIcon,
  LayersIcon,
  SparklesIcon,
  SettingsIcon,
  DashboardIcon,
  UploadIcon,
} from "./components/Icons";

const MODELS = [
  {
    name: "YOLOv8 Detection",
    badge: "Detection Engine",
    icon: CpuIcon,
    accent: "text-amber-600 bg-amber-50",
    desc: "Real-time person bounding box localization & frame sampling.",
  },
  {
    name: "DeepSORT Tracking",
    badge: "Track Memory",
    icon: LayersIcon,
    accent: "text-sky-600 bg-sky-50",
    desc: "Multi-frame trajectory association & keyframe crop indexing.",
  },
  {
    name: "SigLIP / CLIP Engine",
    badge: "768-dim Vision Vectors",
    icon: SparklesIcon,
    accent: "text-indigo-600 bg-indigo-50",
    desc: "Zero-shot text description search & visual crop Re-ID matching.",
  },
  {
    name: "LLM RAG & Evidence",
    badge: "Evidence RAG",
    icon: SearchIcon,
    accent: "text-emerald-600 bg-emerald-50",
    desc: "Natural language query breakdown & automated MP4 clip exports.",
  },
];

const MODULE_LINKS = [
  {
    href: "/dashboard",
    title: "Overview Dashboard",
    desc: "Platform health, system metrics & active pipeline jobs.",
    icon: DashboardIcon,
    accent: "text-indigo-600 bg-indigo-50",
  },
  {
    href: "/pipeline",
    title: "Ingestion Pipeline",
    desc: "Upload surveillance videos & configure YOLOv8 detection.",
    icon: CpuIcon,
    accent: "text-amber-600 bg-amber-50",
  },
  {
    href: "/search",
    title: "AI Search Engine",
    desc: "Query subjects via text descriptions or visual reference crops.",
    icon: SearchIcon,
    accent: "text-violet-600 bg-violet-50",
  },
  {
    href: "/surveillance",
    title: "Surveillance Player",
    desc: "Watch video feeds with live detection overlays & inspect tracks.",
    icon: VideoIcon,
    accent: "text-emerald-600 bg-emerald-50",
  },
  {
    href: "/tracks",
    title: "Track Library",
    desc: "Browse indexed person memories & export video evidence.",
    icon: LayersIcon,
    accent: "text-sky-600 bg-sky-50",
  },
  {
    href: "/settings",
    title: "Engine Settings",
    desc: "Toggle SigLIP vision engine & manage vector search indices.",
    icon: SettingsIcon,
    accent: "text-slate-600 bg-slate-100",
  },
];

export default function HomePage() {
  return (
    <div className="relative app-page w-full animate-fade-in">
      {/* Full-bleed ambient gradient wash */}
      <div className="pointer-events-none absolute -top-40 -right-20 w-[600px] h-[600px] rounded-full bg-gradient-to-br from-indigo-200/50 via-purple-200/35 to-transparent blur-3xl opacity-70" />

      <div className="relative app-container space-y-10">
        {/* Hero Section — Clean & Concise */}
        <section className="text-center pt-2 pb-2 space-y-5 max-w-3xl mx-auto">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-indigo-50 border border-indigo-200/80 shadow-xs">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75 animate-ping" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500" />
            </span>
            <span className="text-xs font-semibold tracking-wide text-indigo-700 uppercase">
              AI Identity Tracking &amp; Video Search
            </span>
          </div>

          <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-zinc-900 leading-tight">
            Multimodal Video Intelligence Engine
          </h1>

          <p className="text-sm sm:text-base text-zinc-600 leading-relaxed max-w-xl mx-auto font-normal">
            Processes surveillance streams using YOLOv8 person detection, indexes 768-dim SigLIP visual embeddings, and retrieves evidence via natural language or crop Re-ID.
          </p>

          {/* Direct Workspace CTAs */}
          <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
            <Link
              href="/dashboard"
              className="bg-zinc-900 hover:bg-zinc-800 text-white font-semibold text-sm px-5 py-2.5 rounded-xl shadow-sm transition-all flex items-center gap-2"
            >
              <span>Launch Workspace</span>
              <span className="text-zinc-400">→</span>
            </Link>
            <Link
              href="/pipeline"
              className="bg-white hover:bg-zinc-50 text-zinc-800 border border-zinc-200/80 font-medium text-sm px-4.5 py-2.5 rounded-xl shadow-xs transition-all flex items-center gap-2"
            >
              <UploadIcon className="w-4 h-4 text-amber-500" />
              <span>Video Pipeline</span>
            </Link>
            <Link
              href="/search"
              className="bg-white hover:bg-zinc-50 text-zinc-800 border border-zinc-200/80 font-medium text-sm px-4.5 py-2.5 rounded-xl shadow-xs transition-all flex items-center gap-2"
            >
              <SearchIcon className="w-4 h-4 text-indigo-500" />
              <span>AI Search</span>
            </Link>
          </div>
        </section>

        {/* AI Model Stack — 4 Compact Cards */}
        <section className="space-y-4">
          <div className="flex items-center justify-between border-b border-zinc-100 pb-3">
            <h2 className="text-base font-bold text-zinc-900">AI Engine Stack</h2>
            <span className="text-xs text-zinc-400 font-medium">YOLOv8 + DeepSORT + SigLIP + LLM RAG</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {MODELS.map((m, i) => {
              const Icon = m.icon;
              return (
                <div key={i} className="app-surface p-4 space-y-2.5 hover:shadow-md transition-shadow">
                  <div className="flex items-center justify-between">
                    <div className={`w-8 h-8 rounded-xl ${m.accent} flex items-center justify-center shrink-0`}>
                      <Icon className="w-4 h-4" />
                    </div>
                    <span className="text-[10px] font-semibold text-zinc-500 bg-zinc-100 px-2 py-0.5 rounded-full">
                      {m.badge}
                    </span>
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-zinc-900">{m.name}</h3>
                    <p className="text-xs text-zinc-500 mt-1 leading-relaxed font-normal">{m.desc}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* Workspace Modules Navigation */}
        <section className="space-y-4">
          <div className="border-b border-zinc-100 pb-3">
            <h2 className="text-base font-bold text-zinc-900">Platform Modules</h2>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {MODULE_LINKS.map((mod, i) => {
              const Icon = mod.icon;
              return (
                <Link
                  key={i}
                  href={mod.href}
                  className="app-surface p-4.5 hover:shadow-md hover:border-zinc-300 transition-all group flex items-start gap-3.5"
                >
                  <div className={`w-9 h-9 rounded-xl ${mod.accent} flex items-center justify-center shrink-0 mt-0.5`}>
                    <Icon className="w-4 h-4" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-1">
                      <h3 className="text-sm font-semibold text-zinc-900 group-hover:text-indigo-600 transition-colors truncate">
                        {mod.title}
                      </h3>
                      <span className="text-zinc-400 group-hover:text-zinc-900 group-hover:translate-x-0.5 transition-all text-xs font-bold shrink-0">
                        →
                      </span>
                    </div>
                    <p className="text-xs text-zinc-500 mt-1 leading-relaxed font-normal">{mod.desc}</p>
                  </div>
                </Link>
              );
            })}
          </div>
        </section>
      </div>
    </div>
  );
}
