"use client";

import React from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@clerk/nextjs";
import { useRouter } from "next/navigation";
import { loadDashboard as fetchDashboard, fetchCurrentIdentity, getUsername } from "../lib/api";
import {
  CpuIcon,
  SearchIcon,
  VideoIcon,
  LayersIcon,
  ClockIcon,
  EyeIcon,
} from "../components/Icons";

/* Module config — one gradient per function, not decorative: amber = the
   pipeline actively processing, indigo = AI/search intelligence,
   emerald = live surveillance, sky = stored data. */
const MODULES = [
  {
    href: "/pipeline",
    icon: CpuIcon,
    title: "Ingestion Pipeline",
    desc: "Upload surveillance videos, pick a YOLOv8 variant, track job progress end to end.",
    cta: "Open pipeline",
    gradient: "from-amber-400 to-orange-500",
    glow: "group-hover:shadow-orange-200",
    big: true,
  },
  {
    href: "/search",
    icon: SearchIcon,
    title: "AI Search",
    desc: "Text queries and visual crop Re-ID lookup, answered with evidence.",
    cta: "Launch search",
    gradient: "from-indigo-400 to-violet-500",
    glow: "group-hover:shadow-indigo-200",
  },
  {
    href: "/surveillance",
    icon: VideoIcon,
    title: "Surveillance Player",
    desc: "Watch feeds with live bounding boxes, export evidence clips.",
    cta: "Open player",
    gradient: "from-emerald-400 to-teal-500",
    glow: "group-hover:shadow-emerald-200",
  },
  {
    href: "/tracks",
    icon: LayersIcon,
    title: "Track Database",
    desc: "Browse indexed subject memories and their crops.",
    cta: "View database",
    gradient: "from-sky-400 to-blue-500",
    glow: "group-hover:shadow-sky-200",
  },
];

export default function DashboardPage() {
  const router = useRouter();
  const { isLoaded, isSignedIn } = useAuth();
  const [nativeUser, setNativeUser] = React.useState("");

  React.useEffect(() => {
    fetchCurrentIdentity().then((identity) => {
      if (identity?.username) setNativeUser(identity.username);
    });
  }, []);

  const effectiveIsSignedIn = Boolean(isSignedIn || nativeUser || getUsername());

  const { data: dashboardData, error: dashboardError } = useQuery({
    queryKey: ["dashboard"],
    queryFn: fetchDashboard,
    staleTime: 3_000,
    refetchInterval: 5_000,
    enabled: effectiveIsSignedIn,
  });

  const dashboard = dashboardData?.dashboard || null;
  const overview = dashboard?.overview || {};
  const semanticStatus = dashboard?.semantic_status || {};
  const jobs = dashboardData?.jobs || [];
  const tracks = dashboardData?.tracks || [];
  const health = dashboardError ? "offline" : dashboardData?.health || "checking";
  const isOnline = health === "online";

  const requiresLogin = isLoaded && (!effectiveIsSignedIn || /login|token/i.test(dashboardError?.message || ""));
  React.useEffect(() => {
    if (requiresLogin) router.replace("/login");
  }, [requiresLogin, router]);

  if (requiresLogin) return null;

  const doneJobs = jobs.filter((j) => j.status === "completed").length;
  const activeJobs = jobs.filter((j) => ["processing", "queued"].includes(j.status)).length;
  const totalJobs = overview.total_jobs ?? jobs.length ?? 0;
  const donePct = totalJobs ? Math.round((doneJobs / totalJobs) * 100) : 0;
  const activePct = totalJobs ? Math.round((activeJobs / totalJobs) * 100) : 0;

  return (
    <div className="relative app-page w-full animate-fade-in">
      {/* full-bleed ambient wash — tuned to overview accent */}
      <div className="pointer-events-none absolute -top-40 -right-20 w-[600px] h-[600px] rounded-full bg-gradient-to-br from-indigo-200/50 via-sky-200/35 to-transparent blur-3xl opacity-70" />

      <div className="relative app-container space-y-6">
        {/* Header */}
        <header className="app-page-header">
          <div>
            <h1 className="app-page-title">Overview</h1>
            <p className="app-page-description">Your video intelligence workspace.</p>
          </div>
          <div className="flex items-center gap-1.5 bg-white border border-zinc-200/80 rounded-xl pl-2.5 pr-3.5 py-1.5 shadow-sm">
            <span className="relative flex h-2 w-2">
              <span
                className={`absolute inline-flex h-full w-full rounded-full ${
                  isOnline ? "bg-emerald-400" : "bg-rose-400"
                } opacity-75 animate-ping`}
              />
              <span className={`relative inline-flex rounded-full h-2 w-2 ${isOnline ? "bg-emerald-500" : "bg-rose-500"}`} />
            </span>
            <span className="text-xs font-medium text-zinc-600">{isOnline ? "All systems live" : "Offline"}</span>
          </div>
        </header>

        {/* Metric tiles */}
        <section className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="app-surface p-5 hover:shadow-md transition-shadow">
            <p className="text-xs font-medium text-zinc-400 mb-2">Indexed profiles</p>
            <p className="text-3xl font-semibold text-zinc-900 tracking-tight tabular-nums">
              {overview.indexed_track_memories ?? tracks.length ?? 0}
            </p>
            <p className="text-xs text-zinc-400 mt-1">Person memories stored</p>
          </div>

          <div className="app-surface p-5 hover:shadow-md transition-shadow">
            <p className="text-xs font-medium text-zinc-400 mb-2">Pipeline jobs</p>
            <p className="text-3xl font-semibold text-zinc-900 tracking-tight tabular-nums">{totalJobs}</p>
            <div className="flex gap-1 mt-3 h-1.5 rounded-full overflow-hidden bg-zinc-100">
              <div className="bg-emerald-500" style={{ width: `${donePct}%` }} />
              <div className="bg-amber-400" style={{ width: `${activePct}%` }} />
            </div>
            <p className="text-xs text-zinc-400 mt-1.5">{doneJobs} done · {activeJobs} active</p>
          </div>

          <div className="app-surface p-5 hover:shadow-md transition-shadow">
            <p className="text-xs font-medium text-zinc-400 mb-2">Vision search</p>
            <p className="text-sm font-semibold text-zinc-900 truncate">
              {semanticStatus?.model_name || "google/siglip-base"}
            </p>
            <div className="flex items-center gap-1.5 mt-3 text-xs text-zinc-500">
              <span className={`w-1.5 h-1.5 rounded-full ${semanticStatus?.clip_enabled !== false ? "bg-indigo-500" : "bg-zinc-300"}`} />
              {semanticStatus?.clip_enabled !== false ? "SigLIP active" : "Attribute mode"}
            </div>
          </div>

          <div className="app-surface p-5 hover:shadow-md transition-shadow">
            <p className="text-xs font-medium text-zinc-400 mb-2">Engine status</p>
            <p className="text-lg font-semibold text-zinc-900 flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${isOnline ? "bg-emerald-500" : "bg-rose-500"}`} />
              {isOnline ? "Operational" : "Offline"}
            </p>
            <p className="text-xs text-zinc-400 mt-1.5">FastAPI backend</p>
          </div>
        </section>

        {/* Bento module grid */}
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {MODULES.map(({ href, icon: Icon, title, desc, cta, big }) => (
            <Link
              key={href}
              href={href}
              className={`group relative bg-white border border-zinc-200/80 rounded-[1.25rem] p-5 md:p-6 flex flex-col justify-between shadow-sm hover:border-indigo-200 hover:shadow-md hover:-translate-y-0.5 transition-all duration-300 ${
                big ? "sm:col-span-2" : ""
              }`}
            >
              <div>
                <div
                  className="w-10 h-10 rounded-xl bg-indigo-50 border border-indigo-100 flex items-center justify-center mb-4 group-hover:bg-indigo-100 transition-colors"
                >
                  <Icon className="w-4 h-4 text-indigo-600" />
                </div>
                <h3 className="font-semibold text-zinc-900 text-[15px] mb-1.5">{title}</h3>
                <p className="text-sm text-zinc-500 leading-relaxed max-w-xs">{desc}</p>
              </div>
              <div className="mt-5 text-sm font-medium text-zinc-900 flex items-center gap-1">
                {cta}
                <span className="transition-transform group-hover:translate-x-1">→</span>
              </div>
            </Link>
          ))}
        </section>

        {/* Recent activity */}
        <section className="app-surface p-5 md:p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-zinc-900 flex items-center gap-2">
              <ClockIcon className="w-4 h-4 text-zinc-400" />
              Recent pipeline jobs
            </h2>
            <Link href="/pipeline" className="text-sm font-medium text-indigo-600 hover:text-indigo-700">
              View all →
            </Link>
          </div>

          {jobs.length === 0 ? (
            <div className="py-10 text-center text-sm text-zinc-400">No video jobs processed yet.</div>
          ) : (
            <div className="divide-y divide-zinc-100">
              {jobs.slice(0, 5).map((job) => (
                <div
                  key={job.job_id}
                  className="py-3.5 flex items-center justify-between gap-4 hover:bg-zinc-50 -mx-2 px-2 rounded-xl transition-colors"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <span
                      className={`shrink-0 w-2 h-2 rounded-full ${
                        job.status === "completed"
                          ? "bg-emerald-500"
                          : job.status === "processing"
                          ? "bg-amber-400"
                          : "bg-zinc-300"
                      }`}
                    />
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-zinc-800 truncate">
                        {job.source_name || job.filename || "Video Stream"}
                      </p>
                      <p className="text-xs text-zinc-400">ID: {job.job_id}</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 shrink-0">
                    <span
                      className={`px-2.5 py-1 rounded-full text-xs font-medium ${
                        job.status === "completed"
                          ? "bg-emerald-50 text-emerald-700"
                          : job.status === "processing"
                          ? "bg-amber-50 text-amber-700"
                          : "bg-zinc-100 text-zinc-500"
                      }`}
                    >
                      {job.status}
                    </span>
                    <Link
                      href={`/surveillance?jobId=${job.job_id}`}
                      className="text-sm font-medium text-zinc-500 hover:text-zinc-900 flex items-center gap-1"
                    >
                      <EyeIcon className="w-3.5 h-3.5" />
                      Watch
                    </Link>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
