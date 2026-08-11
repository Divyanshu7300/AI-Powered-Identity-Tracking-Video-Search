"use client";

import React from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAuth, useUser } from "@clerk/nextjs";
import { useQuery } from "@tanstack/react-query";
import { fetchCurrentIdentity, getUsername, loadDashboard as fetchDashboard, logout as nativeLogout } from "../lib/api";
import AuraLogo from "./AuraLogo";
import {
  DashboardIcon,
  SearchIcon,
  VideoIcon,
  LayersIcon,
  CpuIcon,
  SettingsIcon,
} from "./Icons";

export default function Navbar() {
  const pathname = usePathname();
  const router = useRouter();
  const { isSignedIn, signOut } = useAuth();
  const { user } = useUser();
  const [nativeUser, setNativeUser] = React.useState("");

  React.useEffect(() => {
    fetchCurrentIdentity().then((identity) => {
      if (identity?.username) setNativeUser(identity.username);
    });
  }, []);

  const effectiveIsSignedIn = Boolean(isSignedIn || nativeUser || getUsername());

  const rawUser = user?.fullName || user?.primaryEmailAddress?.emailAddress || user?.username || nativeUser || getUsername();
  const username = React.useMemo(() => {
    if (!rawUser) return "Operator";
    if (/^(user_|usr_|session_|anon_|[0-9a-f]{8}-[0-9a-f]{4}-|[a-z0-9]{20,})/i.test(rawUser)) {
      return "Operator";
    }
    if (rawUser.includes("@")) return rawUser.split("@")[0];
    return rawUser;
  }, [rawUser]);

  const { data: dashboardData, error: dashboardError } = useQuery({
    queryKey: ["dashboard"],
    queryFn: fetchDashboard,
    staleTime: 5_000,
    refetchInterval: 10_000,
    enabled: effectiveIsSignedIn,
  });

  const health = dashboardError ? "offline" : dashboardData?.health || "checking";
  const isOnline = health === "online";

  async function handleLogout() {
    if (isSignedIn && signOut) {
      try {
        await signOut({ redirectUrl: "/login" });
      } catch {}
    }
    await nativeLogout();
    router.replace("/login");
  }

  const navItems = [
    { name: "Overview", href: "/dashboard", icon: DashboardIcon },
    { name: "Pipeline", href: "/pipeline", icon: CpuIcon },
    { name: "Search", href: "/search", icon: SearchIcon },
    { name: "Surveillance", href: "/surveillance", icon: VideoIcon },
    { name: "Tracks", href: "/tracks", icon: LayersIcon },
    { name: "Settings", href: "/settings", icon: SettingsIcon },
  ];

  if (["/login", "/sign-up", "/sso-callback"].includes(pathname)) {
    return null;
  }

  const initial = (username || "O").trim().charAt(0).toUpperCase();

  return (
    <header className="sticky top-0 z-50 w-full bg-white/80 backdrop-blur-xl border-b border-zinc-200/70">
      <div className="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between gap-6">
        {/* Brand */}
        <Link href="/" className="shrink-0" title="AURA Platform & Architecture Home">
          <AuraLogo />
        </Link>

        <nav className="hidden md:flex items-center gap-1 p-1 rounded-xl bg-zinc-50/80 border border-zinc-100">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href || (item.href !== "/dashboard" && pathname.startsWith(item.href));
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-label={item.name}
                title={item.name}
                className={`flex items-center justify-center w-8 h-8 rounded-lg text-sm font-medium transition-all ${
                  isActive ? "bg-white text-zinc-900 shadow-sm" : "text-zinc-400 hover:text-zinc-900 hover:bg-white/70"
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? "text-indigo-500" : ""}`} />
              </Link>
            );
          })}
        </nav>

        {/* Right Controls — condensed into a single row */}
        <div className="flex items-center gap-2.5 shrink-0">
          <span
            className="hidden sm:inline-flex w-1.5 h-1.5 rounded-full shrink-0"
            style={{ backgroundColor: isOnline ? "#10B981" : "#F43F5E" }}
            title={isOnline ? "Engine ready" : "Offline"}
          />

          <button
            type="button"
            onClick={handleLogout}
            className="hidden lg:block text-xs font-medium text-zinc-500 hover:text-rose-600 transition-colors"
          >
            Log out
          </button>

          <div
            className="w-7 h-7 rounded-full bg-gradient-to-br from-indigo-400 to-violet-500 flex items-center justify-center text-white text-[11px] font-semibold shrink-0"
            title={username}
          >
            {initial}
          </div>
        </div>
      </div>

      {/* Mobile Navigation */}
      <div className="md:hidden flex items-center gap-1.5 overflow-x-auto px-5 py-2 border-t border-zinc-100 no-scrollbar">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href || (item.href !== "/dashboard" && pathname.startsWith(item.href));
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium shrink-0 transition-all ${
                isActive ? "bg-zinc-100 text-zinc-900" : "text-zinc-500"
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {item.name}
            </Link>
          );
        })}
      </div>
    </header>
  );
}
