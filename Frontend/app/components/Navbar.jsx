"use client";

import React, { useState, useRef, useEffect, useMemo } from "react";
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
  const [nativeUser, setNativeUser] = useState("");
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    fetchCurrentIdentity().then((identity) => {
      if (identity?.username) setNativeUser(identity.username);
    });
  }, []);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(event) {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setUserMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const effectiveIsSignedIn = Boolean(isSignedIn || nativeUser || getUsername());

  const rawUser = user?.fullName || user?.primaryEmailAddress?.emailAddress || user?.username || nativeUser || getUsername();
  const username = useMemo(() => {
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
        {/* Brand Logo */}
        <Link href="/" className="shrink-0" title="AURA Platform Home">
          <AuraLogo />
        </Link>

        {/* Desktop Navigation Links with Icon + Text */}
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
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  isActive ? "bg-white text-zinc-900 shadow-sm" : "text-zinc-500 hover:text-zinc-900 hover:bg-white/70"
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? "text-indigo-500" : ""}`} />
                <span>{item.name}</span>
              </Link>
            );
          })}
        </nav>

        {/* User Profile Badge & Dropdown */}
        <div className="relative shrink-0" ref={menuRef}>
          <button
            type="button"
            onClick={() => setUserMenuOpen((prev) => !prev)}
            className="flex items-center gap-2.5 p-1.5 pl-2.5 rounded-full bg-zinc-50 hover:bg-zinc-100 border border-zinc-200/80 transition-all cursor-pointer group"
          >
            <span
              className="w-2 h-2 rounded-full shrink-0"
              style={{ backgroundColor: isOnline ? "#10B981" : "#F43F5E" }}
              title={isOnline ? "Engine online" : "Engine offline"}
            />
            <span className="text-xs font-semibold text-zinc-800 max-w-[120px] truncate">
              {username}
            </span>
            <div className="w-6 h-6 rounded-full bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center text-white text-[10px] font-bold shrink-0 shadow-xs">
              {initial}
            </div>
            <svg
              className={`w-3.5 h-3.5 text-zinc-400 group-hover:text-zinc-600 transition-transform duration-200 ${
                userMenuOpen ? "rotate-180" : ""
              }`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {/* User Dropdown Menu */}
          {userMenuOpen && (
            <div className="absolute right-0 mt-2 w-56 bg-white border border-zinc-200/80 rounded-2xl shadow-xl py-2 z-50 animate-in fade-in slide-in-from-top-2 duration-150 space-y-1">
              <div className="px-4 py-2.5 border-b border-zinc-100">
                <p className="text-xs font-semibold text-zinc-900 truncate">{username}</p>
                <p className="text-[11px] text-zinc-400 font-medium flex items-center gap-1.5 mt-0.5">
                  <span className={`w-1.5 h-1.5 rounded-full ${isOnline ? "bg-emerald-500" : "bg-rose-500"}`} />
                  <span>{isOnline ? "Engine online" : "Engine offline"}</span>
                </p>
              </div>

              <div className="px-1">
                <Link
                  href="/settings"
                  onClick={() => setUserMenuOpen(false)}
                  className="flex items-center gap-2 px-3 py-2 text-xs font-medium text-zinc-700 hover:bg-zinc-50 rounded-xl transition-colors"
                >
                  <SettingsIcon className="w-4 h-4 text-zinc-400" />
                  <span>Account Settings</span>
                </Link>
                <button
                  type="button"
                  onClick={() => {
                    setUserMenuOpen(false);
                    handleLogout();
                  }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-xs font-semibold text-rose-600 hover:bg-rose-50 rounded-xl transition-colors cursor-pointer text-left"
                >
                  <svg className="w-4 h-4 text-rose-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                  </svg>
                  <span>Log out</span>
                </button>
              </div>
            </div>
          )}
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
