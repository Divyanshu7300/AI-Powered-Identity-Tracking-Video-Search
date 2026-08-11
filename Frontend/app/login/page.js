"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useSignIn, useSignUp } from "@clerk/nextjs/legacy";
import { login as nativeLogin, signup as nativeSignup } from "../lib/api";
import AuraLogo from "../components/AuraLogo";

function ViewfinderMark() {
  return <div className="auth-brand"><AuraLogo /></div>;
}

function TelemetryPanel() {
  return <section className="relative hidden min-h-screen overflow-hidden bg-zinc-950 p-10 text-white lg:flex lg:w-[52%] lg:flex-col lg:justify-between"><div className="pointer-events-none absolute inset-0 opacity-40 [background-image:linear-gradient(rgba(255,255,255,.045)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,.045)_1px,transparent_1px)] [background-size:42px_42px]" /><div className="pointer-events-none absolute -right-32 top-20 h-96 w-96 rounded-full bg-orange-500/15 blur-3xl" /><div className="relative z-10 flex items-center gap-3"><ViewfinderMark /><div><div className="text-sm font-semibold tracking-tight">AURA</div><div className="font-mono text-[10px] uppercase tracking-[0.22em] text-zinc-500">Vision intelligence engine</div></div></div><div className="relative z-10 max-w-lg space-y-8"><div className="space-y-4"><div className="inline-flex items-center gap-2 rounded-full border border-orange-500/25 bg-orange-500/10 px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.16em] text-orange-300"><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-orange-400" />Secure operator gateway</div><h1 className="max-w-xl text-4xl font-semibold leading-[1.08] tracking-[-0.04em] text-zinc-100 xl:text-6xl">Find the frame. <span className="text-zinc-500">Follow the signal.</span></h1><p className="max-w-md text-sm leading-6 text-zinc-400">Search, track, and review visual evidence through one focused intelligence workspace.</p></div><div className="grid max-w-md grid-cols-3 gap-3">{["TRACKING", "RE-ID", "RAG SEARCH"].map((label, index) => <div key={label} className="rounded-xl border border-white/10 bg-white/[0.04] p-3"><div className="mb-2 font-mono text-[9px] text-zinc-600">0{index + 1}</div><div className="font-mono text-[10px] font-semibold tracking-wider text-zinc-300">{label}</div></div>)}</div></div><div className="relative z-10 flex justify-between font-mono text-[10px] uppercase tracking-wider text-zinc-600"><span>Private session / encrypted transport</span><span>v1.0.0</span></div></section>;
}

export default function LoginPage() {
  const router = useRouter();
  const { isLoaded: signInLoaded, signIn, setActive: setSignInActive } = useSignIn();
  const { isLoaded: signUpLoaded, signUp, setActive: setSignUpActive } = useSignUp();
  const [mode, setMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [code, setCode] = useState("");
  const [verificationPending, setVerificationPending] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const existing = document.getElementById("clerk-captcha");
    if (existing) return undefined;
    const captcha = document.createElement("div");
    captcha.id = "clerk-captcha";
    captcha.className = "sr-only";
    document.body.appendChild(captcha);
    return () => captcha.remove();
  }, []);

  useEffect(() => {
    document.body.dataset.page = "auth";
    return () => { delete document.body.dataset.page; };
  }, []);

  const message = (err) => err?.errors?.[0]?.longMessage || err?.errors?.[0]?.message || err?.message || "Authentication could not be completed.";
  const complete = async (result, setActive) => { await setActive({ session: result.createdSessionId }); router.replace("/"); };

  async function submit(event) {
    event.preventDefault(); setBusy(true); setError("");
    const isClerkAvailable = Boolean(signInLoaded && signIn);

    try {
      if (isClerkAvailable) {
        if (mode === "login") {
          const result = await signIn.create({ identifier: email, password });
          if (result.status === "complete") await complete(result, setSignInActive); else setError("Complete the required sign-in step.");
        } else if (verificationPending) {
          const result = await signUp.attemptEmailAddressVerification({ code });
          if (result.status === "complete") await complete(result, setSignUpActive); else setError("Verification is not complete yet.");
        } else {
          await signUp.create({ emailAddress: email, password });
          await signUp.prepareEmailAddressVerification({ strategy: "email_code" });
          setVerificationPending(true);
        }
      } else {
        // Native Backend Auth Fallback
        const username = email.includes("@") ? email.split("@")[0] : email;
        if (mode === "login") {
          await nativeLogin(username, password);
          router.replace("/");
        } else {
          await nativeSignup(username, password, password);
          router.replace("/");
        }
      }
    } catch (err) {
      if (isClerkAvailable) {
        // If Clerk fails (e.g. Clerk's strict 15-char policy), fall back to Native Auth
        try {
          const username = email.includes("@") ? email.split("@")[0] : email;
          if (mode === "login") {
            await nativeLogin(username, password);
            router.replace("/");
            return;
          } else {
            await nativeSignup(username, password, password);
            router.replace("/");
            return;
          }
        } catch {}
      }
      setError(message(err));
    } finally { setBusy(false); }
  }

  async function social(strategy) {
    setError("");
    if (signInLoaded && signIn) {
      try {
        await signIn.authenticateWithRedirect({ strategy, redirectUrl: "/sso-callback", redirectUrlComplete: "/" });
        return;
      } catch (err) {
        setError(message(err));
        return;
      }
    }
    setError("Social single sign-on requires Clerk authentication.");
  }

  const canSubmit = Boolean(email.trim() && (verificationPending ? code.trim() : password.trim()));
  const signingUp = mode === "signup";

  return <main className="flex min-h-screen bg-[#f8fafc] text-zinc-900"><TelemetryPanel /><section className="flex w-full items-center justify-center p-6 sm:p-10 lg:w-[48%]"><div className="w-full max-w-md"><div className="mb-10 flex items-center gap-3 lg:hidden"><ViewfinderMark /><div><div className="text-sm font-semibold tracking-tight">AURA</div><div className="font-mono text-[10px] uppercase tracking-[.18em] text-zinc-400">Operator access</div></div></div><div className="mb-8 space-y-2"><div className="font-mono text-[10px] font-semibold uppercase tracking-[.2em] text-orange-600">Access console</div><h2 className="text-3xl font-semibold tracking-[-.035em] text-zinc-950">{signingUp ? "Create your AURA account" : "Sign in to AURA"}</h2><p className="text-sm leading-6 text-zinc-500">{verificationPending ? "Enter the verification code sent to your email." : "Use email, Google, or Microsoft to access your private workspace."}</p></div><form onSubmit={submit} className="space-y-5 rounded-2xl border border-zinc-200 bg-white p-6 shadow-[0_20px_60px_-35px_rgba(24,24,27,.35)] sm:p-7">{!verificationPending && <><button type="button" onClick={() => social("oauth_google")} disabled={busy} className="flex w-full items-center justify-center gap-2 rounded-xl border border-zinc-200 bg-white px-4 py-3 text-sm font-semibold text-zinc-700 transition hover:bg-zinc-50 disabled:opacity-50"><span className="text-base font-bold text-[#4285f4]">G</span>Continue with Google</button><button type="button" onClick={() => social("oauth_microsoft")} disabled={busy} className="flex w-full items-center justify-center gap-2 rounded-xl border border-zinc-200 bg-white px-4 py-3 text-sm font-semibold text-zinc-700 transition hover:bg-zinc-50 disabled:opacity-50"><span className="grid h-4 w-4 grid-cols-2 gap-px"><i className="bg-[#f25022]" /><i className="bg-[#7fba00]" /><i className="bg-[#00a4ef]" /><i className="bg-[#ffb900]" /></span>Continue with Microsoft</button><div className="relative py-1 text-center before:absolute before:inset-x-0 before:top-1/2 before:border-t before:border-zinc-200"><span className="relative bg-white px-3 font-mono text-[10px] uppercase tracking-wider text-zinc-400">or continue with email</span></div><label className="block space-y-2"><span className="font-mono text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Email or Username</span><input required type="text" autoComplete="username email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" className="w-full rounded-xl border border-zinc-200 bg-zinc-50 px-3.5 py-3 text-sm outline-none transition focus:border-orange-400 focus:bg-white focus:ring-4 focus:ring-orange-500/10" /></label><label className="block space-y-2"><span className="font-mono text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Password</span><input required type="password" minLength={4} autoComplete={signingUp ? "new-password" : "current-password"} value={password} onChange={(e) => setPassword(e.target.value)} placeholder="At least 4 characters" className="w-full rounded-xl border border-zinc-200 bg-zinc-50 px-3.5 py-3 text-sm outline-none transition focus:border-orange-400 focus:bg-white focus:ring-4 focus:ring-orange-500/10" /></label></>}{verificationPending && <label className="block space-y-2"><span className="font-mono text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Verification code</span><input required inputMode="numeric" autoComplete="one-time-code" value={code} onChange={(e) => setCode(e.target.value)} placeholder="Enter email code" className="w-full rounded-xl border border-zinc-200 bg-zinc-50 px-3.5 py-3 text-sm outline-none transition focus:border-orange-400 focus:bg-white focus:ring-4 focus:ring-orange-500/10" /></label>}{error && <div className="rounded-xl border border-rose-200 bg-rose-50 px-3.5 py-3 text-xs font-medium text-rose-700">{error}</div>}<button disabled={!canSubmit} className="flex w-full items-center justify-center rounded-xl bg-zinc-950 px-4 py-3 text-sm font-semibold text-white transition hover:bg-zinc-800 disabled:opacity-60">{busy ? "Please wait..." : verificationPending ? "Verify email" : signingUp ? "Create account" : "Enter workspace"}</button>{!verificationPending && <button type="button" onClick={() => { setMode(signingUp ? "login" : "signup"); setError(""); }} className="w-full text-center text-sm font-medium text-zinc-500 transition hover:text-orange-600">{signingUp ? "Already have an account? Sign in" : "New here? Create an account"}</button>}</form><p className="mt-6 text-center font-mono text-[10px] uppercase tracking-wider text-zinc-400">Authorized operators only</p></div></section></main>;
}
