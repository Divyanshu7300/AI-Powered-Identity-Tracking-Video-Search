const SESSION_STORAGE_KEY = "mot-reid-session-id";
const AUTH_STATE_KEY = "mot-reid-authenticated";
const USERNAME_STORAGE_KEY = "mot-reid-username";

export function getAccessToken() {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem(AUTH_STATE_KEY) || "";
}

export function clearAccessToken() {
  if (typeof window !== "undefined") {
    window.localStorage.removeItem(AUTH_STATE_KEY);
    window.localStorage.removeItem(USERNAME_STORAGE_KEY);
  }
}

export function getUsername() {
  if (typeof window === "undefined") return "";
  const stored = window.localStorage.getItem(USERNAME_STORAGE_KEY) || "";
  if (!stored || /^(user_|usr_|session_|anon_|[0-9a-f]{8}-[0-9a-f]{4}-|[a-z0-9]{20,})/i.test(stored)) {
    return "";
  }
  return stored;
}

export function setAuthenticated(username) {
  window.localStorage.setItem(AUTH_STATE_KEY, "true");
  if (username && !/^(user_|usr_|session_|anon_|[0-9a-f]{8}-[0-9a-f]{4}-|[a-z0-9]{20,})/i.test(username)) {
    window.localStorage.setItem(USERNAME_STORAGE_KEY, username);
  }
}

export async function logout() {
  try {
    await fetch("/api/auth/logout", { method: "POST" });
  } catch {
    // ignore network errors
  }
  clearAccessToken();
}

export async function fetchCurrentIdentity() {
  try {
    const response = await fetch("/api/auth/me", { cache: "no-store" });
    if (response.ok) {
      const data = await response.json();
      if (data?.username && !/^(user_|usr_|session_|anon_|[0-9a-f]{8}-[0-9a-f]{4}-|[a-z0-9]{20,})/i.test(data.username)) {
        setAuthenticated(data.username);
        return data;
      }
    }
  } catch {}
  return null;
}


export async function login(username, password) {
  const response = await fetch("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  const payload = await readJson(response);
  if (!response.ok) throw new Error(payload.detail || "Login failed.");
  setAuthenticated(payload.username);
  return payload;
}

export async function signup(username, password, passwordConfirmation) {
  const response = await fetch("/api/auth/signup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password, password_confirmation: passwordConfirmation }),
  });
  const payload = await readJson(response);
  if (!response.ok) throw new Error(payload.detail || "Unable to create account.");
  setAuthenticated(payload.username);
  return payload;
}

export function resetSessionId() {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(SESSION_STORAGE_KEY, crypto.randomUUID());
}

export async function readJson(response) {
  const text = await response.text();
  try {
    return text ? JSON.parse(text) : {};
  } catch {
    return { message: text || response.statusText };
  }
}

export async function apiRequest(path, options = {}) {
  const headers = new Headers(options.headers);
  const response = await fetch(path, { ...options, headers, cache: "no-store" });
  const payload = await readJson(response);
  if (!response.ok) {
    if (response.status === 401) clearAccessToken();
    let errMsg = "Request failed.";
    if (typeof payload.detail === "string") {
      errMsg = payload.detail;
    } else if (typeof payload.message === "string") {
      errMsg = payload.message;
    } else if (Array.isArray(payload.detail)) {
      errMsg = payload.detail.map((d) => d.msg || d.detail || JSON.stringify(d)).join(", ");
    } else if (payload.detail && typeof payload.detail === "object") {
      errMsg = payload.detail.msg || payload.detail.message || JSON.stringify(payload.detail);
    } else if (typeof payload === "string") {
      errMsg = payload;
    }
    throw new Error(errMsg);
  }
  return payload;
}

export async function loadDashboard() {
  const [health, dashboard, tracks, jobs] = await Promise.all([
    fetch("/api/health", { cache: "no-store" }),
    apiRequest("/api/tracking/analytics/dashboard"),
    apiRequest("/api/tracking/analytics/tracks"),
    apiRequest("/api/tracking/jobs"),
  ]);

  return {
    health: health.ok ? "online" : "offline",
    dashboard,
    tracks: tracks.track_memories || [],
    jobs: jobs.jobs || [],
  };
}
