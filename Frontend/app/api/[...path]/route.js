import { auth } from "@clerk/nextjs/server";

export const runtime = "nodejs";
export const maxDuration = 300;

const API_BASE_URL = process.env.API_BASE_URL || "http://127.0.0.1:8000";
const PROXY_SECRET = process.env.MOT_REID_INTERNAL_PROXY_SECRET || process.env.BACKEND_INTERNAL_PROXY_SECRET || "";

const PUBLIC_PATH_PREFIXES = [
  "auth/login",
  "auth/signup",
  "health",
  "metrics",
  "media",
];

async function proxy(request, context) {
  const params = await context.params;
  const pathParts = params.path || [];
  const path = pathParts.join("/");
  const incomingUrl = new URL(request.url);
  const targetUrl = new URL(`/${path}${incomingUrl.search}`, API_BASE_URL);

  const isPublicRoute = PUBLIC_PATH_PREFIXES.some(prefix => path === prefix || path.startsWith(`${prefix}/`));

  let userId = null;
  if (process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY) {
    try {
      const authResult = await auth();
      userId = authResult?.userId || null;
    } catch {
      userId = null;
    }
  }

  const cookieHeader = request.headers.get("cookie") || "";
  const authHeader = request.headers.get("authorization") || "";
  const hasNativeAuth = cookieHeader.includes("mot_reid_access_token=") || authHeader.toLowerCase().startsWith("bearer ");

  if (!isPublicRoute && !userId && !hasNativeAuth) {
    return Response.json({ detail: "Login required." }, { status: 401 });
  }

  const headers = new Headers(request.headers);
  headers.delete("host");
  headers.delete("content-length");

  if (userId) {
    headers.set("x-mot-reid-proxy-secret", PROXY_SECRET);
    headers.set("x-mot-reid-user-id", userId);
  }

  const init = {
    method: request.method,
    headers,
    cache: "no-store",
  };

  if (!["GET", "HEAD"].includes(request.method)) {
    init.body = request.body;
    init.duplex = "half";
  }

  const response = await fetch(targetUrl, init);
  const responseHeaders = new Headers(response.headers);
  responseHeaders.delete("content-encoding");
  responseHeaders.delete("content-length");

  if (typeof response.headers.getSetCookie === "function") {
    const cookies = response.headers.getSetCookie();
    if (cookies.length > 0) {
      responseHeaders.delete("set-cookie");
      cookies.forEach((cookie) => responseHeaders.append("set-cookie", cookie));
    }
  }

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: responseHeaders,
  });
}

export async function GET(request, context) {
  return proxy(request, context);
}

export async function POST(request, context) {
  return proxy(request, context);
}

export async function PUT(request, context) {
  return proxy(request, context);
}

export async function DELETE(request, context) {
  return proxy(request, context);
}

export async function PATCH(request, context) {
  return proxy(request, context);
}

