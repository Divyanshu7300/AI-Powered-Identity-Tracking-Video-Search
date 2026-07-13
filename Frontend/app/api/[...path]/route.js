export const runtime = "nodejs";
export const maxDuration = 300;

const API_BASE_URL = process.env.API_BASE_URL || "http://127.0.0.1:8000";

async function proxy(request, context) {
  const params = await context.params;
  const path = (params.path || []).join("/");
  const incomingUrl = new URL(request.url);
  const targetUrl = new URL(`/${path}${incomingUrl.search}`, API_BASE_URL);

  const headers = new Headers(request.headers);
  headers.delete("host");
  headers.delete("content-length");

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
