import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";

const isDashboard = createRouteMatcher(["/"]);
const hasClerk = Boolean(process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY);

export default async function proxy(request, event) {
  if (!hasClerk) {
    if (isDashboard(request)) {
      const token = request.cookies.get("mot_reid_access_token")?.value;
      if (!token) {
        return NextResponse.redirect(new URL("/login", request.url));
      }
    }
    return NextResponse.next();
  }

  const clerkHandler = clerkMiddleware(async (auth, req) => {
    if (isDashboard(req)) {
      await auth.protect({
        unauthenticatedUrl: new URL("/login", req.url).toString(),
      });
    }
  });

  return clerkHandler(request, event);
}

export const config = {
  matcher: ["/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico)).*)"],
};
