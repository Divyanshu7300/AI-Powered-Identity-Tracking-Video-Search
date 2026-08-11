"use client";

import { SignUp } from "@clerk/nextjs";

export default function SignUpPage() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-[#f7f8fc] p-6">
      <SignUp
        routing="path"
        path="/sign-up"
        signInUrl="/login"
        fallbackRedirectUrl="/"
        appearance={{
          variables: { colorPrimary: "#635bff", borderRadius: "0.875rem" },
          elements: { card: "border border-zinc-200 bg-white shadow-sm", formButtonPrimary: "rounded-xl bg-zinc-900 hover:bg-zinc-800" },
        }}
      />
    </main>
  );
}
