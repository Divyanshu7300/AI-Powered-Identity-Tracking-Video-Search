"use client";

export default function GlobalError({ error, reset }) {
  return (
    <html lang="en">
      <body>
        <main className="min-h-screen bg-[#f7f8fc] grid place-items-center p-6 text-zinc-900">
          <section className="app-surface w-full max-w-md p-6 text-center">
            <div className="mb-3">
              <h1 className="text-lg font-semibold tracking-tight">Something went wrong</h1>
            </div>
            <p className="text-sm leading-6 text-zinc-500 mb-5">{error?.message || "Please retry loading this page."}</p>
            <button className="rounded-xl bg-zinc-900 px-4 py-2.5 text-sm font-medium text-white hover:bg-zinc-800 transition-colors" type="button" onClick={reset}>
              Retry
            </button>
          </section>
        </main>
      </body>
    </html>
  );
}
