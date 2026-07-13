"use client";

export default function GlobalError({ error, reset }) {
  return (
    <html lang="en">
      <body>
        <main className="shell">
          <section className="panel">
            <div className="panel-head">
              <h1>Dashboard error</h1>
            </div>
            <p className="message">{error?.message || "Something went wrong."}</p>
            <button className="primary" type="button" onClick={reset}>
              Retry
            </button>
          </section>
        </main>
      </body>
    </html>
  );
}
