# Security Policy

## Supported Versions

Security fixes target the default branch and the latest published release.

## Reporting a Vulnerability

Do not open a public issue for a suspected vulnerability. Send a private report to the repository maintainer with reproduction steps, affected endpoints, impact, and any suggested mitigation.

Until a report is triaged, do not share uploaded footage, embeddings, credentials, or generated media in an issue or pull request.

## Deployment Requirements

- Set unique, high-entropy `MOT_REID_AUTH_PASSWORD` and `MOT_REID_AUTH_SECRET` values.
- Use HTTPS and set `COOKIE_SECURE=true`.
- Keep `data/` outside public/static hosting and restrict filesystem permissions.
- Review retention and consent requirements before processing biometric embeddings or enabling Groq RAG.
