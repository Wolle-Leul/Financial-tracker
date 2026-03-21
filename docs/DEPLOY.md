# Deployment: SPA (Hostinger) + FastAPI API (separate host)

The app is split into two deployable units:

1. **Frontend** — Vite + React in [`web/`](../web/). Build output is static files in `web/dist/`, suitable for Hostinger Node/static hosting.
2. **Backend** — FastAPI in [`finance_tracker/api/`](../finance_tracker/api/). Run with Uvicorn on any host that supports Python (Render, Railway, Fly.io, a VPS, etc.).

## Environment variables (API)

| Variable | Purpose |
|----------|---------|
| `DB_URL` | SQLAlchemy URL (e.g. PostgreSQL or SQLite file path). |
| `PASSWORD_HASH` | Bcrypt hash for the dashboard password (recommended). |
| `PASSWORD` | Plaintext password (dev only; prefer `PASSWORD_HASH`). |
| `SESSION_SECRET` | Secret for signing session cookies (set a long random string in production). |
| `CORS_ORIGINS` | Comma-separated list of allowed browser origins, e.g. `https://yourdomain.com,https://www.yourdomain.com`. |
| `SESSION_SAME_SITE` | Set to **`none`** when the SPA is on a **different domain** than the API (e.g. Hostinger + Render). Required so the browser sends the session cookie on cross-origin `fetch` requests. Use HTTPS on both sides. If unset, the API defaults to `lax` (fine for local dev with the Vite proxy only). |

If `CORS_ORIGINS` is unset, the API defaults to `http://localhost:5173` and `http://127.0.0.1:5173` for local development.

**Hostinger + Render (required for login to stick):**

1. [Render](https://dashboard.render.com) → your **Web Service** (the API) → **Environment**.
2. Add **`SESSION_SAME_SITE`** = **`none`** (lowercase word `none`, no quotes).
3. Add **`CORS_ORIGINS`** = your SPA origin only, e.g. `https://something.hostingersite.com` — copy from the browser address bar (HTTPS, no path; a trailing slash is OK in the env var; the API normalizes it).
4. **Save**, then **Manual Deploy** (or push to the connected branch) so the service restarts.
5. After deploy, open **Logs** and confirm a line like `session cookie: same_site=none https_only=True` on startup.

If you still see “session cookie” errors: check the browser is not blocking **third‑party cookies** for the API (try another browser or turn off strict blocking for a test).

## Hostinger (frontend)

1. Build: `cd web && npm ci && npm run build`.
2. Deploy `web/dist/` contents to your site root (or follow Hostinger’s Node/Vite deployment flow if you use their Git integration).
3. Set **`VITE_API_BASE_URL`** at **build time** to your public API base URL, e.g. `https://api.yourdomain.com` (no trailing slash). Rebuild after changing it.
4. For **client-side routing**, configure SPA fallback so unknown paths serve `index.html` (Hostinger docs vary by plan; Apache-style `FallbackResource` / rewrite rules are common).

## API host (example: Uvicorn)

```bash
export DB_URL="postgresql+psycopg://..."
export PASSWORD_HASH='...'
export SESSION_SECRET='...'
export CORS_ORIGINS='https://your-spa-hostinger-domain.com'
uvicorn run_api:app --host 0.0.0.0 --port 8000
```

Use HTTPS in front of Uvicorn (reverse proxy or platform TLS). Session cookies should be sent only over HTTPS in production; you may set `https_only=True` on `SessionMiddleware` in code when ready.

## Local development

**Terminal 1 — API**

```bash
set PYTHONPATH=.
set PASSWORD=devpassword
uvicorn run_api:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — SPA (uses Vite proxy to the API)**

```bash
cd web
npm run dev
```

Open `http://localhost:5173`. The Vite dev server proxies `/auth`, `/api`, `/health`, and `/version` to `http://127.0.0.1:8000`, so cookies stay on the same browser origin during development.

## Legacy Streamlit UI

The Streamlit entrypoint [`app.py`](../app.py) remains available for local use; production is intended to be the SPA + API.
