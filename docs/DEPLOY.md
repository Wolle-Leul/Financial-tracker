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

**If `CORS_ORIGINS` and `SESSION_SAME_SITE=none` are already correct but login still loops:** modern browsers often block **cross-site cookies** even when CORS is right. The API and SPA also support a **signed Bearer token**: `POST /auth/login` returns `{"ok":true,"token":"..."}` and the SPA stores it in `sessionStorage` and sends `Authorization: Bearer …` on later requests. Deploy the latest API + rebuild/upload the SPA so this path is active — you do not need to change Render env for the token.

If you still see “session cookie” errors: check the browser is not blocking **third‑party cookies** for the API (try another browser or turn off strict blocking for a test).

## Render (API) — build, start command, exit 127 / exit 1

**Build command** (set explicitly if Render asks):

```bash
pip install -r requirements.txt
```

If logs show **`Form data requires "python-multipart"`**, the build must install dependencies from `requirements.txt` (that package is listed there for `POST /api/imports/pdf`).

Run it from the **repository root** (the folder that contains `requirements.txt` and `finance_tracker/`).

### Database migrations (PostgreSQL)

The API runs **Alembic `upgrade head` on startup** (unless `SKIP_DB_MIGRATIONS=1`). If you deployed new code before migrations existed, you may have seen errors like `column salary_rules.budget_strategy does not exist` — redeploy the latest API; startup also runs an **idempotent `ensure_schema`** that adds `budget_strategy` and the `income_sources` table when missing.

To run migrations manually on Render (shell):

```bash
cd $RENDER_PROJECT_ROOT   # repo root
PYTHONPATH=. alembic upgrade head
```

### Start command (use this — avoids `run_api.py` path issues)

Prefer loading the app from the package (works even if **Root Directory** or missing `run_api.py` confuses Render):

```bash
PYTHONPATH=. python -m uvicorn finance_tracker.api.main:app --host 0.0.0.0 --port $PORT
```

**Alternative** (only if the repo root is the service root and [`run_api.py`](../run_api.py) is present):

```bash
PYTHONPATH=. python -m uvicorn run_api:app --host 0.0.0.0 --port $PORT
```

Do **not** use bare `uvicorn ...` on Render unless you know the venv `bin` is on `PATH`.

### Troubleshooting

| Log / exit | Meaning |
|------------|--------|
| **`uvicorn: command not found`** (exit **127**) | Uvicorn not installed → run **`pip install -r requirements.txt`** in build; use **`python -m uvicorn`**. |
| **Exited with status 1** while running | Uvicorn started but **failed to import the app** (wrong **Root Directory**, missing file, or Python error). Fix **Root Directory** to repo root, or use **`finance_tracker.api.main:app`** above. Check **Deploy logs** for `ModuleNotFoundError` or `Traceback`. |
| **Build failed** | Open the **Build** log — fix `pip` errors (network, conflicting pins). |

**Sanity check** (same as Render’s import step; run locally with repo root as cwd):

```bash
set PYTHONPATH=.
python -c "from finance_tracker.api.main import app; print('OK', app.title)"
```

After a good deploy, logs should show Uvicorn listening. `POST /auth/login` should return JSON with a **`token`** field for the Hostinger SPA.

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
python -m uvicorn run_api:app --host 0.0.0.0 --port 8000
```

Use HTTPS in front of Uvicorn (reverse proxy or platform TLS). Session cookies should be sent only over HTTPS in production; you may set `https_only=True` on `SessionMiddleware` in code when ready.

## Local development

**Terminal 1 — API**

```bash
set PYTHONPATH=.
set PASSWORD=devpassword
python -m uvicorn run_api:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — SPA (uses Vite proxy to the API)**

```bash
cd web
npm run dev
```

Open `http://localhost:5173`. The Vite dev server proxies `/auth`, `/api`, `/health`, and `/version` to `http://127.0.0.1:8000`, so cookies stay on the same browser origin during development.

## Legacy Streamlit UI

The Streamlit entrypoint [`app.py`](../app.py) remains available for local use; production is intended to be the SPA + API.
